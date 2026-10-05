"""Evaluate a trained jet tokenizer checkpoint and write per-jet reconstruction plots.

Runs without Hydra/Lightning-Trainer (and without comet_ml), so that the plots of an existing
run can be re-created, e.g.

    python scripts/evaluate_checkpoint_per_jet.py \
        --run-dir  <.../runs/codebook8192> \
        --data-dir <.../data/filtered> \
        --out-dir  <output directory> \
        --subset original_test

Subsets
-------
original_test
    The first ``--n-original-test`` (default 20000) events with >= 1 AK8 jet of every file. This
    is exactly what the "test" split of the April codebook-sweep runs was
    (``dataset_kwargs_test.n_jets_per_file: 20000``). These runs trained on the first 100000
    events of every file, so this subset is a subset of the training data.
held_out
    Events with index >= ``--n-train-per-file`` (default 100000) among the events with >= 1 jet,
    in file order, i.e. events that the April runs never trained on. At most
    ``--max-held-out`` events per class are used. The minbias file has fewer events than that,
    so for minbias all events are used and the result is flagged as "seen in training".

The preprocessing (feature dict) is taken from the checkpoint, so that the model is evaluated
with the preprocessing it was trained with.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gabbro.callbacks.tokenization_callback import _reconstruct_phi_from_cos_sin  # noqa: E402
from gabbro.models.vqvae import VQVAELightning  # noqa: E402
from gabbro.plotting.jet_reconstruction import (  # noqa: E402
    codebook_utilization,
    make_per_jet_plots,
)
from gabbro.utils.arrays import (  # noqa: E402
    ak_pad,
    ak_select_and_preprocess,
    ak_to_np_stack,
    np_to_ak,
)
from gabbro.utils.jet_types import jet_types_dict  # noqa: E402

CLASS_FILES = {
    "l1t_minbias": "minbias_kinematics.parquet",
    "l1t_qcd": "QCD_HT50toInf_kinematics.parquet",
    "l1t_ggHbb": "ggHbb_kinematics.parquet",
    "l1t_VBFHbb": "VBFHbb_kinematics.parquet",
}
COLUMNS = ["L1T_JetPuppiAK8_PT", "L1T_JetPuppiAK8_Eta", "L1T_JetPuppiAK8_Phi"]


def read_events(path, start, stop):
    """Events with >= 1 AK8 jet in file order, rows [start:stop) of the filtered list."""
    table = ak.from_parquet(path, columns=COLUMNS)
    table = table[ak.num(table["L1T_JetPuppiAK8_PT"]) >= 1]
    n_total = len(table)
    table = table[start:stop]
    pt = ak.values_astype(table["L1T_JetPuppiAK8_PT"], "float32")
    eta = ak.values_astype(table["L1T_JetPuppiAK8_Eta"], "float32")
    phi = ak.values_astype(table["L1T_JetPuppiAK8_Phi"], "float32")
    x = ak.Array(
        {
            "part_pt": pt,
            "part_eta": eta,
            "part_phi": phi,
            "part_phi_cos": ak.values_astype(np.cos(phi), "float32"),
            "part_phi_sin": ak.values_astype(np.sin(phi), "float32"),
        }
    )
    return x, n_total


def to_model_input(x_ak, pp_dict, pad_length):
    x_pp = ak_select_and_preprocess(x_ak, pp_dict=pp_dict)
    x_pad, mask = ak_pad(x_pp, maxlen=pad_length, return_mask=True)
    x_np = ak_to_np_stack(x_pad, names=list(pp_dict.keys())).astype("float32")
    return x_np, ak.to_numpy(mask).astype("float32")


@torch.no_grad()
def run_model(model, x_np, mask_np, batch_size, device):
    recos, codes = [], []
    for i in range(0, len(x_np), batch_size):
        xb = torch.from_numpy(x_np[i : i + batch_size]).to(device)
        mb = torch.from_numpy(mask_np[i : i + batch_size]).to(device)
        x_reco, vq_out = model.model(xb, mask=mb)
        recos.append(x_reco.cpu().numpy())
        q = vq_out["q"].cpu().numpy()
        codes.append(q[..., 0] if q.ndim == 3 else q)
    return np.concatenate(recos), np.concatenate(codes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="run folder that contains checkpoints/best.ckpt")
    p.add_argument("--data-dir", required=True, help="folder with the *_kinematics.parquet files")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--subset", choices=["original_test", "held_out"], required=True)
    p.add_argument("--n-original-test", type=int, default=20000)
    p.add_argument("--n-train-per-file", type=int, default=100000)
    p.add_argument("--max-held-out", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt = os.path.join(args.run_dir, "checkpoints", "best.ckpt")
    print(f"Loading {ckpt}")
    model = VQVAELightning.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False)
    model.eval().to(args.device)

    kwargs = model.hparams["model_kwargs"]
    pp_dict = kwargs["input_features_dict"]
    if isinstance(pp_dict, DictConfig):
        pp_dict = OmegaConf.to_container(pp_dict, resolve=True)
    pp_dict = dict(pp_dict)
    n_codes = kwargs["vq_kwargs"]["num_codes"]
    pad_length = 7
    print(f"num_codes={n_codes}, features={list(pp_dict.keys())}")

    x_pp_all, mask_all, label_all = [], [], []
    seen_flags, class_counts = {}, {}
    for name, fname in CLASS_FILES.items():
        path = os.path.join(args.data_dir, fname)
        if args.subset == "original_test":
            start, stop, seen = 0, args.n_original_test, True
        else:
            start, stop, seen = args.n_train_per_file, args.n_train_per_file + args.max_held_out, False
        x_ak, n_total = read_events(path, start, stop)
        if len(x_ak) == 0:
            # e.g. minbias: no events beyond the training range -> use all events (seen in training)
            x_ak, n_total = read_events(path, 0, None)
            seen = True
        seen_flags[name] = seen
        class_counts[name] = {"n_events_file": int(n_total), "n_events_used": int(len(x_ak))}
        print(f"{name}: file has {n_total} events with >= 1 jet, using {len(x_ak)} "
              f"({'seen in training' if seen else 'held out'})")
        x_np, m_np = to_model_input(x_ak, pp_dict, pad_length)
        x_pp_all.append(x_np)
        mask_all.append(m_np)
        label_all.append(np.full(len(x_np), jet_types_dict[name]["label"]))

    x_np = np.concatenate(x_pp_all)
    mask_np = np.concatenate(mask_all)
    labels = np.concatenate(label_all)

    print(f"Running the model on {len(x_np)} events")
    x_reco_np, code_idx = run_model(model, x_np, mask_np, args.batch_size, args.device)

    # reconstruction error in the units the model is trained in (sum over features, mean over jets)
    se = ((x_reco_np - x_np) ** 2 * mask_np[..., None]).sum()
    recon_loss = float(se / mask_np.sum())

    names = list(pp_dict.keys())
    x_reco_ak = ak_select_and_preprocess(np_to_ak(x_reco_np, mask=mask_np, names=names), pp_dict=pp_dict, inverse=True)
    x_orig_ak = ak_select_and_preprocess(np_to_ak(x_np, mask=mask_np, names=names), pp_dict=pp_dict, inverse=True)
    x_reco_ak = _reconstruct_phi_from_cos_sin(x_reco_ak)
    x_orig_ak = _reconstruct_phi_from_cos_sin(x_orig_ak)

    os.makedirs(args.out_dir, exist_ok=True)
    k_label = f"K={n_codes}"
    metrics = make_per_jet_plots(
        x_orig_ak, x_reco_ak, labels, jet_types_dict, args.out_dir, "test",
        title_suffix=f"({k_label})",
    )
    util = codebook_utilization(code_idx, mask_np, n_codes)
    summary = {
        "checkpoint": ckpt,
        "subset": args.subset,
        "num_codes": n_codes,
        "features": names,
        "reconstruction_loss_per_jet_summed_over_features": recon_loss,
        "codebook_utilization": util,
        "classes": {
            name: {**class_counts[name], "seen_in_training": seen_flags[name], **metrics.get(name, {})}
            for name in CLASS_FILES
        },
    }
    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ["reconstruction_loss_per_jet_summed_over_features", "codebook_utilization"]}, indent=2))
    print(f"Done. Output in {args.out_dir}")


if __name__ == "__main__":
    main()
