"""Callback for evaluating the tokenization of AK8 jets (L1T).

Every "particle" of the tokenizer is one AK8 jet. The callback compares the original and the
reconstructed kinematics (pT, eta, phi) of the *individual jets* and logs the codebook
utilization. Jets of an event are not combined into a super-jet and no mass is computed.
"""

import json
import os

import awkward as ak
import lightning as L
import numpy as np

import gabbro.plotting.utils as plot_utils
from gabbro.plotting.jet_reconstruction import codebook_utilization, make_per_jet_plots
from gabbro.utils.arrays import ak_select_and_preprocess, np_to_ak
from gabbro.utils.jet_types import jet_types_dict
from gabbro.utils.pylogger import get_pylogger

pylogger = get_pylogger("TokenizationEvalCallback")


def _reconstruct_phi_from_cos_sin(ak_arr):
    """Replace part_phi_cos/part_phi_sin fields with part_phi = atan2(sin, cos)."""
    if "part_phi_cos" not in ak_arr.fields or "part_phi_sin" not in ak_arr.fields:
        return ak_arr
    counts = ak.num(ak_arr.part_phi_cos)
    phi = np.arctan2(
        ak.to_numpy(ak.flatten(ak_arr.part_phi_sin)),
        ak.to_numpy(ak.flatten(ak_arr.part_phi_cos)),
    )
    phi_ak = ak.unflatten(phi, counts)
    new_fields = {f: ak_arr[f] for f in ak_arr.fields if f not in ("part_phi_cos", "part_phi_sin")}
    new_fields["part_phi"] = phi_ak
    return ak.Array(new_fields)


class TokenizationEvalCallback(L.Callback):
    def __init__(
        self,
        image_path: str = None,
        image_filetype: str = "png",
        no_trainer_info_in_filename: bool = False,
        save_result_arrays: bool = None,
    ):
        """Callback for evaluating the tokenization of AK8 jets.

        Parameters
        ----------
        image_path : str
            Path to save the images to. If None, the images are saved to the
            default_root_dir of the trainer.
        image_filetype : str
            Kept for backwards compatibility. The plots are always written as png.
        no_trainer_info_in_filename : bool
            If True, the filenames of the images will not contain the epoch and
            global step information. Default is False.
        save_result_arrays : bool
            Unused. Kept for backwards compatibility.
        """
        super().__init__()
        self.comet_logger = None
        self.image_path = image_path
        self.image_filetype = image_filetype
        self.no_trainer_info_in_filename = no_trainer_info_in_filename
        self.save_results_arrays = save_result_arrays

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.concat_validation_loop_predictions()
        self.plot(trainer, pl_module, stage="val")

    def on_test_epoch_end(self, trainer, pl_module):
        pl_module.concat_test_loop_predictions()
        self.plot(trainer, pl_module, stage="test")

    def plot(self, trainer, pl_module, stage="val"):
        plot_utils.set_mpl_style()
        if stage == "val" and not hasattr(pl_module, "val_x_original_concat"):
            pylogger.info("No validation predictions found. Skipping plotting.")
            return

        pylogger.info(
            f"Running TokenizationEvalCallback epoch: {trainer.current_epoch} step:"
            f" {trainer.global_step}"
        )
        # get the comet logger (if any) for logging the plots and metrics
        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.CometLogger):
                self.comet_logger = logger.experiment

        plot_dir = (
            self.image_path
            if self.image_path is not None
            else trainer.default_root_dir + "/plots/"
        )
        os.makedirs(plot_dir, exist_ok=True)

        if self.no_trainer_info_in_filename:
            prefix = "evaluation"
        elif stage == "val":
            prefix = f"val_epoch{trainer.current_epoch}_gstep{trainer.global_step}"
        elif stage == "test":
            prefix = "test"
        else:
            raise ValueError(f"stage {stage} not recognized")

        # get the results from the validation/test loop
        if stage == "val":
            x_recos = pl_module.val_x_reco_concat
            x_originals = pl_module.val_x_original_concat
            masks = pl_module.val_mask_concat
            labels = pl_module.val_labels_concat
            code_idx = pl_module.val_code_idx_concat
        else:
            if not hasattr(pl_module, "test_x_original_concat"):
                pylogger.info("No test predictions found. Skipping plotting.")
                return
            x_recos = pl_module.test_x_reco_concat
            x_originals = pl_module.test_x_original_concat
            masks = pl_module.test_mask_concat
            labels = pl_module.test_labels_concat
            code_idx = pl_module.test_code_idx_concat

        pp_dict = trainer.datamodule.hparams.dataset_kwargs_common.feature_dict

        # only use events with at least one jet
        has_jet = np.sum(masks, axis=1) >= 1
        x_recos, x_originals = x_recos[has_jet], x_originals[has_jet]
        masks, labels, code_idx = masks[has_jet], labels[has_jet], code_idx[has_jet]

        # back to physical units (pT in GeV, eta, phi in rad)
        x_reco_ak = ak_select_and_preprocess(
            np_to_ak(x_recos, mask=masks, names=pp_dict.keys()), pp_dict=pp_dict, inverse=True
        )
        x_original_ak = ak_select_and_preprocess(
            np_to_ak(x_originals, mask=masks, names=pp_dict.keys()), pp_dict=pp_dict, inverse=True
        )
        x_reco_ak = _reconstruct_phi_from_cos_sin(x_reco_ak)
        x_original_ak = _reconstruct_phi_from_cos_sin(x_original_ak)

        # per-jet plots: kinematics and residuals, separately for every jet class
        metrics = make_per_jet_plots(
            x_original_ak, x_reco_ak, labels, jet_types_dict, plot_dir, prefix
        )

        # codebook utilization (real jets only, and all positions incl. padding)
        n_codes = pl_module.model.vq_kwargs["num_codes"]
        util = codebook_utilization(code_idx, masks, n_codes)
        pylogger.info(f"Codebook utilization: {util}")
        with open(os.path.join(plot_dir, f"{prefix}_codebook_utilization.json"), "w") as f:
            json.dump(util, f, indent=2)

        if self.comet_logger is not None:
            for name in metrics:
                for kind in ["kinematics", "residuals"]:
                    fname = os.path.join(plot_dir, f"{prefix}_jet_{kind}_{name}.png")
                    self.comet_logger.log_image(
                        fname, name=os.path.basename(fname), step=trainer.global_step
                    )
            self.comet_logger.log_metric(
                f"{stage}_codebook_utilization",
                util["utilization_all_positions"],
                step=trainer.global_step,
            )
            self.comet_logger.log_metric(
                f"{stage}_codebook_utilization_real_jets",
                util["utilization_real_jets"],
                step=trainer.global_step,
            )
            # mean (abs) error per input feature in preprocessed space, real jets only
            n_feat = x_recos.shape[-1]
            diff = x_recos.reshape(-1, n_feat) - x_originals.reshape(-1, n_feat)
            valid = masks.reshape(-1) > 0
            for i, feature in enumerate(pp_dict.keys()):
                self.comet_logger.log_metric(
                    f"{stage}_mean_abserr_{feature}",
                    float(np.mean(np.abs(diff[valid, i]))),
                    step=trainer.global_step,
                )
                self.comet_logger.log_metric(
                    f"{stage}_mean_err_{feature}",
                    float(np.mean(diff[valid, i])),
                    step=trainer.global_step,
                )
