"""Per-jet reconstruction plots for the L1T AK8 jet tokenizer.

In the L1T setup every "particle" of the tokenizer is one AK8 jet, and every event is a
sequence of up to 7 jets. The functions in this module compare the original and the
reconstructed (encoder -> codebook -> decoder) kinematics of the *individual jets*. The
jets of an event are not combined: no vector sum, no super-jet, no mass.

Values outside the plotted range are collected in the first and last bin (under-/overflow). The
fraction of such values is printed in the panel. The phi range is exactly [-pi, pi], so the edge
bins of phi only contain values that are genuinely beyond +-pi (possible for reconstructed jets).
"""

import json
import os
import time

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

# feature name (without the "part_" prefix) -> plotting settings
JET_FEATURES = {
    "pt": {
        "label": r"AK8 jet $p_\mathrm{T}$ [GeV]",
        "res_label": r"AK8 jet $p_\mathrm{T}^\mathrm{reco} - p_\mathrm{T}^\mathrm{original}$ [GeV]",
        "bins": np.linspace(150.0, 800.0, 66),
        "logy": True,
    },
    "eta": {
        "label": r"AK8 jet $\eta$",
        "res_label": r"AK8 jet $\eta^\mathrm{reco} - \eta^\mathrm{original}$",
        "bins": np.linspace(-5.0, 5.0, 101),
        "logy": False,
    },
    "phi": {
        "label": r"AK8 jet $\phi$",
        "res_label": r"AK8 jet $\Delta\phi$ (reco $-$ original, wrapped to $[-\pi,\pi)$)",
        "bins": np.linspace(-np.pi, np.pi, 101),
        "logy": False,
    },
}

COLOR_ORIGINAL = "#3d8fd9"
COLOR_RECO = "#c8200e"


def _savefig(fig, path, retries=5):
    """Save a figure, retrying on transient file errors (e.g. a file locked by a sync client)."""
    for attempt in range(retries):
        try:
            fig.savefig(path, dpi=200)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(1.0)


def wrap_phi(delta_phi):
    """Wrap an angle difference to [-pi, pi)."""
    return (delta_phi + np.pi) % (2.0 * np.pi) - np.pi


def _flat(ak_arr, field):
    """Flatten the jagged (event, jet) array of one field into a 1D float numpy array."""
    return ak.to_numpy(ak.flatten(ak_arr[f"part_{field}"], axis=None)).astype(np.float64)


def _hist_density(values, bins):
    """Density histogram with Poisson errors. Values outside the range go into the edge bins."""
    counts, _ = np.histogram(np.clip(values, bins[0], bins[-1]), bins=bins)
    n_total = max(len(values), 1)
    widths = np.diff(bins)
    density = counts / (n_total * widths)
    err = np.sqrt(counts) / (n_total * widths)
    return density, err


def _step_with_band(ax, bins, density, err, color, label):
    ax.stairs(density, bins, color=color, lw=1.3, label=label)
    ax.stairs(density + err, bins, baseline=np.maximum(density - err, 0), fill=True,
              color=color, alpha=0.25, lw=0)


def _outside_fraction(values, bins):
    return float(np.mean((values < bins[0]) | (values > bins[-1]))) if len(values) else 0.0


def plot_jet_kinematics(x_original_ak, x_reco_ak, title, out_path):
    """Original vs. reconstructed pT, eta, phi of the individual jets (one figure, 3 panels)."""
    overflow = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (feat, cfg) in zip(axes, JET_FEATURES.items()):
        orig = _flat(x_original_ak, feat)
        reco = _flat(x_reco_ak, feat)
        bins = cfg["bins"]
        d_o, e_o = _hist_density(orig, bins)
        d_r, e_r = _hist_density(reco, bins)
        _step_with_band(ax, bins, d_o, e_o, COLOR_ORIGINAL, "Original")
        _step_with_band(ax, bins, d_r, e_r, COLOR_RECO, "Reco")
        ax.set_xlabel(cfg["label"])
        ax.set_ylabel("Normalized")
        if cfg["logy"]:
            ax.set_yscale("log")
        ax.set_xlim(bins[0], bins[-1])
        overflow[feat] = {
            "original": _outside_fraction(orig, bins),
            "reco": _outside_fraction(reco, bins),
        }
        ax.legend(frameon=False, loc="upper right")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)
    return overflow


def _robust_sigma(x):
    q16, q84 = np.quantile(x, [0.16, 0.84])
    return float((q84 - q16) / 2.0)


def plot_jet_residuals(x_original_ak, x_reco_ak, title, out_path):
    """Per-jet residuals reco - original of pT, eta, phi. Returns summary statistics."""
    stats = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (feat, cfg) in zip(axes, JET_FEATURES.items()):
        diff = _flat(x_reco_ak, feat) - _flat(x_original_ak, feat)
        if feat == "phi":
            diff = wrap_phi(diff)
        diff = diff[np.isfinite(diff)]
        lo, hi = np.quantile(diff, [0.005, 0.995])
        bins = np.linspace(lo, hi, 101)
        density, err = _hist_density(diff, bins)
        _step_with_band(ax, bins, density, err, COLOR_ORIGINAL, "Reco $-$ original")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel(cfg["res_label"])
        ax.set_ylabel("Normalized")
        ax.set_xlim(bins[0], bins[-1])
        s = {
            "mean": float(np.mean(diff)),
            "std": float(np.std(diff)),
            "median": float(np.median(diff)),
            "robust_sigma_68": _robust_sigma(diff),
            "fraction_in_edge_bins": float(np.mean((diff < lo) | (diff > hi))),
        }
        stats[feat] = s
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)
    return stats


def codebook_utilization(code_idx, mask, n_codes):
    """Codebook utilization.

    ``code_idx`` and ``mask`` have shape (n_events, n_jets). Padded positions are also
    quantized by the model (their latent is the zero vector), so counting all positions can
    include the code assigned to padding. We therefore report both numbers.
    """
    code_idx = np.asarray(code_idx)
    if code_idx.ndim == 3:
        code_idx = code_idx[..., 0]
    valid = np.asarray(mask) > 0
    n_used_valid = int(len(np.unique(code_idx[valid])))
    n_used_all = int(len(np.unique(code_idx)))
    return {
        "n_codes": int(n_codes),
        "n_used_real_jets": n_used_valid,
        "utilization_real_jets": n_used_valid / n_codes,
        "n_used_all_positions": n_used_all,
        "utilization_all_positions": n_used_all / n_codes,
    }


def make_per_jet_plots(x_original_ak, x_reco_ak, labels, jet_types, out_dir, prefix, title_suffix=""):
    """Write per-jet kinematics and residual plots for every jet class.

    Parameters
    ----------
    x_original_ak, x_reco_ak : ak.Array
        Jagged (event, jet) arrays with fields ``part_pt`` (GeV), ``part_eta`` and ``part_phi``
        in physical units, i.e. after inverting the preprocessing.
    labels : array-like
        Class label per event.
    jet_types : dict
        ``{name: {"label": int, "tex_label": str, ...}}``.
    out_dir : str
        Output directory.
    prefix : str
        Filename prefix, e.g. ``"test"``.
    title_suffix : str
        Appended to the figure titles, e.g. ``"(K=8192)"``.

    Returns
    -------
    dict
        Per-class summary statistics (also written to ``{prefix}_per_jet_metrics.json``).
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = np.asarray(labels)
    metrics = {}
    for name, info in jet_types.items():
        sel = labels == info["label"]
        if not np.any(sel):
            continue
        orig = x_original_ak[sel]
        reco = x_reco_ak[sel]
        tex = info.get("tex_label", name)
        n_jets = int(ak.sum(ak.num(orig["part_pt"], axis=1)))
        suffix = f" {title_suffix}" if title_suffix else ""
        overflow = plot_jet_kinematics(
            orig, reco, f"AK8 jet kinematics of {tex} events{suffix}",
            os.path.join(out_dir, f"{prefix}_jet_kinematics_{name}.png"),
        )
        res_stats = plot_jet_residuals(
            orig, reco, f"AK8 jet residuals of {tex} events{suffix}",
            os.path.join(out_dir, f"{prefix}_jet_residuals_{name}.png"),
        )
        metrics[name] = {
            "n_events": int(np.sum(sel)),
            "n_jets": n_jets,
            "fraction_in_edge_bins_kinematics": overflow,
            "residuals": res_stats,
        }
    with open(os.path.join(out_dir, f"{prefix}_per_jet_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
