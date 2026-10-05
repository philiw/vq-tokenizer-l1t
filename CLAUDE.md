# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Tokenization for Real-Time Particle Data Compression** — a VQ-VAE (Vector-Quantized Variational Autoencoder) framework for learned compression of CMS Level-1 trigger particle data, targeting FPGA deployment. Continuous detector features are discretized into a finite codebook of learned tokens for efficient streaming and downstream anomaly detection within strict latency budgets.

Authors: Philipp Wagner et al. (ETH Zurich) + Hamburg group.

## Framework

The tokenizer is built on the Hamburg group's `enhancing-ntp4jets` repo, using **Hydra** for config management. Entry point: `Tokenizer/scripts/train.py`, main config: `Tokenizer/configs/train.yaml`, active experiment: `Tokenizer/configs/experiment/l1t_tokenization.yaml`.

## Data

| Location | Content |
|----------|---------|
| `data/filtered/*.parquet` | CMS L1 scouting data (AK8 jets): `minbias`, `QCD_HT50toInf`, `ggHbb`, `VBFHbb` |

**Features per jet:** `L1T_JetPuppiAK8_PT`, `L1T_JetPuppiAK8_Eta`, `L1T_JetPuppiAK8_Phi`

All four datasets are used in every split (train/val/test) to maximize phase-space coverage.

## Data Preprocessing

Defined in `Tokenizer/configs/feature_dict/feature_dict_l1t_jets.yaml`, applied by `ak_select_and_preprocess()` in `Tokenizer/gabbro/utils/arrays.py`:

| Feature | Transform |
|---|---|
| `part_pt` | clip ≥ 1 GeV → `log(pT)` → subtract 5.0 |
| `part_eta` | multiply by 0.5 |
| `part_phi_cos` | `cos(phi)` — no scaling, already ∈ [−1, 1] |
| `part_phi_sin` | `sin(phi)` — no scaling, already ∈ [−1, 1] |

All transforms are invertible. `phi` is recovered via `atan2(sin, cos)` in the callback. Events are padded to **7 jets** (max at L1T level); padding is tracked via a boolean mask applied during attention. Data loading: `Tokenizer/gabbro/data/iterable_dataset_jetclass.py`.

## Model Architecture: VQVAETransformer

Implemented in `Tokenizer/gabbro/models/vqvae.py`. Input shape: `(batch, seq_len=7, n_features=4)` — each event is a sequence of 7 jet tokens, each with 4 features (pt, eta, cos_phi, sin_phi).

```
Input: (batch, 7 jets, 4 features)   ← one event = sequence of 7 jet tokens
                    │
          [applied per jet token]
                    │
  → Linear projection:  4 → 128        (per-jet, shared weights)
  → Transformer Encoder: 4 blocks, 8 heads, pre-norm, GELU, MLP expansion=4
  │                                     (attention across all 7 jets)
  → Linear: 128 → 8                    (per-jet, compress to latent)
  → Vector Quantization                (per-jet independently)
  │    codebook: 8192 codes
  │    commitment weight beta=0.9
  │    dead code replacement every 500 steps
  → Linear: 8 → 128                   (per-jet, expand from latent)
  → Transformer Decoder: 4 blocks, causal masking
  │                                     (attention across all 7 jets)
  → Linear: 128 → 4                   (per-jet, reconstruct features)
                    │
Output: (batch, 7 jets, 3 features)
```

**Compression:** Each jet is mapped to one codebook index (13 bits). One event: 7 jets × 4 floats = 28 floats → 7 integers.

## Training

- **Loss:** `MSE_reconstruction + 10 × VQ_commitment_loss`
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-2; **Scheduler:** constant LR
- **Batch size:** 512 (train), 1000 (val/test)
- **Max steps:** 20,000; validation every 1,000 steps (20 checks total); early stopping patience=10 checks (i.e. stops if no improvement for 10,000 steps)
- **Checkpointing:** every 1,000 steps; best checkpoint kept by `val_loss`
- **Train/val/test split:** fraction-based, disjoint slices from the same 4 parquet files (minbias, QCD, ggHbb, VBFHbb). After filtering to events with ≥1 jet, each file is split 80/10/10: train rows 0–80%, val rows 80–90%, test rows 90–100%. All events in each slice are used. Approximate sizes: train ~3.83M, val ~478k, test ~478k (dominated by QCD). Class imbalance is intentional and not corrected. Batching controls RAM, not an event cap. Config: `Tokenizer/configs/data/iter_dataset_l1t_parquet.yaml` (`start_fraction`/`end_fraction` per split block).

## Evaluation & Metrics

Implemented in `Tokenizer/gabbro/callbacks/tokenization_callback.py` (plots in `Tokenizer/gabbro/plotting/jet_reconstruction.py`), triggered after every validation epoch and after the test loop. Plots are saved per jet class (minbias, QCD, ggHbb, VBFHbb).

**Plots produced (individual AK8 jets only):**
- `*_jet_kinematics_<class>.png`: pt/η/φ of the individual AK8 jets, original vs reconstructed
- `*_jet_residuals_<class>.png`: residuals (reco − original) of pt, η and φ per jet (Δφ wrapped to [−π, π))
- `*_per_jet_metrics.json`: mean, std, median and 68% half-width of the residuals per class

The jets of an event are **not** combined: there is no vector sum / super-jet and no mass. (The earlier "event hadronic activity" plots were the vector sum of all AK8 jets of an event with massless jets. They were removed because they only probe correlations between jets and carry little physics for this dataset.) Values outside the plotted range are collected in the first/last bin (under-/overflow), and their fraction is stored in `*_per_jet_metrics.json` (there is no text on the plots). The pT axis is 150–800 GeV, the φ range is exactly [−π, π].

**Codebook utilization:** fraction of the codes actually used, logged as a scalar metric and written to `*_codebook_utilization.json`. Padded positions are quantized as well, so both the number over all positions and the number over real jets only are reported.

**Jet substructure:** removed from the callback. `jet_substructure.py` clustered the AK8 jets of an event into a super-jet (kt, R=0.8) to compute τ₂₁, τ₃₂, D2, which requires ≥3 AK8 jets per event and is not meaningful for L1T. The module is no longer used.

**Re-creating the plots of an existing run** without Hydra or comet_ml: `scripts/evaluate_checkpoint_per_jet.py` (see its docstring). `--subset original_test` reproduces the test definition of the April codebook-sweep runs (first 20,000 events per file, which were part of the training data). `--subset held_out` uses events that those runs never trained on.

## Setup

```bash
pip install 'weaver-core>=0.4' pyarrow awkward uproot vector numpy pandas matplotlib fastjet
```
