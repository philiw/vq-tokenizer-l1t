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

Implemented in `Tokenizer/gabbro/callbacks/tokenization_callback.py`, triggered after every validation epoch. Plots are saved per jet class (minbias, QCD, ggHbb, VBFHbb).

**Plots actually produced:**
- **Individual jet kinematics** (most relevant): pt/η/φ distributions of original vs reconstructed AK8 jets, plus residuals (reco − original) per jet class
- **Event-level hadronic activity**: vector sum of all AK8 jets per event (proxy for HT/MHT), pt/η/φ/mass original vs reco and residuals per jet class

**Codebook utilization:** fraction of the 8,192 codes actually used, logged as a scalar metric.

**Jet substructure (`jet_substructure.py`) — not meaningful for this dataset:** the callback calls `JetSubstructure`, which treats the AK8 jets within one event as "particles" and clusters them into a super-jet (kt, R=0.8) to compute τ₂₁, τ₃₂, D2. This requires ≥3 AK8 jets per event, which is rare in the L1T data (minbias has only 1,344 jets total). Results are logged as scalar mean errors to CometML but not saved as plots. This code is inherited from the JetClass setup and is effectively dead weight for L1T — can be removed.

## Setup

```bash
pip install 'weaver-core>=0.4' pyarrow awkward uproot vector numpy pandas matplotlib fastjet
```
