"""
Token quality evaluation via multi-class classifier accuracy.

Reproduces a plot similar to Figure 4 of "From particle clouds to tokens: building
foundation models for particle physics", using our CMS L1T scouting data with four
classes (minbias, QCD, ggHbb, VBFHbb) and codebook sizes 512-8192.

Two classifier architectures are trained and evaluated:
  - MLP on bag-of-tokens (histogram of token IDs per event)
  - Transformer on token embeddings (sequence of token IDs)

Classifiers trained on original particle features provide the upper-bound accuracy.
"""

import os
import sys
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from gabbro.models.vqvae import VQVAELightning
from gabbro.models.classifiers import ClassifierTransformer
from gabbro.data.loading import read_l1t_parquet_file
from gabbro.utils.arrays import ak_pad, ak_select_and_preprocess, ak_to_np_stack

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = r"C:\Users\phili\Desktop\Studium\8. Semester\Semesterarbeit"
DATA_DIR = os.path.join(BASE_DIR, "data", "filtered")
LOGS_DIR = os.path.join(BASE_DIR, "Tokenizer", "logs", "l1t_tokenization", "runs")
OUT_DIR  = os.path.join(BASE_DIR, "experiment_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_FILE = os.path.join(OUT_DIR, f"token_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
# Force stdout to be line-buffered so both handlers flush immediately
sys.stdout.reconfigure(line_buffering=True)
log = logging.getLogger()

def log_print(msg):
    log.info(msg)

CODEBOOK_SIZES = [512, 1024, 2048, 4096, 8192]

PARQUET_FILES = {
    "l1t_minbias": os.path.join(DATA_DIR, "minbias_kinematics.parquet"),
    "l1t_qcd":     os.path.join(DATA_DIR, "QCD_HT50toInf_kinematics.parquet"),
    "l1t_ggHbb":   os.path.join(DATA_DIR, "ggHbb_kinematics.parquet"),
    "l1t_VBFHbb":  os.path.join(DATA_DIR, "VBFHbb_kinematics.parquet"),
}
CLASS_LABELS = {name: i for i, name in enumerate(PARQUET_FILES)}  # 0-3

# Preprocessing from feature_dict_l1t_jets.yaml
PP_DICT = {
    "part_pt":  {"multiply_by": 1, "subtract_by": 5.0, "func": "np.log",
                 "inv_func": "np.exp", "clip_min_input_space": 1.0},
    "part_eta": {"multiply_by": 0.5},
    "part_phi": {"multiply_by": 0.3},
}

PAD_LENGTH      = 7      # max jets per event (from config)
N_CLASSES       = 4
N_MAX_PER_CLASS = None   # None → probe all files and cap at the smallest class
FRAC_TRAIN      = 0.70
FRAC_VAL        = 0.15
# remaining fraction goes to test
BATCH_SIZE  = 128   # smaller batches work better for small balanced dataset
N_EPOCHS    = 100   # more epochs to compensate for fewer samples per epoch
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_print(f"Using device: {DEVICE}")


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_class_data(filepath, n_load, label_idx):
    """Load, preprocess, and pad particle data for one class."""
    x_ak, _, _ = read_l1t_parquet_file(
        filepath, particle_features=["part_pt", "part_eta", "part_phi"], n_load=n_load
    )
    x_ak = ak_select_and_preprocess(x_ak, pp_dict=PP_DICT)
    x_padded, mask = ak_pad(x_ak, maxlen=PAD_LENGTH, return_mask=True)
    x_np   = ak_to_np_stack(x_padded, names=list(PP_DICT.keys()))  # (N, 7, 3)
    mask_np = ak.to_numpy(mask).astype("float32")                   # (N, 7)
    y_np   = np.full(len(x_np), label_idx, dtype="int64")
    return x_np, mask_np, y_np


def build_dataset():
    """Load each class, cap at the smallest class size, split 70/15/15, concatenate."""
    rng = np.random.default_rng(42)

    # --- probe step: find the minimum available count across all classes ---
    if N_MAX_PER_CLASS is None:
        log_print("  Probing class sizes to determine balance cap...")
        sizes = {}
        for name, path in PARQUET_FILES.items():
            x, _, _ = load_class_data(path, n_load=None, label_idx=0)
            sizes[name] = len(x)
            log_print(f"    {name}: {len(x)} events")
        cap = min(sizes.values())
        log_print(f"  → Capping all classes at {cap} events (balanced)")
    else:
        cap = N_MAX_PER_CLASS

    splits = {"train": [], "val": [], "test": []}
    class_counts = []

    for name, path in PARQUET_FILES.items():
        label = CLASS_LABELS[name]
        log_print(f"  Loading {name}...")
        x, m, y = load_class_data(path, n_load=cap, label_idx=label)
        n = len(x)
        class_counts.append(n)
        perm = rng.permutation(n)
        x, m, y = x[perm], m[perm], y[perm]
        n_tr = int(n * FRAC_TRAIN)
        n_va = int(n * FRAC_VAL)
        splits["train"].extend([x[:n_tr],           m[:n_tr],           y[:n_tr]])
        splits["val"].extend(  [x[n_tr:n_tr+n_va],  m[n_tr:n_tr+n_va],  y[n_tr:n_tr+n_va]])
        splits["test"].extend( [x[n_tr+n_va:],      m[n_tr+n_va:],      y[n_tr+n_va:]])
        log_print(f"    → {n_tr} train / {n_va} val / {n-n_tr-n_va} test  (total loaded: {n})")

    def cat_shuffle(arrs):
        xs = np.concatenate(arrs[0::3])
        ms = np.concatenate(arrs[1::3])
        ys = np.concatenate(arrs[2::3])
        p  = rng.permutation(len(xs))
        return xs[p], ms[p], ys[p]

    # class weights inversely proportional to class frequency (for imbalanced data)
    total = sum(class_counts)
    weights = torch.tensor([total / (N_CLASSES * c) for c in class_counts], dtype=torch.float)
    log_print(f"  Class counts: {class_counts}  →  weights: {weights.tolist()}")

    return (cat_shuffle(splits["train"]),
            cat_shuffle(splits["val"]),
            cat_shuffle(splits["test"]),
            weights)


log_print("Loading data...")
(x_train, mask_train, y_train), \
(x_val,   mask_val,   y_val),   \
(x_test,  mask_test,  y_test),  \
class_weights = build_dataset()

log_print(f"Train: {len(x_train)}  Val: {len(x_val)}  Test: {len(x_test)}")

x_train_t  = torch.from_numpy(x_train).float()
mask_train_t = torch.from_numpy(mask_train).float()
y_train_t  = torch.from_numpy(y_train).long()

x_val_t    = torch.from_numpy(x_val).float()
mask_val_t = torch.from_numpy(mask_val).float()
y_val_t    = torch.from_numpy(y_val).long()

x_test_t   = torch.from_numpy(x_test).float()
mask_test_t = torch.from_numpy(mask_test).float()
y_test_t   = torch.from_numpy(y_test).long()


# ─── Token Extraction ─────────────────────────────────────────────────────────

def get_token_ids(ckpt_path, x_np, mask_np, batch_size=512):
    """Run the VQ-VAE encoder + quantizer and return integer token IDs."""
    model = VQVAELightning.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    model.to(DEVICE)
    all_ids = []
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb = torch.from_numpy(x_np[i:i+batch_size]).float().to(DEVICE)
            mb = torch.from_numpy(mask_np[i:i+batch_size]).float().to(DEVICE)
            z_embed, _ = model.model.encode(xb, mb)
            _, vq_out = model.model.quantize(z_embed)
            ids = vq_out["q"].detach().cpu().numpy()   # (B, 7) or (B, 7, 1)
            if ids.ndim == 3:
                ids = ids.squeeze(-1)          # → (B, 7)
            all_ids.append(ids)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_ids, axis=0).astype("int64")


# ─── Classifier Helpers ───────────────────────────────────────────────────────

def make_dataloader(tensors, batch_size=BATCH_SIZE, shuffle=True):
    ds = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_and_eval(model, train_loader, val_loader, test_loader,
                   n_epochs=N_EPOCHS, lr=LR, label=""):
    """Train `model` and return best-val-epoch test accuracy."""
    model.to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            batch = [b.to(DEVICE) for b in batch]
            opt.zero_grad()
            logits = model(*batch[:-1])
            loss   = crit(logits, batch[-1])
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(DEVICE) for b in batch]
                logits = model(*batch[:-1])
                preds  = logits.argmax(-1)
                val_correct += (preds == batch[-1]).sum().item()
                val_total   += batch[-1].size(0)
        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            import copy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = [b.to(DEVICE) for b in batch]
            logits = model(*batch[:-1])
            preds  = logits.argmax(-1)
            test_correct += (preds == batch[-1]).sum().item()
            test_total   += batch[-1].size(0)
    return test_correct / test_total


# ─── MLP bag-of-tokens ────────────────────────────────────────────────────────

class BagOfTokensMLP(nn.Module):
    def __init__(self, num_codes, n_classes=N_CLASSES):
        super().__init__()
        self.num_codes = num_codes
        self.net = nn.Sequential(
            nn.Linear(num_codes, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, token_ids, mask):
        # token_ids: (B, 7)  mask: (B, 7)  → histogram over codebook
        B = token_ids.shape[0]
        hist = torch.zeros(B, self.num_codes, device=token_ids.device)
        valid = (mask > 0)  # (B, 7)
        for i in range(token_ids.shape[1]):
            idx = token_ids[:, i].clamp(0, self.num_codes - 1)
            hist[torch.arange(B), idx] += valid[:, i].float()
        # normalize by number of real jets
        n_real = valid.sum(dim=1, keepdim=True).float().clamp(min=1)
        hist = hist / n_real
        return self.net(hist)


# ─── Transformer on token embeddings ──────────────────────────────────────────

class TokenTransformerClassifier(nn.Module):
    def __init__(self, num_codes, embed_dim=64, hidden_dim=128,
                 num_heads=4, num_class_blocks=3, n_classes=N_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, embed_dim)
        self.cls = ClassifierTransformer(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_enc_blocks=0,
            num_class_blocks=num_class_blocks,
            n_out_nodes=n_classes,
            dropout_rate=0.1,
            cross_attention_model_class="NormformerCrossBlockv2",
        )

    def forward(self, token_ids, mask):
        x = self.embedding(token_ids.clamp(0))   # (B, 7, embed_dim)
        return self.cls(x, mask)


# ─── Original-features MLP ────────────────────────────────────────────────────

class OriginalFeatureMLP(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(PAD_LENGTH * 3, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128),            nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, mask):
        # zero out padding before flattening
        return self.net(x * mask.unsqueeze(-1))


# ─── Original-features Transformer ───────────────────────────────────────────

class OriginalFeatureTransformer(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.cls = ClassifierTransformer(
            input_dim=3,
            hidden_dim=128,
            num_heads=4,
            num_enc_blocks=0,
            num_class_blocks=3,
            n_out_nodes=n_classes,
            dropout_rate=0.1,
            cross_attention_model_class="NormformerCrossBlockv2",
        )

    def forward(self, x, mask):
        return self.cls(x, mask)


# ─── Run Experiments ──────────────────────────────────────────────────────────

results = {}   # {codebook_size: {"mlp": acc, "transformer": acc}}

# --- Original features (upper bound) -----------------------------------------

log_print("\n=== Training classifiers on original features (upper bound) ===")

def make_orig_loaders():
    train_dl = make_dataloader([x_train_t, mask_train_t, y_train_t])
    val_dl   = make_dataloader([x_val_t,   mask_val_t,   y_val_t],   shuffle=False)
    test_dl  = make_dataloader([x_test_t,  mask_test_t,  y_test_t],  shuffle=False)
    return train_dl, val_dl, test_dl

train_dl, val_dl, test_dl = make_orig_loaders()

orig_mlp = OriginalFeatureMLP()
acc_orig_mlp = train_and_eval(orig_mlp, train_dl, val_dl, test_dl)
log_print(f"  Original MLP accuracy:         {acc_orig_mlp:.4f}")

orig_trans = OriginalFeatureTransformer()
acc_orig_trans = train_and_eval(orig_trans, train_dl, val_dl, test_dl)
log_print(f"  Original Transformer accuracy: {acc_orig_trans:.4f}")


# --- Per-codebook -------------------------------------------------------------

for cb_size in CODEBOOK_SIZES:
    ckpt = os.path.join(LOGS_DIR, f"codebook{cb_size}", "checkpoints", "best.ckpt")
    log_print(f"\n=== Codebook {cb_size}: encoding data... ===")

    # Encode all splits in one model load to avoid triple checkpoint loading
    x_all   = np.concatenate([x_train,    x_val,    x_test],    axis=0)
    m_all   = np.concatenate([mask_train, mask_val, mask_test], axis=0)
    ids_all = get_token_ids(ckpt, x_all, m_all)
    n_tr, n_va = len(x_train), len(x_val)
    ids_train = ids_all[:n_tr]
    ids_val   = ids_all[n_tr:n_tr+n_va]
    ids_test  = ids_all[n_tr+n_va:]
    log_print(f"  Encoded {len(ids_all)} events.")

    ids_train_t = torch.from_numpy(ids_train).long()
    ids_val_t   = torch.from_numpy(ids_val).long()
    ids_test_t  = torch.from_numpy(ids_test).long()

    train_dl_tok = make_dataloader([ids_train_t, mask_train_t, y_train_t])
    val_dl_tok   = make_dataloader([ids_val_t,   mask_val_t,   y_val_t],   shuffle=False)
    test_dl_tok  = make_dataloader([ids_test_t,  mask_test_t,  y_test_t],  shuffle=False)

    # MLP bag-of-tokens
    mlp_model = BagOfTokensMLP(num_codes=cb_size)
    acc_mlp   = train_and_eval(mlp_model, train_dl_tok, val_dl_tok, test_dl_tok)
    log_print(f"  MLP bag-of-tokens accuracy:  {acc_mlp:.4f}")

    # Transformer on token embeddings
    trans_model = TokenTransformerClassifier(num_codes=cb_size)
    acc_trans   = train_and_eval(trans_model, train_dl_tok, val_dl_tok, test_dl_tok)
    log_print(f"  Transformer accuracy:         {acc_trans:.4f}")

    results[cb_size] = {"mlp": acc_mlp, "transformer": acc_trans}


# ─── Plot ─────────────────────────────────────────────────────────────────────

cb_sizes  = CODEBOOK_SIZES
acc_mlp   = [results[cb]["mlp"]         for cb in cb_sizes]
acc_trans = [results[cb]["transformer"] for cb in cb_sizes]

fig, ax = plt.subplots(figsize=(6, 4.5))

color_mlp   = "#7b52ab"   # purple
color_trans = "#2ca02c"   # green

ax.plot(cb_sizes, acc_mlp,   "o-", color=color_mlp,   lw=2, markersize=6,
        label="MLP (bag of tokens)")
ax.plot(cb_sizes, acc_trans, "s-", color=color_trans, lw=2, markersize=6,
        label="Transformer (token emb.)")

ax.axhline(acc_orig_mlp,   color=color_mlp,   linestyle="--", lw=1.5, alpha=0.7,
           label=f"MLP (original, upper limit = {acc_orig_mlp:.3f})")
ax.axhline(acc_orig_trans, color=color_trans, linestyle="--", lw=1.5, alpha=0.7,
           label=f"Transformer (original, upper limit = {acc_orig_trans:.3f})")

ax.set_xscale("log", base=2)
ax.set_xticks(cb_sizes)
ax.set_xticklabels([str(c) for c in cb_sizes])
ax.set_xlabel("Codebook size", fontsize=12)
ax.set_ylabel("4-class accuracy", fontsize=12)
ax.set_title("Token quality: classifier accuracy vs codebook size\n"
             "(L1T: minbias / QCD / ggHbb / VBFHbb)", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.3)

out_path = os.path.join(OUT_DIR, "token_quality_classifier_accuracy_balanced.pdf")
fig.tight_layout()
fig.savefig(out_path, dpi=150)
log_print(f"\nPlot saved to {out_path}")

# also print a summary
log_print("\n=== Summary ===")
log_print(f"{'Codebook':>10}  {'MLP':>8}  {'Transformer':>12}")
for cb in cb_sizes:
    log_print(f"{cb:>10}  {results[cb]['mlp']:>8.4f}  {results[cb]['transformer']:>12.4f}")
log_print(f"{'Original':>10}  {acc_orig_mlp:>8.4f}  {acc_orig_trans:>12.4f}")
