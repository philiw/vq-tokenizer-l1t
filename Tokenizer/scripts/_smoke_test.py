"""Smoke test: runs the full pipeline with tiny N to catch bugs fast."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gabbro.models.vqvae import VQVAELightning
from gabbro.models.classifiers import ClassifierTransformer
from gabbro.data.loading import read_l1t_parquet_file
from gabbro.utils.arrays import ak_pad, ak_select_and_preprocess, ak_to_np_stack

BASE_DIR = r"C:\Users\phili\Desktop\Studium\8. Semester\Semesterarbeit"
DATA_DIR = os.path.join(BASE_DIR, "data", "filtered")
LOGS_DIR = os.path.join(BASE_DIR, "Tokenizer", "logs", "l1t_tokenization", "runs")

PP_DICT = {
    "part_pt":  {"multiply_by": 1, "subtract_by": 5.0, "func": "np.log",
                 "inv_func": "np.exp", "clip_min_input_space": 1.0},
    "part_eta": {"multiply_by": 0.5},
    "part_phi": {"multiply_by": 0.3},
}
PAD_LENGTH = 7
N_CLASSES  = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARQUET_FILES = {
    "l1t_minbias": os.path.join(DATA_DIR, "minbias_kinematics.parquet"),
    "l1t_qcd":     os.path.join(DATA_DIR, "QCD_HT50toInf_kinematics.parquet"),
    "l1t_ggHbb":   os.path.join(DATA_DIR, "ggHbb_kinematics.parquet"),
    "l1t_VBFHbb":  os.path.join(DATA_DIR, "VBFHbb_kinematics.parquet"),
}
CLASS_LABELS = {name: i for i, name in enumerate(PARQUET_FILES)}

def load_class_data(filepath, n_load, label_idx):
    x_ak, _, _ = read_l1t_parquet_file(
        filepath, particle_features=["part_pt", "part_eta", "part_phi"], n_load=n_load
    )
    x_ak = ak_select_and_preprocess(x_ak, pp_dict=PP_DICT)
    x_padded, mask = ak_pad(x_ak, maxlen=PAD_LENGTH, return_mask=True)
    x_np    = ak_to_np_stack(x_padded, names=list(PP_DICT.keys()))
    mask_np = ak.to_numpy(mask).astype("float32")
    y_np    = np.full(len(x_np), label_idx, dtype="int64")
    return x_np, mask_np, y_np

N_SMALL = 100
xs, masks, ys = [], [], []
for name, path in PARQUET_FILES.items():
    label = CLASS_LABELS[name]
    x, m, y = load_class_data(path, N_SMALL, label)
    xs.append(x[:N_SMALL]); masks.append(m[:N_SMALL]); ys.append(y[:N_SMALL])

xs    = np.concatenate(xs)
masks = np.concatenate(masks)
ys    = np.concatenate(ys)
print(f"xs: {xs.shape}  masks: {masks.shape}  ys: {ys.shape}")

# test tokenisation
ckpt = os.path.join(LOGS_DIR, "codebook512", "checkpoints", "best.ckpt")
model = VQVAELightning.load_from_checkpoint(ckpt, map_location="cpu")
model.eval()
with torch.no_grad():
    xb = torch.from_numpy(xs[:8]).float()
    mb = torch.from_numpy(masks[:8]).float()
    z, _ = model.model.encode(xb, mb)
    _, vq_out = model.model.quantize(z)
    ids = vq_out["q"].numpy()
    if ids.ndim == 3:
        ids = ids.squeeze(-1)
print(f"token ids shape: {ids.shape}")
del model

# test MLP bag-of-tokens forward pass
num_codes = 512

class BagOfTokensMLP(nn.Module):
    def __init__(self, num_codes, n_classes=N_CLASSES):
        super().__init__()
        self.num_codes = num_codes
        self.net = nn.Sequential(
            nn.Linear(num_codes, 64), nn.GELU(), nn.Linear(64, n_classes)
        )
    def forward(self, token_ids, mask):
        B = token_ids.shape[0]
        hist = torch.zeros(B, self.num_codes)
        valid = mask > 0
        for i in range(token_ids.shape[1]):
            idx = token_ids[:, i].clamp(0, self.num_codes - 1)
            hist[torch.arange(B), idx] += valid[:, i].float()
        n_real = valid.sum(dim=1, keepdim=True).float().clamp(min=1)
        return self.net(hist / n_real)

mlp = BagOfTokensMLP(num_codes)
t_ids = torch.from_numpy(ids).long()
t_mask = torch.from_numpy(masks[:8]).float()
out = mlp(t_ids, t_mask)
print(f"MLP output shape: {out.shape}")

# test Transformer forward pass
class TokenTransformerClassifier(nn.Module):
    def __init__(self, num_codes, embed_dim=32, hidden_dim=64, n_classes=N_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, embed_dim)
        self.cls = ClassifierTransformer(
            input_dim=embed_dim, hidden_dim=hidden_dim,
            num_heads=2, num_enc_blocks=0, num_class_blocks=2,
            n_out_nodes=n_classes, cross_attention_model_class="NormformerCrossBlockv2",
        )
    def forward(self, token_ids, mask):
        x = self.embedding(token_ids.clamp(0))
        return self.cls(x, mask)

trans = TokenTransformerClassifier(num_codes)
out2 = trans(t_ids, t_mask)
print(f"Transformer output shape: {out2.shape}")

print("ALL SMOKE TESTS PASSED")
