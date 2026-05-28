import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, awkward as ak

from gabbro.data.loading import read_l1t_parquet_file
from gabbro.utils.arrays import ak_pad, ak_select_and_preprocess, ak_to_np_stack

PP_DICT = {
    "part_pt":  {"multiply_by": 1, "subtract_by": 5.0, "func": "np.log",
                 "inv_func": "np.exp", "clip_min_input_space": 1.0},
    "part_eta": {"multiply_by": 0.5},
    "part_phi": {"multiply_by": 0.3},
}

path = r"C:\Users\phili\Desktop\Studium\8. Semester\Semesterarbeit\data\filtered\minbias_kinematics.parquet"
x_ak, _, _ = read_l1t_parquet_file(path, particle_features=["part_pt","part_eta","part_phi"], n_load=200)
x_ak = ak_select_and_preprocess(x_ak, pp_dict=PP_DICT)
x_pad, mask = ak_pad(x_ak, maxlen=7, return_mask=True)
x_np   = ak_to_np_stack(x_pad, names=["part_pt","part_eta","part_phi"])
mask_np = ak.to_numpy(mask).astype("float32")
print("x_np shape:", x_np.shape, "  mask_np shape:", mask_np.shape)
print("x_np[0]:", x_np[0])
print("mask[0]:", mask_np[0])

from gabbro.models.vqvae import VQVAELightning
ckpt = os.path.join(os.path.dirname(__file__), "..", "logs", "l1t_tokenization", "runs", "codebook512", "checkpoints", "best.ckpt")
model = VQVAELightning.load_from_checkpoint(ckpt, map_location="cpu")
model.eval()

xb = torch.from_numpy(x_np[:10]).float()
mb = torch.from_numpy(mask_np[:10]).float()
with torch.no_grad():
    z, _ = model.model.encode(xb, mb)
    _, vq_out = model.model.quantize(z)
    ids = vq_out["q"]
print("token ids shape:", ids.shape)
print("token ids sample:", ids[0])
print("SUCCESS")
