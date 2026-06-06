import h5py, numpy as np
from PIL import Image
from pathlib import Path

f = h5py.File("nyu_depth_v2_labeled.mat", "r")
for i in range(f["images"].shape[0]):
    img   = f["images"][i].T          # (H, W, 3)
    depth = f["depths"][i].T          # (H, W)
    label = f["labels"][i].T          # (H, W)

    stem = f"{i:05d}"
    Image.fromarray(img).save(f"data/raw/images/{stem}.png")
    np.save(f"data/raw/depth/{stem}.npy", depth.astype("float32"))
    np.save(f"data/raw/semantic/{stem}.npy", label.astype("int64"))