"""
Dataset
--------
PyTorch Dataset and DataLoader factory for 3D scene prediction.

Expected directory structure:

    data/
    └── raw/
        ├── images/          ← RGB images  (*.jpg, *.png)
        ├── depth/           ← Depth maps  (*.npy or *.png 16-bit)
        ├── semantic/        ← Semantic segmentation masks  (*.npy or *.png)
        └── poses/           ← Object 3-D poses  (*.json)

Each sample is identified by a common stem, e.g.:
    images/000001.jpg
    depth/000001.npy
    semantic/000001.npy
    poses/000001.json

Reference (book style):
  Raschka S. & Mirjalili V. – Python Machine Learning, 3rd ed.,
  Chapter 12 (PyTorch dataset / data-loader patterns).
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data._utils.collate import default_collate

from data.preprocessing import preprocess_image


def scene_collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate that only stacks keys present in *every* sample.
    This handles datasets where optional ground-truth files (depth, semantic,
    poses) may be missing for some samples.
    """
    # Find keys common to all samples in the batch
    common_keys = set(batch[0].keys())
    for sample in batch[1:]:
        common_keys &= set(sample.keys())

    # Separate string keys (e.g. 'stem') from tensor keys
    collated = {}
    for key in common_keys:
        values = [s[key] for s in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = default_collate(values)
        else:
            collated[key] = values  # keep as list for strings
    return collated


class SceneDataset(Dataset):
    """
    Dataset for single-image 3D scene prediction.

    Args:
        root_dir   (str | Path): path to the data root directory.
        image_size (int): images are resized to (image_size x image_size).
        split      (str): 'train', 'val', or 'test'.
        transform  (callable, optional): additional augmentation transform.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 224,
        split: str = "train",
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.split = split
        self.transform = transform

        self.image_dir = self.root_dir / "images"
        self.depth_dir = self.root_dir / "depth"
        self.semantic_dir = self.root_dir / "semantic"
        self.pose_dir = self.root_dir / "poses"

        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[str]:
        """Return sorted list of sample stems present in all modalities."""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        stems = sorted(
            p.stem
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        return stems

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem = self.samples[idx]

        # --- RGB image ---
        img_path = next(
            (self.image_dir / f"{stem}{ext}" for ext in [".jpg", ".jpeg", ".png"]
             if (self.image_dir / f"{stem}{ext}").exists()),
            None,
        )
        image = preprocess_image(img_path, self.image_size)

        if self.transform is not None:
            image = self.transform(image)

        sample = {"image": image, "stem": stem}

        # --- Depth map (optional) ---
        depth_path = self.depth_dir / f"{stem}.npy"
        if depth_path.exists():
            depth = torch.from_numpy(np.load(depth_path)).float()
            sample["depth"] = depth

        # --- Semantic mask (optional) ---
        sem_path = self.semantic_dir / f"{stem}.npy"
        if sem_path.exists():
            # NYU labels are 1-indexed (1–894); shift to 0-indexed (0–893).
            # Original label 0 (unlabeled) maps to -1 and is ignored by the loss.
            semantic = torch.from_numpy(np.load(sem_path)).long() - 1
            sample["semantic"] = semantic

        # --- Object poses (optional) ---
        pose_path = self.pose_dir / f"{stem}.json"
        if pose_path.exists():
            with open(pose_path) as f:
                poses = json.load(f)
            # Expect list of {"position": [x, y, z], "class_id": int}
            positions = torch.tensor(
                [obj["position"] for obj in poses], dtype=torch.float
            )
            class_ids = torch.tensor(
                [obj["class_id"] for obj in poses], dtype=torch.long
            )
            sample["positions_3d"] = positions
            sample["class_ids"] = class_ids

        return sample


def get_dataloaders(
    root_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 16,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / validation / test DataLoaders from a single dataset directory.

    Args:
        root_dir   : path to data root.
        image_size : images resized to this size.
        batch_size : samples per batch.
        val_split  : fraction for validation.
        test_split : fraction for test.
        num_workers: parallel data-loading workers.
        seed       : random seed for reproducible splits.

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset = SceneDataset(root_dir, image_size=image_size)
    total = len(dataset)

    n_test = int(total * test_split)
    n_val = int(total * val_split)
    n_train = total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=scene_collate_fn,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
