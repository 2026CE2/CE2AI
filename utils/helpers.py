"""
Utility Helpers
----------------
General-purpose helper functions used across the project.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


def set_seed(seed: int = 42):
    """
    Fix all random seeds for reproducibility.

    Sets seeds for Python's random, NumPy, and PyTorch (CPU & CUDA).
    Recommended practice from Raschka & Mirjalili, Python Machine Learning,
    Ch. 12 (reproducible experiments).

    Args:
        seed: integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Return the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.

    Args:
        model: any nn.Module.

    Returns:
        Number of trainable parameters.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total


def save_config(config: dict, path: str | Path):
    """
    Save a configuration dictionary to a YAML file.

    Args:
        config : dict of hyperparameters.
        path   : output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved → {path}")


def load_config(path: str | Path) -> dict:
    """
    Load a YAML configuration file.

    Args:
        path: path to YAML file.

    Returns:
        dict of configuration values.
    """
    with open(path) as f:
        return yaml.safe_load(f)
