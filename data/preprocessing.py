"""
Preprocessing and Augmentation Utilities
------------------------------------------
Follows the scikit-learn / torchvision pipeline philosophy described in
Raschka & Mirjalili – Python Machine Learning (Chapter 12–13).

Preprocessing pipeline:
  1. Load image from disk (PIL).
  2. Resize to a fixed square.
  3. Normalise pixel values using ImageNet statistics.
  4. Convert to a PyTorch tensor (C, H, W).

Augmentation pipeline (training only):
  - Random horizontal flip.
  - Random colour jitter (brightness, contrast, saturation).
  - Random rotation (±15°).
"""

from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# ImageNet normalisation statistics (common for pre-trained backbones)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_image(
    image_path: str | Path,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Load and preprocess a single image.

    Steps:
        1. Open with PIL and convert to RGB.
        2. Resize to (image_size × image_size).
        3. Convert to tensor in [0, 1].
        4. Normalise with ImageNet mean and std.

    Args:
        image_path : path to the image file.
        image_size : target spatial resolution.

    Returns:
        tensor of shape (3, image_size, image_size).
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image)


def augment_image(
    image: torch.Tensor,
    flip_prob: float = 0.5,
    jitter_prob: float = 0.3,
    rotation_degrees: float = 15.0,
) -> torch.Tensor:
    """
    Apply random augmentations to a pre-processed image tensor.

    Args:
        image             : (3, H, W) normalised tensor.
        flip_prob         : probability of horizontal flip.
        jitter_prob       : probability of colour jitter.
        rotation_degrees  : maximum rotation angle.

    Returns:
        Augmented tensor of the same shape.
    """
    # Random horizontal flip
    if torch.rand(1).item() < flip_prob:
        image = TF.hflip(image)

    # Random colour jitter (applied in image space, so we de-normalise first)
    if torch.rand(1).item() < jitter_prob:
        jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        # Convert back to PIL for jitter then re-normalise
        to_pil = T.ToPILImage()
        to_tensor = T.Compose([T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
        image = to_tensor(jitter(to_pil(image)))

    # Random rotation
    angle = (torch.rand(1).item() * 2 - 1) * rotation_degrees
    image = TF.rotate(image, angle)

    return image


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation for visualisation.

    Args:
        tensor: (3, H, W) or (B, 3, H, W) normalised tensor.

    Returns:
        Tensor with pixel values in [0, 1].
    """
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return (tensor * std + mean).clamp(0, 1)
