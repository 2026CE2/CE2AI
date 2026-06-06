"""
Visualisation Utilities
------------------------
Helper functions for plotting results and inspecting model behaviour.

References:
  - Raschka & Mirjalili, Python Machine Learning, Ch. 12 (matplotlib patterns).
  - Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.preprocessing import denormalise


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict[str, list],
    save_path: str | Path = None,
):
    """
    Plot training and validation loss curves.

    Args:
        history   : dict with 'train_loss' and 'val_loss' lists.
        save_path : if provided, saves the figure to this path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training history plot → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Prediction visualisation
# ---------------------------------------------------------------------------

def visualize_predictions(
    images: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor] = None,
    num_samples: int = 4,
    save_path: str | Path = None,
):
    """
    Side-by-side visualisation of input image, predicted depth, and semantic map.

    Args:
        images      : (batch, 3, H, W) normalised tensor.
        predictions : dict with 'depth', 'semantic' tensors.
        targets     : optional ground-truth dict.
        num_samples : how many samples to display.
        save_path   : optional file path to save the figure.
    """
    n = min(num_samples, images.size(0))
    cols = 3 if targets is None else 5
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3, n * 3))

    if n == 1:
        axes = [axes]

    for i in range(n):
        # Input image
        img = denormalise(images[i]).permute(1, 2, 0).cpu().numpy()
        axes[i][0].imshow(img)
        axes[i][0].set_title("Input Image")
        axes[i][0].axis("off")

        # Predicted depth (reshape from patches to a rough square)
        if "depth" in predictions:
            depth = predictions["depth"][i].squeeze(-1).cpu().detach().numpy()
            side = int(len(depth) ** 0.5)
            depth_map = depth[: side * side].reshape(side, side)
            im = axes[i][1].imshow(depth_map, cmap="plasma")
            axes[i][1].set_title("Pred Depth")
            axes[i][1].axis("off")
            plt.colorbar(im, ax=axes[i][1], fraction=0.046)

        # Predicted semantic map
        if "semantic" in predictions:
            sem = predictions["semantic"][i].argmax(-1).cpu().detach().numpy()
            side = int(len(sem) ** 0.5)
            sem_map = sem[: side * side].reshape(side, side)
            axes[i][2].imshow(sem_map, cmap="tab20")
            axes[i][2].set_title("Pred Semantic")
            axes[i][2].axis("off")

        # Ground-truth comparisons
        if targets is not None:
            if "depth" in targets:
                gt_depth = targets["depth"][i].squeeze(-1).cpu().numpy()
                side = int(len(gt_depth) ** 0.5)
                gt_map = gt_depth[: side * side].reshape(side, side)
                im = axes[i][3].imshow(gt_map, cmap="plasma")
                axes[i][3].set_title("GT Depth")
                axes[i][3].axis("off")
                plt.colorbar(im, ax=axes[i][3], fraction=0.046)

            if "semantic" in targets:
                gt_sem = targets["semantic"][i].cpu().numpy()
                side = int(len(gt_sem) ** 0.5)
                gt_sem_map = gt_sem[: side * side].reshape(side, side)
                axes[i][4].imshow(gt_sem_map, cmap="tab20")
                axes[i][4].set_title("GT Semantic")
                axes[i][4].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved predictions visualisation → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Attention map visualisation
# ---------------------------------------------------------------------------

def visualize_attention_maps(
    attention_weights: torch.Tensor,
    image: torch.Tensor,
    patch_size: int = 16,
    layer: int = 0,
    head: int = 0,
    save_path: str | Path = None,
):
    """
    Visualise which image patches a query token attends to.

    Args:
        attention_weights : (batch, heads, seq_q, seq_k) from encoder/decoder.
        image             : (3, H, W) normalised tensor.
        patch_size        : patch size used in PatchEmbedding.
        layer             : which transformer layer to visualise.
        head              : which attention head.
        save_path         : optional save path.
    """
    attn = attention_weights[0, head].cpu().detach().numpy()  # (seq_q, seq_k)
    num_patches = attn.shape[-1]
    side = int(num_patches ** 0.5)

    img_np = denormalise(image).permute(1, 2, 0).cpu().numpy()
    H, W = img_np.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Mean attention over all queries
    mean_attn = attn.mean(axis=0)[: side * side].reshape(side, side)
    attn_resized = np.kron(mean_attn, np.ones((patch_size, patch_size)))

    axes[1].imshow(img_np)
    axes[1].imshow(attn_resized[:H, :W], alpha=0.6, cmap="hot")
    axes[1].set_title(f"Attention Overlay (head={head})")
    axes[1].axis("off")

    axes[2].imshow(mean_attn, cmap="hot", interpolation="nearest")
    axes[2].set_title("Raw Attention Map")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved attention map → {save_path}")
    plt.show()
