"""
Inference Script
-----------------
Run the trained ScenePredictor on one or more images and save visualisations.

Usage:
    # Single image
    python predict.py --checkpoint checkpoints/checkpoint_best.pt --input path/to/image.jpg

    # Directory of images
    python predict.py --checkpoint checkpoints/checkpoint_best.pt --input path/to/images/ --output results/predictions/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.preprocessing import preprocess_image, denormalise
from models.scene_predictor import ScenePredictor
from utils.helpers import load_config, get_device


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model(cfg: dict) -> ScenePredictor:
    m = cfg["model"]
    d = cfg["data"]
    return ScenePredictor(
        image_size=d["image_size"],
        patch_size=d["patch_size"],
        d_model=m["d_model"],
        num_heads=m["num_heads"],
        num_enc_layers=m["num_enc_layers"],
        num_dec_layers=m["num_dec_layers"],
        d_ff=m["d_ff"],
        num_queries=m["num_queries"],
        num_classes=m["num_classes"],
        dropout=m["dropout"],
    )


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def save_prediction(
    image_path: Path,
    image_tensor: torch.Tensor,
    preds: dict[str, torch.Tensor],
    output_dir: Path,
):
    """Save a side-by-side figure: input image | predicted depth | predicted semantics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(image_path.name)

    # Input image (de-normalise from ImageNet stats)
    img = denormalise(image_tensor).permute(1, 2, 0).cpu().numpy()
    img = img.clip(0, 1)
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Predicted depth — reshape patch sequence back to a square grid
    depth = preds["depth"].squeeze(-1).cpu().numpy()   # (num_patches,)
    side = int(len(depth) ** 0.5)
    depth_map = depth[: side * side].reshape(side, side)
    im = axes[1].imshow(depth_map, cmap="plasma")
    axes[1].set_title("Predicted Depth")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Predicted semantic map
    sem = preds["semantic"].argmax(-1).cpu().numpy()   # (num_queries,)
    side_s = int(len(sem) ** 0.5)
    sem_map = sem[: side_s * side_s].reshape(side_s, side_s)
    axes[2].imshow(sem_map, cmap="tab20")
    axes[2].set_title("Predicted Semantics")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = output_dir / f"{image_path.stem}_prediction.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


@torch.no_grad()
def predict(args):
    cfg = load_config(args.config)
    device = get_device()

    # Load model
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    image_size = cfg["data"]["image_size"]
    input_path = Path(args.input)
    output_dir = Path(args.output)
    image_paths = collect_images(input_path)

    if not image_paths:
        raise FileNotFoundError(f"No supported images found at: {input_path}")

    print(f"Running inference on {len(image_paths)} image(s)...\n")

    for img_path in image_paths:
        image_tensor = preprocess_image(img_path, image_size)          # (3, H, W)
        batch = image_tensor.unsqueeze(0).to(device)                   # (1, 3, H, W)

        preds = model(batch)

        # Print raw numeric summaries to stdout
        depth_vals = preds["depth"][0].squeeze(-1)
        print(f"[{img_path.name}]")
        print(f"  Depth  — min: {depth_vals.min():.3f}  max: {depth_vals.max():.3f}  mean: {depth_vals.mean():.3f}")

        sem_class = preds["semantic"][0].argmax(-1)
        unique_classes = sem_class.unique().tolist()
        print(f"  Semantic — {len(unique_classes)} unique classes predicted")

        if "position" in preds:
            pos = preds["position"][0]   # (num_queries, 3)
            print(f"  Position — mean (x,y,z): ({pos[:,0].mean():.3f}, {pos[:,1].mean():.3f}, {pos[:,2].mean():.3f})")

        save_prediction(img_path, image_tensor, {k: preds[k][0] for k in preds}, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained ScenePredictor.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--input", type=str, required=True, help="Image file or directory.")
    parser.add_argument("--output", type=str, default="results/predictions", help="Output directory for visualisations.")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
