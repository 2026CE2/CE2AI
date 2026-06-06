"""
Main Entry Point
-----------------
Runs the full pipeline: data loading → model creation → training → evaluation.

Usage:
    python main.py --config configs/config.yaml
    python main.py --config configs/config.yaml --mode eval --checkpoint checkpoints/checkpoint_best.pt
"""

import argparse
from pathlib import Path

import torch

from data.dataset import get_dataloaders
from evaluation.metrics import depth_metrics, semantic_metrics, position_metrics
from evaluation.visualize import plot_training_history
from models.scene_predictor import ScenePredictor
from training.trainer import Trainer
from utils.helpers import count_parameters, get_device, load_config, set_seed


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


def train(cfg: dict, device: torch.device):
    print("\n=== Building data loaders ===")
    d = cfg["data"]
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=d["root_dir"],
        image_size=d["image_size"],
        batch_size=d["batch_size"],
        val_split=d["val_split"],
        test_split=d["test_split"],
        num_workers=d["num_workers"],
        seed=d["seed"],
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    print("\n=== Building model ===")
    model = build_model(cfg)
    count_parameters(model)

    t = cfg["training"]
    l = cfg["loss"]
    trainer_cfg = {
        "d_model": cfg["model"]["d_model"],
        "learning_rate": t["learning_rate"],
        "weight_decay": t["weight_decay"],
        "warmup_steps": t["warmup_steps"],
        "lambda_depth": l["lambda_depth"],
        "lambda_semantic": l["lambda_semantic"],
        "lambda_position": l["lambda_position"],
    }

    print("\n=== Training ===")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        device=str(device),
        checkpoint_dir=t["checkpoint_dir"],
    )

    history = trainer.fit(
        num_epochs=t["num_epochs"],
        early_stopping_patience=t["early_stopping_patience"],
    )

    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, save_path=results_dir / "training_history.png")

    return trainer, test_loader


@torch.no_grad()
def evaluate(model: ScenePredictor, test_loader, cfg: dict, device: torch.device):
    print("\n=== Evaluating on test set ===")
    model.eval()
    model.to(device)

    all_depth_metrics = []
    all_sem_metrics = []
    all_pos_metrics = []

    for batch in test_loader:
        images = batch["image"].to(device)
        preds = model(images)

        if "depth" in batch:
            dm = depth_metrics(preds["depth"].cpu(), batch["depth"])
            all_depth_metrics.append(dm)

        if "semantic" in batch:
            sm = semantic_metrics(
                preds["semantic"].cpu(),
                batch["semantic"],
                num_classes=cfg["model"]["num_classes"],
            )
            all_sem_metrics.append(sm)

        if "positions_3d" in batch:
            pm = position_metrics(preds["position"].cpu(), batch["positions_3d"])
            all_pos_metrics.append(pm)

    def _avg(metrics_list: list, key: str) -> float:
        return sum(m[key] for m in metrics_list) / len(metrics_list) if metrics_list else float("nan")

    print("\n--- Depth ---")
    print(f"  AbsRel : {_avg(all_depth_metrics, 'abs_rel'):.4f}")
    print(f"  RMSE   : {_avg(all_depth_metrics, 'rmse'):.4f}")
    print(f"  δ<1.25 : {_avg(all_depth_metrics, 'delta_1'):.4f}")

    print("--- Semantic ---")
    print(f"  mIoU   : {_avg(all_sem_metrics, 'miou'):.4f}")
    print(f"  Acc    : {_avg(all_sem_metrics, 'accuracy'):.4f}")

    print("--- 3-D Position ---")
    print(f"  MED    : {_avg(all_pos_metrics, 'med'):.4f}")


def main():
    parser = argparse.ArgumentParser(description="3D Scene Predictor")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="'train' or 'eval'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (required for --mode eval).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    device = get_device()

    if args.mode == "train":
        trainer, test_loader = train(cfg, device)
        evaluate(trainer.model, test_loader, cfg, device)

    elif args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("--checkpoint must be provided for eval mode.")
        model = build_model(cfg)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        d = cfg["data"]
        _, _, test_loader = get_dataloaders(
            root_dir=d["root_dir"],
            image_size=d["image_size"],
            batch_size=d["batch_size"],
        )
        evaluate(model, test_loader, cfg, device)


if __name__ == "__main__":
    main()
