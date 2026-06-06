"""
Trainer
--------
Encapsulates the training loop, validation loop, checkpointing, and
early stopping following the pattern recommended in:
  Raschka & Mirjalili – Python Machine Learning, 3rd ed., Ch. 12.
"""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from training.loss import SceneLoss
from training.scheduler import get_transformer_scheduler


class Trainer:
    """
    Manages the full training lifecycle.

    Args:
        model          : the ScenePredictor (or any nn.Module).
        train_loader   : DataLoader for training data.
        val_loader     : DataLoader for validation data.
        config         : dict / namespace with hyperparameters.
        device         : 'cuda', 'mps', or 'cpu'.
        checkpoint_dir : directory to save model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # --- Optimiser: AdamW (weight decay regularisation) ---
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-2),
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # --- Scheduler: transformer warm-up ---
        self.scheduler = get_transformer_scheduler(
            self.optimizer,
            d_model=config.get("d_model", 512),
            warmup_steps=config.get("warmup_steps", 4000),
        )

        # --- Loss function ---
        self.criterion = SceneLoss(
            lambda_depth=config.get("lambda_depth", 1.0),
            lambda_semantic=config.get("lambda_semantic", 1.0),
            lambda_position=config.get("lambda_position", 0.5),
        )

        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_batch(self, batch: dict) -> dict:
        """Move all tensors in a batch dict to the target device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _build_targets(self, batch: dict) -> dict:
        """Extract ground-truth tensors from a batch dict."""
        targets = {}
        for key in ("depth", "semantic", "positions_3d"):
            if key in batch:
                out_key = "position" if key == "positions_3d" else key
                targets[out_key] = batch[key]
        return targets

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = self._move_batch(batch)
            targets = self._build_targets(batch)

            self.optimizer.zero_grad()
            predictions = self.model(batch["image"])
            losses = self.criterion(predictions, targets)
            losses["total"].backward()

            # Gradient clipping prevents exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += losses["total"].item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            batch = self._move_batch(batch)
            targets = self._build_targets(batch)
            predictions = self.model(batch["image"])
            losses = self.criterion(predictions, targets)
            total_loss += losses["total"].item()

        return total_loss / len(self.val_loader)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "best"):
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
            },
            path,
        )
        print(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
        return ckpt["epoch"]

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
    ) -> dict:
        """
        Train for up to num_epochs, saving the best model by validation loss.

        Args:
            num_epochs              : maximum training epochs.
            early_stopping_patience : stop if val_loss doesn't improve.

        Returns:
            Training history dict with 'train_loss' and 'val_loss' lists.
        """
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()
            elapsed = time.time() - start

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(
                f"Epoch [{epoch:>4}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}  "
                f"({elapsed:.1f}s)"
            )

            # --- Save best model ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, tag="best")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

            # Save periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_loss, tag=f"epoch_{epoch}")

        return self.history
