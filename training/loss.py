"""
Loss Functions
---------------
Combined loss for 3D scene prediction:

    L_total = λ_depth    * L_depth
            + λ_semantic * L_semantic
            + λ_position * L_position

  - L_depth    : smooth L1 loss between predicted and ground-truth depth.
  - L_semantic : cross-entropy loss over semantic class logits.
  - L_position : mean squared error between predicted and true 3D positions.

Reference: Raschka & Mirjalili, Python Machine Learning, Ch. 14–15
(custom loss functions in PyTorch).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLoss(nn.Module):
    """
    Scale-invariant depth loss (eigen et al., 2014) with smooth-L1 fallback.

    Args:
        use_scale_invariant (bool): if True use SI-log loss, else smooth L1.
    """

    def __init__(self, use_scale_invariant: bool = True):
        super().__init__()
        self.use_scale_invariant = use_scale_invariant

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred   : (batch, num_patches, 1) predicted depth values.
            target : (batch, num_patches, 1) OR (batch, H, W) ground-truth depth.
                     If target is a full-resolution map it is downsampled to the
                     patch grid automatically.
        """
        # --- Align target shape to pred (batch, num_patches, 1) ---
        if target.dim() == 3 and target.shape[1:] != pred.shape[1:]:
            # target is (B, H, W) – downsample to patch grid
            num_patches = pred.size(1)
            side = int(num_patches ** 0.5)
            target = F.interpolate(
                target.unsqueeze(1).float(),   # (B, 1, H, W)
                size=(side, side),
                mode="bilinear",
                align_corners=False,
            )                                  # (B, 1, side, side)
            target = target.view(target.size(0), -1, 1)  # (B, num_patches, 1)

        if self.use_scale_invariant:
            mask = target > 0
            # Clamp to guarantee log receives a strictly positive value
            pred_c   = pred[mask].clamp(min=1e-6)
            target_c = target[mask].clamp(min=1e-6)
            d = torch.log(pred_c) - torch.log(target_c)
            return d.pow(2).mean() - 0.5 * d.mean().pow(2)
        return F.smooth_l1_loss(pred, target)


class SemanticLoss(nn.Module):
    """Cross-entropy loss for semantic segmentation."""

    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred   : (batch, num_queries, num_classes) logits.
            target : (batch, num_queries) OR (batch, H, W) class indices.
                     If target is a full-resolution mask it is downsampled via
                     nearest-neighbour to match the number of query tokens.
        """
        B, Q, C = pred.shape

        # Downsample full-resolution mask to query count
        if target.dim() == 3 and target.shape[1:] != torch.Size([Q]):
            side = int(Q ** 0.5)
            target = F.interpolate(
                target.unsqueeze(1).float(),   # (B, 1, H, W)
                size=(side, side),
                mode="nearest",
            ).long().view(B, -1)               # (B, num_queries)

        return self.loss_fn(pred.view(B * Q, C), target.view(B * Q))


class PositionLoss(nn.Module):
    """Mean squared error between predicted and target 3D positions."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred   : (batch, num_queries, 3)
            target : (batch, num_queries, 3)
        """
        return F.mse_loss(pred, target)


class SceneLoss(nn.Module):
    """
    Weighted combination of all scene prediction losses.

    Args:
        lambda_depth    (float): weight for depth loss.
        lambda_semantic (float): weight for semantic loss.
        lambda_position (float): weight for 3D position loss.
    """

    def __init__(
        self,
        lambda_depth: float = 1.0,
        lambda_semantic: float = 1.0,
        lambda_position: float = 0.5,
    ):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.lambda_semantic = lambda_semantic
        self.lambda_position = lambda_position

        self.depth_loss_fn = DepthLoss()
        self.semantic_loss_fn = SemanticLoss()
        self.position_loss_fn = PositionLoss()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict with keys 'depth', 'semantic', 'position'.
            targets    : dict with matching ground-truth tensors.

        Returns:
            dict with keys 'total', 'depth', 'semantic', 'position'.
        """
        losses = {}
        terms = []   # collect differentiable tensors so total always has a grad_fn

        if "depth" in targets and "depth" in predictions:
            losses["depth"] = self.depth_loss_fn(predictions["depth"], targets["depth"])
            terms.append(self.lambda_depth * losses["depth"])

        if "semantic" in targets and "semantic" in predictions:
            losses["semantic"] = self.semantic_loss_fn(
                predictions["semantic"], targets["semantic"]
            )
            terms.append(self.lambda_semantic * losses["semantic"])

        if "position" in targets and "position" in predictions:
            losses["position"] = self.position_loss_fn(
                predictions["position"], targets["position"]
            )
            terms.append(self.lambda_position * losses["position"])

        if not terms:
            raise RuntimeError(
                "SceneLoss: no matching targets found in this batch.\n"
                "Expected at least one of: 'depth', 'semantic', 'position'.\n"
                "Check that all .npy files exist for every sample in data/raw/."
            )

        losses["total"] = sum(terms)
        return losses
