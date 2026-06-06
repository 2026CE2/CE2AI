"""
Evaluation Metrics
-------------------
Standard metrics for each prediction head.

Depth estimation:
  - AbsRel : absolute relative error = mean(|d - d*| / d*)
  - RMSE   : root mean squared error
  - delta_1: % pixels where max(d/d*, d*/d) < 1.25

Semantic segmentation:
  - mIoU   : mean intersection-over-union across classes
  - Accuracy: pixel accuracy

3D Position estimation:
  - Mean Euclidean Distance (MED) between predicted and true (x, y, z).
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Depth metrics
# ---------------------------------------------------------------------------

def depth_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """
    Compute standard monocular depth estimation metrics.

    Args:
        pred   : (batch, N, 1) or (batch, N) predicted depth.
        target : (batch, N, 1), (batch, N), or (batch, H, W) ground-truth depth.
                 Full-resolution spatial maps are downsampled to match pred.

    Returns:
        dict with keys 'abs_rel', 'rmse', 'delta_1'.
    """
    pred = pred.squeeze(-1).float()      # (batch, N)
    target = target.float()

    # Align full-resolution spatial target (batch, H, W) to (batch, N)
    if target.dim() == 3 and target.shape[1:] != pred.shape[1:]:
        num_patches = pred.size(1)
        side = int(num_patches ** 0.5)
        target = F.interpolate(
            target.unsqueeze(1),          # (batch, 1, H, W)
            size=(side, side),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)                      # (batch, side*side)
        target = target.view(target.size(0), -1)  # (batch, N)

    target = target.squeeze(-1)

    mask = target > 0
    pred_m = pred[mask]
    target_m = target[mask]

    abs_rel = ((pred_m - target_m).abs() / target_m).mean().item()
    rmse = ((pred_m - target_m) ** 2).mean().sqrt().item()

    ratio = torch.max(pred_m / target_m, target_m / pred_m)
    delta_1 = (ratio < 1.25).float().mean().item()

    return {"abs_rel": abs_rel, "rmse": rmse, "delta_1": delta_1}


# ---------------------------------------------------------------------------
# Semantic segmentation metrics
# ---------------------------------------------------------------------------

def semantic_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, float]:
    """
    Compute pixel accuracy and mean IoU.

    Args:
        pred_logits : (batch, num_queries, num_classes) logits.
        target      : (batch, num_queries) integer class indices.
        num_classes : number of semantic classes.
        ignore_index: class index to ignore (default 255).

    Returns:
        dict with keys 'accuracy' and 'miou'.
    """
    B, Q, C = pred_logits.shape
    pred = pred_logits.argmax(dim=-1)  # (batch, num_queries)

    # Downsample full-resolution mask (batch, H, W) to (batch, num_queries)
    if target.dim() == 3 and target.shape[1:] != pred.shape[1:]:
        side = int(Q ** 0.5)
        target = F.interpolate(
            target.unsqueeze(1).float(),   # (batch, 1, H, W)
            size=(side, side),
            mode="nearest",
        ).long().view(B, -1)              # (batch, num_queries)

    mask = target != ignore_index
    correct = (pred[mask] == target[mask]).float().sum().item()
    total = mask.sum().item()
    accuracy = correct / (total + 1e-9)

    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        tgt_cls = target == cls
        intersection = (pred_cls & tgt_cls & mask).sum().float().item()
        union = ((pred_cls | tgt_cls) & mask).sum().float().item()
        if union > 0:
            iou_per_class.append(intersection / union)

    miou = sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0.0

    return {"accuracy": accuracy, "miou": miou}


# ---------------------------------------------------------------------------
# 3-D position metrics
# ---------------------------------------------------------------------------

def position_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """
    Compute mean Euclidean distance between predicted and true 3D positions.

    Args:
        pred   : (batch, num_queries, 3) predicted positions.
        target : (batch, num_queries, 3) ground-truth positions.

    Returns:
        dict with key 'med' (mean Euclidean distance).
    """
    distance = (pred - target).pow(2).sum(dim=-1).sqrt()  # (batch, num_queries)
    med = distance.mean().item()
    return {"med": med}
