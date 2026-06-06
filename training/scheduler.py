"""
Learning-Rate Scheduler
------------------------
The original transformer paper (Vaswani et al., 2017) uses a custom
warm-up schedule:

    lr(step) = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

This increases the learning rate linearly for the first warmup_steps, then
decreases it proportionally to the inverse square root of the step number.
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_transformer_scheduler(
    optimizer: Optimizer,
    d_model: int,
    warmup_steps: int = 4000,
) -> LambdaLR:
    """
    Create the Noam learning-rate schedule from "Attention is All You Need".

    Args:
        optimizer     : the optimiser to attach the schedule to.
        d_model       : model dimensionality (controls scale).
        warmup_steps  : number of linear warm-up steps.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        # Avoid division by zero at step 0
        step = max(current_step, 1)
        scale = d_model ** (-0.5)
        return scale * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_scheduler(
    optimizer: Optimizer,
    num_epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
) -> LambdaLR:
    """
    Cosine annealing schedule with linear warm-up.

    Args:
        optimizer     : optimiser.
        num_epochs    : total training epochs.
        warmup_epochs : linear warm-up period in epochs.
        min_lr        : minimum learning rate (as fraction of base lr).
    """
    import math

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
