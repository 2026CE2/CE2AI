"""
Position-wise Feed-Forward Network
------------------------------------
After each multi-head attention sub-layer, a two-layer MLP is applied
independently to each position:

    FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2

    x  in R^(n x d_model)
    W_1 in R^(d_model x d_ff)
    W_2 in R^(d_ff x d_model)

Original paper settings: d_model = 512, d_ff = 2048.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """
    Two-layer position-wise feed-forward network.

    Args:
        d_model (int): input / output dimensionality.
        d_ff (int): inner (hidden) dimensionality.
        dropout (float): dropout probability.
        activation (str): 'relu' or 'gelu'.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'relu' or 'gelu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
