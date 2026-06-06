"""
Transformer Encoder
--------------------
Each encoder layer consists of:
  1. Multi-Head Self-Attention sub-layer
  2. Position-wise Feed-Forward Network sub-layer

Both sub-layers use residual (skip) connections followed by Layer Norm:
    output = LayerNorm(x + SubLayer(x))

Residual connections (He et al., 2016 – ResNet) help with gradient
propagation and allow training deeper networks.

The full encoder stacks N such layers (N=6 in the original paper).
"""

import torch
import torch.nn as nn
from models.multi_head_attention import MultiHeadAttention
from models.feed_forward import FeedForwardNetwork


class EncoderLayer(nn.Module):
    """
    Single encoder layer: self-attention + FFN, each wrapped in Add & Norm.

    Args:
        d_model (int): model dimensionality.
        num_heads (int): number of attention heads.
        d_ff (int): feed-forward inner dimension.
        dropout (float): dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x        : (batch, seq_len, d_model)
            src_mask : optional padding mask for source sequence.

        Returns:
            (batch, seq_len, d_model)
        """
        # Sub-layer 1: self-attention with residual connection
        attn_out, _ = self.self_attention(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class TransformerEncoder(nn.Module):
    """
    Stack of N encoder layers.

    Args:
        num_layers (int): number of encoder layers (N).
        d_model (int): model dimensionality.
        num_heads (int): number of attention heads.
        d_ff (int): feed-forward inner dimension.
        dropout (float): dropout probability.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x        : (batch, seq_len, d_model)
            src_mask : optional mask.

        Returns:
            (batch, seq_len, d_model) – contextual representations.
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
