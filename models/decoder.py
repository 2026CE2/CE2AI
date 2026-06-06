"""
Transformer Decoder
--------------------
Each decoder layer has three sub-layers:
  1. Masked Multi-Head Self-Attention (auto-regressive masking)
  2. Multi-Head Cross-Attention over encoder output (K, V from encoder)
  3. Position-wise Feed-Forward Network

The self-attention in the decoder is *masked* so position i can only
attend to positions <= i, preserving the auto-regressive property.

    A = softmax(mask(Q * K^T / sqrt(d_k))) * V

All sub-layers use residual connections + Layer Norm.
"""

import torch
import torch.nn as nn
from models.multi_head_attention import MultiHeadAttention
from models.feed_forward import FeedForwardNetwork


class DecoderLayer(nn.Module):
    """
    Single decoder layer: masked self-attention + cross-attention + FFN.

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
        # Sub-layer 1: masked self-attention (auto-regressive)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        # Sub-layer 2: cross-attention over encoder output
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        # Sub-layer 3: feed-forward
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x              : (batch, tgt_len, d_model) – decoder input.
            encoder_output : (batch, src_len, d_model) – from encoder stack.
            src_mask       : mask for encoder keys (padding mask).
            tgt_mask       : causal mask preventing decoder from attending to future.

        Returns:
            x                   : (batch, tgt_len, d_model)
            self_attn_weights   : (batch, heads, tgt_len, tgt_len)
            cross_attn_weights  : (batch, heads, tgt_len, src_len)
        """
        # Sub-layer 1: masked self-attention
        self_attn_out, self_attn_w = self.self_attention(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Sub-layer 2: cross-attention (Q from decoder, K/V from encoder)
        cross_attn_out, cross_attn_w = self.cross_attention(
            x, encoder_output, encoder_output, mask=src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Sub-layer 3: feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x, self_attn_w, cross_attn_w


class TransformerDecoder(nn.Module):
    """
    Stack of N decoder layers.

    Args:
        num_layers (int): number of decoder layers.
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
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            x              : (batch, tgt_len, d_model)
            encoder_output : (batch, src_len, d_model)
            src_mask       : padding mask.
            tgt_mask       : causal mask.

        Returns:
            x              : (batch, tgt_len, d_model)
            attention_maps : list of (self_attn, cross_attn) per layer.
        """
        attention_maps = []
        for layer in self.layers:
            x, self_w, cross_w = layer(x, encoder_output, src_mask, tgt_mask)
            attention_maps.append((self_w, cross_w))
        return self.norm(x), attention_maps
