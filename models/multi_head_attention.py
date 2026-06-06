"""
Multi-Head Attention
--------------------
Instead of computing attention once in d_m dimensions, we split the work
across h heads each operating in d_m/h dimensions, then concatenate and
project back.

    MHA(Q, K, V) = Concat(head_1, ..., head_h) * W_O
    head_i       = Attention(Q * W_i_Q, K * W_i_K, V * W_i_V)

From the original paper (Vaswani et al., 2017):
    h = 8,  d_k = d_v = d_m / h = 64

Benefits:
  - Different heads can capture different types of relationships.
  - Total computation is the same as single-head attention.
"""

import torch
import torch.nn as nn
from models.attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    Args:
        d_model (int): model dimensionality (d_m).
        num_heads (int): number of attention heads (h).
        dropout (float): dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Learnable projection matrices W_Q, W_K, W_V for all heads at once
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Final output projection W_O
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dimension into (num_heads, d_k) and transpose."""
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq_len, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse of _split_heads."""
        batch, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query : (batch, seq_q, d_model)
            key   : (batch, seq_k, d_model)
            value : (batch, seq_k, d_model)
            mask  : optional attention mask.

        Returns:
            output          : (batch, seq_q, d_model)
            attention_weights: (batch, heads, seq_q, seq_k)
        """
        # Step 1: linear projections + split into h heads
        Q = self._split_heads(self.W_Q(query))
        K = self._split_heads(self.W_K(key))
        V = self._split_heads(self.W_V(value))

        # Step 2: scaled dot-product attention on each head in parallel
        attn_output, attention_weights = self.attention(Q, K, V, mask=mask)

        # Step 3: concatenate heads and apply final linear projection
        output = self.W_O(self._merge_heads(attn_output))

        return output, attention_weights
