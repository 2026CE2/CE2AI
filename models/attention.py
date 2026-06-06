"""
Scaled Dot-Product Attention
-----------------------------
Based on: Vaswani et al. (2017) "Attention is All You Need"

Given query Q, key K, and value V matrices:

    A(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

- Q: query matrix  (batch, seq_len, d_k)
- K: key matrix    (batch, seq_len, d_k)
- V: value matrix  (batch, seq_len, d_v)

Dividing by sqrt(d_k) prevents dot products from growing too large,
which would push softmax into regions with very small gradients.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.

    Args:
        dropout (float): dropout probability applied to attention weights.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query : (batch, heads, seq_q, d_k)
            key   : (batch, heads, seq_k, d_k)
            value : (batch, heads, seq_k, d_v)
            mask  : (batch, 1, seq_q, seq_k) optional boolean mask;
                    True positions are masked out (set to -inf before softmax).

        Returns:
            output          : (batch, heads, seq_q, d_v)
            attention_weights: (batch, heads, seq_q, seq_k)
        """
        d_k = query.size(-1)

        # Step 1: dot product between Q and K^T
        scores = torch.matmul(query, key.transpose(-2, -1))  # (..., seq_q, seq_k)

        # Step 2: scale
        scores = scores / math.sqrt(d_k)

        # Step 3: optional mask (used in decoder to prevent attending to future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 4: softmax to get attention weights that sum to 1
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Step 5: weighted sum of values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
