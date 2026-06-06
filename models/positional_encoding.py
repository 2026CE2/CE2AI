"""
Positional Encoding
--------------------
Transformers process all tokens in parallel; there is no built-in notion
of word order. We inject positional information by adding a fixed vector
to each token embedding.

Sinusoidal encoding (Vaswani et al., 2017):

    PE[pos, 2k]   = sin(pos / 10000^(2k / d_model))
    PE[pos, 2k+1] = cos(pos / 10000^(2k / d_model))

Properties:
  - Each position gets a unique vector.
  - Generalises to sequence lengths not seen at training time.
  - No extra learnable parameters.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Args:
        d_model (int): embedding dimension.
        max_len (int): maximum sequence length.
        dropout (float): dropout applied after adding the encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the encoding matrix once and register as a non-learnable buffer
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # Compute the frequency terms: 1 / 10000^(2k / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token embeddings of shape (batch, seq_len, d_model).

        Returns:
            Embeddings with positional information added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embedding (alternative to sinusoidal).

    Note: Unlike sinusoidal encoding, this does NOT generalise to unseen
    sequence lengths and introduces extra parameters.

    Args:
        d_model (int): embedding dimension.
        max_len (int): maximum sequence length.
        dropout (float): dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.embedding(positions)
        return self.dropout(x)
