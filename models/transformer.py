"""
Full Transformer Model
-----------------------
Combines the encoder and decoder stacks with input/output embeddings
and positional encoding into a complete sequence-to-sequence model.

Architecture (Vaswani et al., 2017):
  - Input Embedding  + Positional Encoding
  - N Encoder Layers
  - Output Embedding + Positional Encoding
  - N Decoder Layers
  - Linear projection + Softmax  →  output probabilities

For 3D scene prediction, the "sequence" is a set of image patches
(following ViT – Dosovitskiy et al., 2020), and the decoder produces
structured 3D scene tokens (e.g. object positions, depth maps).
"""

import torch
import torch.nn as nn
from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer.

    Args:
        src_vocab_size  (int): source vocabulary / patch-feature size.
        tgt_vocab_size  (int): target vocabulary / output token size.
        d_model         (int): embedding dimensionality.
        num_heads       (int): number of attention heads.
        num_enc_layers  (int): encoder stack depth.
        num_dec_layers  (int): decoder stack depth.
        d_ff            (int): feed-forward hidden size.
        max_seq_len     (int): maximum sequence length for positional encoding.
        dropout         (float): dropout probability.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Linear(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding for both encoder and decoder
        self.src_pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.tgt_pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder and decoder stacks
        self.encoder = TransformerEncoder(num_enc_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_dec_layers, d_model, num_heads, d_ff, dropout)

        # Final linear projection to vocabulary logits
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for all linear layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates a causal (lower-triangular) mask to prevent the decoder from
        attending to future positions.

        Returns:
            mask: (1, 1, seq_len, seq_len) boolean tensor.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def encode(
        self, src: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Run the encoder on source features."""
        x = self.src_pos_enc(self.src_embedding(src))
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, list]:
        """Run the decoder."""
        x = self.tgt_pos_enc(self.tgt_embedding(tgt))
        return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            src      : (batch, src_len, src_feat) – source image features.
            tgt      : (batch, tgt_len) – target token indices.
            src_mask : optional source mask.
            tgt_mask : optional causal target mask (auto-generated if None).

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        if tgt_mask is None:
            tgt_mask = self.make_causal_mask(tgt.size(1), tgt.device)

        encoder_output = self.encode(src, src_mask)
        decoder_output, _ = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)
