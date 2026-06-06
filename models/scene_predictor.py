"""
3D Scene Predictor
-------------------
Main model that takes a single RGB image and predicts a structured
3D scene representation (depth + semantic segmentation + object poses).

Architecture (Vision Transformer – ViT style, Dosovitskiy et al., 2020):

  1. Patch Embedding
     - Split image (H x W x 3) into fixed-size patches (P x P x 3).
     - Flatten each patch and project linearly → d_model dimensional token.

  2. Positional Encoding (sinusoidal)
     - Added to patch tokens so the model knows spatial arrangement.

  3. Transformer Encoder
     - Self-attention over all patch tokens in parallel.
     - Produces contextual patch representations.

  4. Learned Query Tokens (decoder queries)
     - A set of learnable vectors representing output 3D scene tokens.

  5. Transformer Decoder
     - Cross-attends to encoder output.
     - Produces per-query scene features.

  6. Output Heads
     - Depth head      : predicts per-pixel depth map.
     - Semantic head   : predicts per-pixel class logits.
     - 3D Position head: predicts (x, y, z) object centres.
"""

import torch
import torch.nn as nn
from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.positional_encoding import PositionalEncoding


class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and projects each to d_model.

    Args:
        image_size  (int): input image height/width (assumed square).
        patch_size  (int): patch height/width.
        in_channels (int): image channels (3 for RGB).
        d_model     (int): output embedding dimension.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 512,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."

        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, H, W)

        Returns:
            patches: (batch, num_patches, d_model)
        """
        B, C, H, W = x.shape
        P = self.patch_size

        # Rearrange into patches: (B, num_patches, patch_dim)
        x = x.unfold(2, P, P).unfold(3, P, P)  # (B, C, H/P, W/P, P, P)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * P * P)

        return self.projection(x)


class ScenePredictor(nn.Module):
    """
    Transformer-based 3D scene predictor from a single RGB image.

    Args:
        image_size      (int): input image size (square assumed).
        patch_size      (int): patch size.
        d_model         (int): transformer model dimension.
        num_heads       (int): attention heads.
        num_enc_layers  (int): encoder depth.
        num_dec_layers  (int): decoder depth.
        d_ff            (int): feed-forward hidden size.
        num_queries     (int): number of output scene query tokens.
        num_classes     (int): number of semantic segmentation classes.
        dropout         (float): dropout probability.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        d_model: int = 512,
        num_heads: int = 8,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6,
        d_ff: int = 2048,
        num_queries: int = 100,
        num_classes: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        num_patches = (image_size // patch_size) ** 2

        # --- Input side ---
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=num_patches + 1, dropout=dropout)

        # --- Transformer backbone ---
        self.encoder = TransformerEncoder(num_enc_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_dec_layers, d_model, num_heads, d_ff, dropout)

        # Learnable query tokens for the decoder (one per output scene entity)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # --- Output heads ---
        # Depth estimation head (regresses a depth value per patch).
        # Softplus ensures output is always positive (required for log-based loss).
        self.depth_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

        # Semantic segmentation head (classifies each query / patch)
        self.semantic_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes),
        )

        # 3D position head (predicts normalised (x, y, z) object centre)
        self.position_3d_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),
            nn.Sigmoid(),  # normalise to [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            images: (batch, 3, H, W) – RGB images.

        Returns:
            dict with keys:
                'depth'    : (batch, num_patches, 1)
                'semantic' : (batch, num_queries, num_classes)
                'position' : (batch, num_queries, 3)
        """
        batch = images.size(0)

        # 1. Patch embedding + positional encoding
        patches = self.patch_embed(images)              # (B, num_patches, d_model)
        patches = self.pos_enc(patches)

        # 2. Encode patches with self-attention
        memory = self.encoder(patches)                   # (B, num_patches, d_model)

        # 3. Expand learnable queries for the batch
        queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)

        # 4. Decode: cross-attend queries over encoded patch memory
        scene_features, _ = self.decoder(queries, memory)  # (B, num_queries, d_model)

        return {
            "depth": self.depth_head(memory),            # per-patch depth
            "semantic": self.semantic_head(scene_features),  # per-query class logits
            "position": self.position_3d_head(scene_features),  # 3D positions
        }
