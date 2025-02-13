"""
MaxPooling and AvePooling model for WSI         Script  ver： Nov 4th 12:30


Implementations of pooling-based multiple instance learning baseline.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# For convenience, import all path to sys
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir))
sys.path.append(str(this_file_dir.parent))
sys.path.append(str(this_file_dir.parent.parent))
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

from ModelBase.WSI_models.WSI_Transformer.WSI_pos_embed import get_2d_sincos_pos_embed

from typing import NamedTuple

import torch
from torch import nn


class Slide_PatchEmbed(nn.Module):
    """Slide Patch Embedding"""

    def __init__(
            self,
            in_chans=1536,
            embed_dim=768,
            norm_layer=None,
            bias=True,
    ):
        super().__init__()

        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, N, D = x.shape
        x = self.proj(x)
        x = self.norm(x)
        return x


class PoolingModel(nn.Module):
    """
    Pooling baseline for multiple-instance learning.

    For certain MIL-based methods, they reduce the features into only one task token.
    We call them MTL_token_design == None, just like feature extraction methods in the MTL model building process.

    We put the MTL_token_num to the task projection in MTL model building instead of here.

    [B, N, D] -> [B, D] (Max/Ave pooling).
    """

    def __init__(
            self, ROI_feature_dim=768, embed_dim=768, pooling_methods='Ave',
            slide_pos: bool = False, slide_ngrids: int = 1000):

        super().__init__()
        self.slide_pos = slide_pos
        self.embed_dim = embed_dim
        self.ROI_feature_dim = ROI_feature_dim
        self.slide_ngrids = slide_ngrids

        if self.ROI_feature_dim == self.embed_dim:
            self.patch_embed = nn.Identity()
            # notice in this case there is no parameter in the pooling model backbone!!
        else:
            self.patch_embed = Slide_PatchEmbed(in_chans=self.ROI_feature_dim, embed_dim=self.embed_dim)

        # Optional position embeddings, aligned with Slide_Transformer_blocks
        if self.slide_pos:
            num_patches = self.slide_ngrids ** 2
            self.register_buffer('pos_embed', torch.zeros(1, num_patches, self.embed_dim), persistent=False)

        self.pooling_methods = pooling_methods

        # Initialize weights
        self.initialize_weights()

    def coords_to_pos(self, coords):
        """
        Convert coordinates to positional indices, similar to Slide_Transformer_blocks.

        Arguments:
            coords: torch.Tensor - Coordinates of the patches, shape [B, N, 2]
        Returns:
            torch.Tensor - Positional indices, shape [B, N]
        """
        coords_ = torch.floor(coords / 256.0)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        return pos.long()

    def initialize_weights(self):
        if self.slide_pos:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.slide_ngrids, add_token=0)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, coords=None):
        """
        Forward pass for the Pooling MIL model.
        [B, N, D] -> [B, D] (Max/Ave pooling).

        Arguments:
            x: torch.Tensor - Input features, shape [batch_size, tile_num, in_features].
            coords: torch.Tensor - Coordinates for position embeddings, optional.

        Returns:
            torch.Tensor - Encoded and attended features.
        """
        # Feature embedding
        x = self.patch_embed(x)

        # Apply positional embeddings if enabled
        if self.slide_pos:
            if coords is None:
                raise ValueError("Coordinates must be provided for positional embeddings when slide_pos is enabled.")

            pos = self.coords_to_pos(coords)
            x = x + self.pos_embed[:, pos, :].squeeze(0)

        # Feature Pooling
        if self.pooling_methods == 'Ave':
            x = x.mean(dim=1)  # Average pooling across the tile dimension
        elif self.pooling_methods == 'Max':
            x, _ = x.max(dim=1)  # Max pooling across the tile dimension
        else:
            print(f'Pooling method "{self.pooling_methods}" is not implemented.')
            raise NotImplementedError

        return x  # Output shape: [batch_size, embed_dim]


if __name__ == "__main__":
    # Check for CUDA availability
    print('CUDA availability:', torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    default_ROI_feature_dim = 1536
    slide_embed_dim = 768

    # Create the model
    model = PoolingModel(ROI_feature_dim=default_ROI_feature_dim,embed_dim=slide_embed_dim,pooling_methods='Ave')
    model.to(dev)
    model.train()  # Set model to training mode

    # Generate example data
    coords = torch.randn(2, 1234, 2).to(dev)
    x = torch.randn(2, 1234, default_ROI_feature_dim).to(dev)

    # Forward pass
    out = model(x, coords)

    print("Output shape:", out.shape)

    # Here we need a proper target for calculating loss
    # For demonstration, let's create a dummy target with the same shape as `out`
    target = torch.randn(out.shape).to(dev)

    # Define a loss function (using MSELoss as an example)
    loss_fn = nn.MSELoss()
    loss = loss_fn(out, target)  # Calculate the loss with the target

    # Backward pass
    loss.backward()
    print("Test successful!")

