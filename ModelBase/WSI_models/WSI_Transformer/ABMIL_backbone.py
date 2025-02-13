"""
ABMIL model          Script  ver： Feb 4th 12:30

http://proceedings.mlr.press/v80/ilse18a.html
Implementations of attention-based multiple instance learning models.
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
from torch.nn import functional as F


def AttentionLayer(L: int, D: int, K: int):
    """Attention layer (without gating)."""
    return nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Linear(D, K))  # NxK


class GatedAttentionLayer(nn.Module):
    """Gated attention layer."""

    def __init__(self, L: int, D: int, K: int, *, dropout: float = 0.25):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K
        self.dropout = dropout

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Dropout(self.dropout),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication. NxK
        return A


class AttentionMILModelOutput(NamedTuple):
    """A container for the outputs of an attention MIL model."""

    logits: torch.Tensor
    attention: torch.Tensor


class AttentionMILModel_ori(nn.Module):
    """Attention multiple-instance learning model."""

    def __init__(
            self,
            *,
            in_features: int,
            L: int,
            D: int,
            K: int = 1,
            dropout: float = 0.25,
            gated_attention: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.L = L
        self.D = D
        self.K = K
        # self.num_classes = num_classes
        self.dropout = dropout
        self.gated_attention = gated_attention

        self.encoder = nn.Sequential(nn.Linear(in_features, L), nn.ReLU(), nn.Dropout(dropout))

        if gated_attention:
            self.attention_weights = GatedAttentionLayer(L=L, D=D, K=K, dropout=dropout)
        else:
            self.attention_weights = AttentionLayer(L=L, D=D, K=K)

    def forward(self, H: torch.Tensor, coords=None) -> AttentionMILModelOutput:
        # H.shape is N x in_features
        H = H.squeeze(0)  # Remove batch dimension introduced by DataLoader.
        if H.ndim != 2:
            raise ValueError(f"Expected H to have 2 dimensions but got {H.ndim}")

        H = self.encoder(H)  # NxL
        A = self.attention_weights(H)  # NxK
        A_raw = A
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        return [M]


class AttentionMILModel(nn.Module):
    """Attention multiple-instance learning model with a structure similar to Slide_Transformer_blocks.
    
    notice: although we have slide_pos for location embedding, the ABMIL is designed to be permutation invariant,
    so by default we dont need to have this
    """

    def __init__(
            self, in_features: int, L: int, D: int, dropout: float = 0.25, gated_attention: bool = True,
            slide_pos: bool = False, slide_ngrids: int = 1000, MTL_token_num: int = 0):

        super().__init__()
        self.in_features = in_features
        self.L = L
        self.D = D
        self.K = MTL_token_num if MTL_token_num > 0 else 1  # at least 1 for MIL
        '''
        for some MIL-based method, they reduce the features into several task tokens (at least 1)
        we call them as MTL_token_design == "MIL_to" in the MTL model building process
        
        in model design, we put the MTL_token_num to the task projection in their model
        '''
        self.dropout = dropout
        self.gated_attention = gated_attention
        self.slide_pos = slide_pos  # default to be false for this permutation invariant method
        self.slide_ngrids = slide_ngrids
        self.MTL_token_num = self.K

        # Define encoder similar to patch_embed structure
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, L),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # Attention layers
        if self.gated_attention:
            self.attention_weights = GatedAttentionLayer(L=self.L, D=self.D, K=self.K, dropout=self.dropout)
        else:
            self.attention_weights = AttentionLayer(L=self.L, D=self.D, K=self.K)

        # Optional position embeddings, aligned with Slide_Transformer_blocks
        if self.slide_pos:
            num_patches = self.slide_ngrids ** 2
            self.register_buffer('pos_embed', torch.zeros(1, num_patches, self.L), persistent=False)

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
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.slide_ngrids,
                                                add_token=0)
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

    def forward(self, H: torch.Tensor, coords=None):
        """
        Forward pass for the adapted Attention MIL model.

        Arguments:
            H: torch.Tensor - Input features, shape [batch_size, tile_num, in_features]
            coords: torch.Tensor - Coordinates for position embeddings, optional

        Returns:
            torch.Tensor - Encoded and attended features.
        """

        # Feature encoding
        H = self.encoder(H)  # Output shape: [batch_size, tile_num, L]

        # Apply positional embeddings if enabled
        if self.slide_pos:
            if coords is None:
                raise ValueError("Coordinates must be provided for positional embeddings when slide_pos is enabled.")

            pos = self.coords_to_pos(coords)
            H = H + self.pos_embed[:, pos, :].squeeze(0)

        # Apply attention weights
        A = self.attention_weights(H)  # Shape: [batch_size, tile_num, K]
        A = torch.transpose(A, 1, 2)  # Shape: [batch_size, K, tile_num]
        A = F.softmax(A, dim=2)  # Softmax over tile_num

        # Reshape H for batched matrix multiplication
        M = torch.bmm(A, H)  # Output shape: [batch_size, K, L]

        return M


if __name__ == "__main__":
    # cuda issue
    print('cuda availability:', torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MTL_token_num = 20
    default_ROI_feature_dim = 768
    slide_embed_dim = 768

    model = AttentionMILModel(in_features=default_ROI_feature_dim, L=slide_embed_dim, D=384,
                              gated_attention=True, MTL_token_num=MTL_token_num)
    model.to(dev)
    # play data (to see that Transformer can do multiple length, therefore can do MTL easily)
    coords = torch.randn(2, 1234, 2).to(dev)
    x = torch.randn(2, 1234, default_ROI_feature_dim).to(dev)

    out = model.forward(x, coords)
    print(out.shape)
    loss = out.sum()
    loss.backward()
    print("Test successful!")
