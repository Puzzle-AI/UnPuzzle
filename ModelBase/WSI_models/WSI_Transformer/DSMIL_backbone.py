# BigModel import
from __future__ import annotations
import os
import sys
# from pathlib import Path

# # For convenience, import all path to sys
# this_file_dir = Path(__file__).resolve().parent
# sys.path.append(str(this_file_dir))
# sys.path.append(str(this_file_dir.parent))
# sys.path.append(str(this_file_dir.parent.parent))
# sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

# DSMIL import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from ModelBase.WSI_models.WSI_Transformer.WSI_pos_embed import get_2d_sincos_pos_embed
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, add_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, slide_embed_dim] or [1+grid_size*grid_size, slide_embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_token > 0:
        pos_embed = np.concatenate([np.zeros([add_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    slide_embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x D
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # B x N x D, B x N x C, Q=128
        device = feats.device
        V = self.v(feats)  # B x N x D, unsorted
        # print("V = self.v(feats):", V.shape)
        Q = self.q(feats)  # .view(feats.shape[0], -1) # B x N x Q, unsorted
        # print("Q = self.q(feats):", Q.shape)

        class_num = c.shape[2]

        # handle multiple classes without for loop

        # _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        # m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x D

        # initialize result list
        m_indices_list = []
        m_feats_list = []

        # iterate through each batch
        for b in range(c.shape[0]):
            # get the data of the current batch
            curr_c = c[b]  # N x C
            curr_feats = feats[b]  # N x D

            _, curr_m_indices = torch.sort(curr_c, 0, descending=True)  # N x C
            curr_m_feats = torch.index_select(curr_feats, dim=0, index=curr_m_indices[0, :])  # C x D

            m_indices_list.append(curr_m_indices)
            m_feats_list.append(curr_m_feats)

        m_indices = torch.stack(m_indices_list, dim=0)  # B x N x C
        # print("m_indices:", m_indices.shape)
        m_feats = torch.stack(m_feats_list, dim=0)  # B x C x D
        # print("m_feats:", m_feats.shape)

        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape B x C x Q
        # print("q_max = self.q(m_feats):", q_max.shape)
        A = torch.bmm(Q, q_max.transpose(1,
                                         2))  # compute inner product of Q to each entry of q_max, A in shape B x N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[2], dtype=torch.float32, device=device)),
                      1)  # normalize attention scores, A in shape B x N x C,
        B = torch.bmm(A.transpose(1, 2), V)  # compute bag representation, B in shape B x C x D

        # B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x D
        C = self.fcc(B)  # B x C x 1
        C = C.view(B.shape[0], -1)  # B x C
        return C, A, B


class DSMIL(nn.Module):
    def __init__(self,
                 in_features: int,
                 feats_size=512,
                 num_classes=2,
                 dropout_node=200,
                 non_linearity=1,
                 slide_pos: bool = False,
                 slide_ngrids: int = 1000,
                 dropout: float = 0.25
                 ):

        super(DSMIL, self).__init__()

        self.in_features = in_features
        self.feats_size = feats_size
        self.num_classes = num_classes
        self.dropout_node = dropout_node
        self.non_linearity = non_linearity
        self.slide_pos = slide_pos  # default to be false for this permutation invariant method
        self.slide_ngrids = slide_ngrids
        self.dropout = dropout

        self.encoder = nn.Sequential(nn.Linear(in_features, feats_size),
                                     nn.ReLU(),
                                     nn.Dropout(dropout))

        self.i_classifier = FCLayer(in_size=feats_size,
                                    out_size=num_classes)

        self.b_classifier = BClassifier(input_size=feats_size,
                                        output_class=num_classes,
                                        dropout_v=dropout_node,
                                        nonlinear=non_linearity)

        # Optional position embeddings, aligned with Slide_Transformer_blocks
        if self.slide_pos:
            num_patches = self.slide_ngrids ** 2
            self.register_buffer('pos_embed', torch.zeros(1, num_patches, self.feats_size), persistent=False)

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

    # initialize (and freeze) pos_embed by sin-cos embedding
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

    def forward(self, x, coords=None):

        # Feature encoding
        x = self.encoder(x)  # Output shape: [batch_size, tile_num, feats_size]

        # Apply positional embeddings if enabled
        if self.slide_pos:
            if coords is None:
                raise ValueError("Coordinates must be provided for positional embeddings when slide_pos is enabled.")

            pos = self.coords_to_pos(coords)
            x = x + self.pos_embed[:, pos, :].squeeze(0)

        feats, classes = self.i_classifier(x)
        # print("classes:", classes.shape)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        # print("prediction_bag:", prediction_bag.shape)
        # print("normalize attention scores:", A.shape)
        # print("bag representation:", B.shape)

        return B


if __name__ == "__main__":
    # cuda issue
    print('cuda avaliablity:', torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MTL_token_num = 20
    ROI_feature_dim = 512

    model = DSMIL(
        feats_size=ROI_feature_dim,
        num_classes=MTL_token_num
    )

    model.to(dev)
    # play data (to see that Transformer can do multiple length, therefore can do MTL easily)

    x = torch.randn(3, 123, ROI_feature_dim).to(dev)
    coords = torch.randn(3, 123, 2).to(dev)

    y = model(x, coords=coords)
    print("Shape of return_features:", y.shape)

    print("Test sucessful!")

    loss = y[0].sum()
    loss.backward()

'''
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)        
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()

'''
