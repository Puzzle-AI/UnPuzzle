import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


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
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self,
                 in_chans=1280,  # args.MODEL_T2T.NUM_INPUT_CHANNELS, feature_dim of each patch
                 embed_dim=512,  # args.MODEL_T2T.EMBED_DIM
                 slide_pos=True,
                 slide_ngrids=1000,
                 n_classes=2,  # for MTL we project this to MTL tokens num
                 drop_rate=0.
                 ):

        super().__init__()
        # SSL specifics
        self.SSL = False  # enable outside if you need to warp in MAE

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.slide_pos = slide_pos
        if self.slide_pos:
            self.slide_ngrids = slide_ngrids
            num_patches = slide_ngrids ** 2
            self.register_buffer('pos_embed', torch.zeros(1, 1 + num_patches, embed_dim),
                                 persistent=False)  # fixed sin-cos embedding
        # --------------------------------------------------------------------------
        self.n_classes = n_classes

        self.pos_layer = PPEG(dim=self.embed_dim)
        # self.patch_embed, embed_dim fix to 512
        self._fc1 = nn.Sequential(nn.Linear(in_chans, self.embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.layer1 = TransLayer(dim=self.embed_dim)
        self.layer2 = TransLayer(dim=self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

        # CLS layer
        # self._fc2 = nn.Linear(embed_dim, self.n_classes)

        # trunc_normal_(self.cls_token, std=.02)
        self.initialize_vit_weights()

    def initialize_vit_weights(self):
        # initialization
        if self.slide_pos:
            # initialize (and freeze) pos_embed by sin-cos embedding, add one cls token
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.slide_ngrids, add_token=1)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def coords_to_pos(self, coords):
        """
        This function is used to convert the coordinates to the positional indices

        Arguments:
        ----------
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        output: torch.Tensor
            The positional indices of the patches, of shape [N, L]
        """
        coords_ = torch.floor(coords / 256.0)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        return pos.long()

    def forward(self, x, coords):
        """"
        The forward pass of the model

        Arguments:
        ----------
        x: torch.Tensor
            The input tile embeddings, of shape [B, n, in_chans]
        coords: torch.Tensor
            The coordinates of the patches, of shape [B, n, 2]
        """

        # x [B, n, in_chans]

        h = self._fc1(x)  # h [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        if self.slide_pos:
            assert coords != None
            coords = torch.cat([coords, coords[:, :add_length, :]], dim=1)  # [B, N, 2]
            # get pos indices
            pos = self.coords_to_pos(coords)  # [B, N]
            h = h + self.pos_embed[:, pos, :].squeeze(0)
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        else:
            cls_token = self.cls_token

        # ---->concatenate the cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        x = self.norm(h)

        # ---->features
        # [B, num_classes, D]
        if not self.SSL:
            # [B, 1, D] take the CLS token
            outcome = x[:, 0]
        else:
            # all embedding shape print(x.shape)  [B, num_classes, D]
            outcome = x
        return outcome


if __name__ == "__main__":
    # cuda issue
    print('cuda avaliablity:', torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MTL_token_num = 20
    ROI_feature_dim = 51
    sllide_embed_dim = 768

    # slide_task_model = build_WSI_task_model(
    #     MTL_token_design=None,
    #     model_name="gigapath",
    #     local_weight_path=None,
    #     ROI_feature_dim=ROI_feature_dim,
    #     latent_feature_dim=128,
    #     MTL_heads=[nn.Linear(128, 3), nn.Linear(128, 4)],
    # )

    model = TransMIL(
        in_chans=ROI_feature_dim,
        embed_dim=sllide_embed_dim,
        n_classes=2
    )

    model.to(dev)
    # play data (to see that Transformer can do multiple length, therefore can do MTL easily)

    x = torch.randn(3, 123, ROI_feature_dim).to(dev)
    coords = torch.randn(3, 123, 2).to(dev)

    y = model(x, coords=coords)
    print("Shape of return_features:", y.shape)  # [B,D] as taking the CLS token out

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
