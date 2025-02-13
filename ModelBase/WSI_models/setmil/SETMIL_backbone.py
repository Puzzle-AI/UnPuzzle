import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from .t2t_module.token_transformer import Token_transformer
from .t2t_module.token_performer import Token_performer
from .t2t_module.transformer_block import Block, get_sinusoid_encoding


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}


class ASPPBlock(nn.Module):
    def __init__(self, img_size, in_chans, embed_dim=768, K=3, S=1, P=2, D=1, **kwargs):
        super().__init__()

        self.soft_split = nn.Unfold(kernel_size=(K, K), stride=(S, S), padding=(P, P), dilation=(D, D))
        self.attention = Token_transformer(dim=in_chans * K * K, in_dim=embed_dim, num_heads=1, mlp_ratio=1.0, **kwargs)
        self.num_patches = ((img_size + 2 * P - (D * (K - 1) + 1)) // S + 1) ** 2

    def forward(self, x):
        # step0: soft split
        x = self.soft_split(x).transpose(1, 2)
        # [bs, stride_times1, kernel_size*kernel_size*in_chans]

        # iteration1: re-structurization/reconstruction
        x = self.attention(x)
        # [B, stride_times1 (new_HW), embed_dim]

        return x


class ASPP(nn.Module):
    def __init__(self, img_size, in_chans, embed_dim=768, **kwargs):
        super().__init__()
        ##self.p0 = ASPPBlock(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, K=1,)
        self.p1 = ASPPBlock(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, K=3, P=(3 - 1) // 2, **kwargs)
        self.p2 = ASPPBlock(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, K=5, P=(5 - 1) // 2, **kwargs)
        self.p3 = ASPPBlock(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, K=7, P=(7 - 1) // 2, **kwargs)
        # self.num_patches = self.p1.num_patches + self.p2.num_patches + self.p3.num_patches
        self.num_patches = self.p1.num_patches

    def forward(self, x):
        # x0 = self.p0(x)
        x1 = self.p1(x)
        x2 = self.p2(x)
        x3 = self.p3(x)
        x = torch.cat((x1, x2, x3), dim=2)
        return x


class ASPP2(nn.Module):
    """
    progressive
    todo: debug
    """

    def __init__(self, img_size, in_chans, embed_dim=768, **kwargs):
        super().__init__()
        ##self.p0 = ASPPBlock(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, K=1,)
        self.p1 = ASPPBlock(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, K=7, **kwargs)
        self.p2 = ASPPBlock(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, K=3, **kwargs)
        self.p3 = ASPPBlock(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, K=3, **kwargs)
        # self.num_patches = self.p1.num_patches + self.p2.num_patches + self.p3.num_patches
        self.num_patches = self.p1.num_patches

    def forward(self, x):
        # x0 = self.p0(x)
        x = self.p1(x)
        B, C, new_HW = x.shape
        x = x.reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = self.p2(x)
        B, C, new_HW = x.shape
        x = x.reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = self.p3(x)
        return x


class Downsample(nn.Module):
    """
    dimensionality reduction
    """

    def __init__(self, img_size, in_chans, embed_dim=768, k=3):
        super().__init__()
        self.reduce = ASPPBlock(img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, K=k, S=(k - 1) // 2,
                                P=(k - 1) // 2)
        # self.reduce = nn.Conv1d(in_chans, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.reduce(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        # B, C, H, W = x.shape
        # x = x.reshape(B, C, -1)
        # x = self.reduce(x)
        # B, C, new_HW = x.shape
        # x = x.reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        return x


class PatchEmbed(nn.Module):
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


class SETMIL(nn.Module):
    def __init__(self,
                 in_chans=1280,  # args.MODEL_T2T.NUM_INPUT_CHANNELS, feature_dim of each patch
                 embed_dim=768,  # args.MODEL_T2T.EMBED_DIM
                 slide_pos=True,
                 slide_ngrids=1000,

                 img_size=48,  # args.DATASET.FEATURE_MAP_SIZE
                 tokens_type='transformer',
                 num_classes=2,  # args.MODEL_T2T.NUM_CLASSES
                 depth=12,  # args.MODEL_T2T.DEPTH
                 num_heads=64,  # args.MODEL_T2T.NUM_HEADS
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.2,  # args.MODEL_T2T.DROP_RATE
                 attn_drop_rate=0.2,  # args.MODEL_T2T.ATTN_DROP_RATE
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 token_dim=128,  # args.MODEL_T2T.TOKEN_DIM, for feature pyramid
                 channel_reduce_rate=5,  # args.MODEL_T2T.CHANNEL_REDUCE_RATE
                 aspp_flag=1):

        super().__init__()

        # MAE encoder specifics
        self.SSL = False  # enable outside if you need to warp in MAE
        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(in_chans, embed_dim)
        self.slide_pos = slide_pos
        self.slide_ngrids = slide_ngrids
        num_patches = slide_ngrids ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer('pos_embed', torch.zeros(1, 1 + num_patches, embed_dim),
                             persistent=False)  # fixed sin-cos embedding

        irpe = {
            "rpe": 1,  # False
            "method": "euc",
            "mode": "bias",
            "shared_head": False,
            "rpe_on": "q",
        }
        token_t = {
            "rpe": 0,
            "drop": 0.2,
            "attn_drop": 0.2,
            "method": "euc",
            "mode": "bias",
            "shared_head": False,
            "rpe_on": "q",
        }

        self.aspp_flag = aspp_flag
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # self.channel_reduce = Downsample(img_size=img_size, in_chans=in_chans, embed_dim=token_dim,
        #                                  k=channel_reduce_rate)
        # if self.aspp_flag:
        #     self.aspp = ASPP(img_size=img_size//(channel_reduce_rate//2), in_chans=token_dim, embed_dim=embed_dim, **token_t)
        #     num_patches = self.aspp.num_patches
        #     decoder_embed_dim = embed_dim * 3
        # else:
        #     num_patches = (img_size//(channel_reduce_rate//2))**2
        #     decoder_embed_dim = token_dim

        decoder_embed_dim = embed_dim

        # # cls and pos setting in the original paper
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=decoder_embed_dim), requires_grad=False)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, **irpe)
            for i in range(depth)])
        self.norm = norm_layer(decoder_embed_dim)

        # Classifier head
        self.head = nn.Linear(decoder_embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.cls_token, std=.02)
        self.initialize_vit_weights()

    def initialize_vit_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.slide_ngrids, add_token=1)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
        return pos.long() + 1  # add 1 for the cls token

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, coords, MTL_tokens=None):
        # x = sample.tensors
        ## x = sample
        ##  [B, C, H, W]
        # B, C, H, W = x.shape
        ## optional if h,w are too large
        # x = self.channel_reduce(x)
        ## [B, C, sqrt(H), sqrt(W)]
        # if self.aspp_flag:
        #    x = self.aspp(x)
        # else:
        #    B, C = x.shape[:2]
        #    x = x.reshape(B, C,-1).transpose(1,2)

        # input size [B, stride_times1 (new_HW), embed_dim]

        # embed patches
        x = self.patch_embed(x)

        # get pos indices
        pos = self.coords_to_pos(coords)  # [N, L]

        x = x + self.pos_embed[:, pos, :].squeeze(0)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # # cls and pos coding in the original paper
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed # remove for ablation study

        x = self.pos_drop(x)
        # print("pos_drop(x):", x.shape)
        for blk in self.blocks:
            x = blk(x, coords)
        x = self.norm(x)
        # [B, N + 1, D]
        # return [B, D] extract the first token of x (cls_token)
        return x

    def forward(self, x, coords, MTL_tokens=None):

        """"
        The forward pass of the model

        Arguments:
        ----------
        x: torch.Tensor
            The input tile embeddings, of shape [B, N, D]
        coords: torch.Tensor
            The coordinates of the patches, of shape [B, N, 2]

        """

        x = self.forward_features(x, coords, MTL_tokens)
        # [B, D]
        # x = self.head(x)
        # # [B, num_classes]

        if not self.SSL:
            outcome = x[:, 0]
        else:
            # all embedding shape print(x.shape)
            outcome = x
        return outcome


def build(args):
    device = torch.device("cuda")

    img_size = args.DATASET.FEATURE_MAP_SIZE

    # model hyper-parameters
    num_classes = args.MODEL_T2T.NUM_CLASSES
    in_chans = args.MODEL_T2T.NUM_INPUT_CHANNELS  # feature_dim of each patch
    token_dim = args.MODEL_T2T.TOKEN_DIM  # for feature pyramid
    # for final transformer
    embed_dim = args.MODEL_T2T.EMBED_DIM
    depth = args.MODEL_T2T.DEPTH
    num_heads = args.MODEL_T2T.NUM_HEADS
    drop_rate = args.MODEL_T2T.DROP_RATE
    attn_drop_rate = args.MODEL_T2T.ATTN_DROP_RATE
    # irpe_method = args.MODEL_T2T.IRPE_METHOD
    # irpe_mode = args.MODEL_T2T.IRPE_MODE
    # irpe_shared_head = args.MODEL_T2T.IRPE_SHARE_HEAD
    # irpe_rpe_on = args.MODEL_T2T.IRPE_RPE_ON
    irpe = {
        "rpe": args.MODEL_T2T.IRPE,
        "method": args.MODEL_T2T.IRPE_METHOD,
        "mode": args.MODEL_T2T.IRPE_MODE,
        "shared_head": args.MODEL_T2T.IRPE_SHARE_HEAD,
        "rpe_on": args.MODEL_T2T.IRPE_RPE_ON,
    }
    # for token transformer
    # token_t_drop_rate = args.MODEL_T2T.TOKEN_T_DROP_RATE
    # token_t_attn_drop_rate = args.MODEL_T2T.TOKEN_T_ATTN_DROP_RATE
    # token_t_irpe_method = args.MODEL_T2T.TOKEN_T_IRPE_METHOD
    # token_t_irpe_mode = args.MODEL_T2T.TOKEN_T_IRPE_MODE
    # token_t_irpe_shared_head = args.MODEL_T2T.TOKEN_T_IRPE_SHARE_HEAD
    # token_t_irpe_rpe_on = args.MODEL_T2T.TOKEN_T_IRPE_RPE_ON
    aspp_flag = args.MODEL_T2T.ASPP_FLAG
    token_t = {
        "rpe": args.MODEL_T2T.TOKEN_T_IRPE,
        "drop": args.MODEL_T2T.TOKEN_T_DROP_RATE,
        "attn_drop": args.MODEL_T2T.TOKEN_T_ATTN_DROP_RATE,
        "method": args.MODEL_T2T.TOKEN_T_IRPE_METHOD,
        "mode": args.MODEL_T2T.TOKEN_T_IRPE_MODE,
        "shared_head": args.MODEL_T2T.TOKEN_T_IRPE_SHARE_HEAD,
        "rpe_on": args.MODEL_T2T.TOKEN_T_IRPE_RPE_ON,
    }

    channel_reduce_rate = args.MODEL_T2T.CHANNEL_REDUCE_RATE

    model = SETMIL(img_size=img_size, tokens_type='transformer', num_classes=num_classes,
                   in_chans=in_chans, token_dim=token_dim, embed_dim=embed_dim, depth=depth,
                   num_heads=num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                   drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=0., norm_layer=nn.LayerNorm,
                   irpe=irpe, token_t=token_t, channel_reduce_rate=channel_reduce_rate, aspp_flag=aspp_flag,
                   )
    model.default_cfg = default_cfgs['T2t_vit_14']

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    return model, criterion
