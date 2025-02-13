import torch.nn as nn
from timm.models.layers import DropPath
from .transformer_block import Mlp

from .irpe import build_rpe
from .irpe import get_rpe_config



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                rpe=False, method="product", mode='ctx', shared_head=True, rpe_on='k',):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # iRPE
        if rpe:
            self.rpe_config = get_rpe_config(
                ratio=1.9,
                method=method,
                mode=mode,
                shared_head=shared_head,
                skip=0, # w/o cls tokens
                rpe_on=rpe_on,
            )
            self.rpe_q, self.rpe_k, self.rpe_v = \
                build_rpe(self.rpe_config,
                          head_dim=in_dim,
                          num_heads=num_heads)
        else:
            self.rpe_q, self.rpe_k, self.rpe_v = None, None, None

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q *= self.scale
        attn = (q @ k.transpose(-2, -1))

        # iRPE
        # image relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q)
        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # iRPE
        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x

class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 rpe=False, method="product", mode='ctx', shared_head=True, rpe_on='k'
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            rpe=rpe, method=method, mode=mode, shared_head=shared_head, rpe_on=rpe_on,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
