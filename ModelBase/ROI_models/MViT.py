"""
MVIT offcial code for model structure

sample from https://github.com/facebookresearch/SlowFast/slowfast/models/attention.py
"""

import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
import numpy as np


class FFN(nn.Module):  # Mlp from timm
    """
    FFN (from timm)

    :param in_features:
    :param hidden_features:
    :param out_features:
    :param act_layer:
    :param drop:
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


class PatchEmbed(nn.Module):  # PatchEmbed from timm
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        # x: (B, 14*14, 768)
        return x


# MViT modules
# from https://github.com/facebookresearch/SlowFast/slowfast/models/attention.py
def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    """
    attention pooling constructor

    input:
    tensor of (B, Head, N, C) or (B, N, C)
    thw_shape: T, H, W  对应CNN的特征图形状（2D形状）T is video frams

    numpy.prob(T, H, W) == N(Num_patches) - 1 (cls token if it is there)

    output:
    tensor of (B, Head, N_O, C) or (B, N_O, C)
    thw_shape: T_O, H_O, W_O

    :param tensor: input feature patches
    :param pool: pooling/conv layer
    :param thw_shape: reconstruction feature map shape
    :param has_cls_embed: if cls token is used
    :param norm:  norm layer

    """
    if pool is None:  # no pool
        return tensor, thw_shape

    tensor_dim = tensor.ndim

    # fix dim: [B, Head, N, C]
    # N is Num_patches in Transformer modeling

    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:  # [B, N, C] -> [B, Head(1), N, C]
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, Head, N, C = tensor.shape
    T, H, W = thw_shape  # numpy.prob(T, H, W) == N(Num_patches) - 1 (cls token if it is there)

    # [B, Head, N, C] -> [B * Head, T, H, W, C] -> [B * Head, C, T, H, W]
    tensor = (tensor.reshape(B * Head, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous())
    # use tensor.contiguous() to matain its memory location

    # [B * Head, C, T, H, W] -> [B * Head, C, T_O, H_O, W_O]
    tensor = pool(tensor)  # 3D Pooling/ 3D Conv

    # output T, H, W
    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    # output Num_patches: numpy.prob(T, H, W)
    N_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]

    # [B * Head, C, T_O, H_O, W_O] -> [B, Head, C, N_O(T_O*H_O*W_O)] -> [B, Head, N_O, C]
    tensor = tensor.reshape(B, Head, C, N_pooled).transpose(2, 3)

    if has_cls_embed:
        # [B, Head, N_O, C] -> [B, Head, N_O+1(cls token), C]
        tensor = torch.cat((cls_tok, tensor), dim=2)

    # norm
    if norm is not None:
        tensor = norm(tensor)

    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:  # [B, Head, N_O, C] multi-head
        pass
    else:  # tensor_dim == 3: this is a single Head
        tensor = tensor.squeeze(1)  # [B, N_O, C]

    return tensor, thw_shape


'''
# case 1 single-head no pooling scale
x = torch.randn(1, 197, 768)
thw_shape = [1, 14, 14]
pool = nn.MaxPool3d((1, 1, 1), (1, 1, 1), (0, 0, 0), ceil_mode=False)
y, thw = attention_pool(x, pool, thw_shape)

print(y.shape)  # torch.Size([1, 197, 768])
print(thw)  # [1, 14, 14]


# case 2  multi-head no pooling scale
x = torch.randn(1, 8, 197, 96)  # [B, Head, N_O, C] multi-head
thw_shape = [1, 14, 14]
pool = nn.MaxPool3d((1, 1, 1), (1, 1, 1), (0, 0, 0), ceil_mode=False)
y, thw = attention_pool(x, pool, thw_shape)

print(y.shape)  # torch.Size([1, 8, 197, 96])
print(thw)  # [1, 14, 14]


# case 3 pooling scale
x = torch.randn(1, 197, 768)
thw_shape = [1, 14, 14]
pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2), (0, 0, 0), ceil_mode=False)
y, thw = attention_pool(x, pool, thw_shape)

print(y.shape)  # torch.Size([1, 50, 768])
print(thw)  # [1, 7, 7]


# case 4 multi-head pooling scale
x = torch.randn(1, 8, 197, 96)  # [B, Head, N_O, C] multi-head
thw_shape = [1, 14, 14]
pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2), (0, 0, 0), ceil_mode=False)
y, thw = attention_pool(x, pool, thw_shape)

print(y.shape)  # torch.Size([1, 8, 50, 96])
print(thw)  # [1, 7, 7]
'''


class MultiScaleAttention(nn.Module):  # Attention module
    """
    Attention module constructor

        input:
        tensor of (B, N, C)
        thw_shape: T, H, W  对应CNN的特征图形状（2D形状）T is video frams

        numpy.prob(T, H, W) == N(Num_patches) - 1 (cls token if it is there)

        output:
        tensor of (B, N_O, C)
        thw_shape: T_O, H_O, W_O

        :param dim: Transformer feature dim
        :param num_heads: Transformer heads
        :param qkv_bias: projecting bias
        :param drop_rate: dropout rate after attention calculation and mlp

        :param kernel_q: pooling kernal size for q
        :param kernel_kv: pooling kernal size for k and v
        :param stride_q: pooling kernal stride for q
        :param stride_kv: pooling kernal stride for k and v

        :param norm_layer:  norm layer
        :param has_cls_embed: if cls token is used
        :param mode: mode for attention pooling(downsampling) Options include `conv`, `avg`, and `max`.
        :param pool_first: process pooling(downsampling) before liner projecting

    """

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            drop_rate=0.0,
            kernel_q=(1, 1, 1),
            kernel_kv=(1, 1, 1),
            stride_q=(1, 1, 1),
            stride_kv=(1, 1, 1),
            norm_layer=nn.LayerNorm,
            has_cls_embed=True,
            # Options include `conv`, `avg`, and `max`.
            mode="conv",
            # If True, perform pool before projection.
            pool_first=False,
    ):
        super().__init__()

        self.pool_first = pool_first
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # squre root
        self.has_cls_embed = has_cls_embed

        padding_q = [int(q // 2) for q in kernel_q]  # 以半个kernal size进行padding，向下取整
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        # projecting mlp
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if np.prod(kernel_q) == 1 and np.prod(stride_q) == 1:
            kernel_q = ()  # clear
        if np.prod(kernel_kv) == 1 and np.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode in ("avg", "max"):  # use nn.MaxPool3d or nn.AvgPool3d
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None  # Skip pooling if kernel is cleared
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )

        elif mode == "conv":  # use nn.Conv3d with depth wise conv and fixed channel tasks_to_run
            self.pool_q = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(head_dim) if len(kernel_q) > 0 else None

            self.pool_k = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(head_dim) if len(kernel_kv) > 0 else None

            self.pool_v = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(head_dim) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

    def forward(self, x, thw_shape):
        """
        x: Transformer feature patches
        thw_shape: reconstruction feature map shape
        """

        B, N, C = x.shape

        # step 1: duplicate projecting + head split: [B, N, C] -> [B, H, N, C/H]

        if self.pool_first:  # step a.1 embedding
            # head split [B, N, C] -> [B, N, H, C/H] -> [B, H, N, C/H]
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            q = k = v = x

        else:  # step b.1 projecting first
            # mlp transform + head split: [B, N, C] -> [B, N, H, C/H] -> [B, H, N, C/H]
            # todo 这里我觉得可能共享mlp映射更好，能有更好的交互，但是分离mlp更节约计算量
            q = k = v = x
            q = (
                self.q(q)
                    .reshape(B, N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )
            k = (
                self.k(k)
                    .reshape(B, N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )
            v = (
                self.v(v)
                    .reshape(B, N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )

        # step 2: calculate attention_pool feature sequence and its shape
        # [B, H, N0, C/H] -> [B, H, N1, C/H]
        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        if self.pool_first:  # step a.3 MLP projecting
            # calculate slide_feature number, q_N, k_N, v_N
            q_N = (
                np.prod(q_shape) + 1
                if self.has_cls_embed
                else np.prod(q_shape)
            )
            k_N = (
                np.prod(k_shape) + 1
                if self.has_cls_embed
                else np.prod(k_shape)
            )
            v_N = (
                np.prod(v_shape) + 1
                if self.has_cls_embed
                else np.prod(v_shape)
            )

            # [B, H, N1, C/H] -> [B, N1, H, C/H] -> [B, N1, C] -> MLP
            # -> [B, N1, C] -> [B, N1, H, C/H] -> [B, H, N1, C/H]
            q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
            q = (
                self.q(q)
                    .reshape(B, q_N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
            v = (
                self.v(v)
                    .reshape(B, v_N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
            k = (
                self.k(k)
                    .reshape(B, k_N, self.num_heads, C // self.num_heads)
                    .permute(0, 2, 1, 3)
            )

        # step 3: attention calculation
        # multi-head self attention [B, H, N1, C/H] -> [B, H, N1, C/H]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # head squeeze [B, H, N1, C/H] -> [B, N1, H, C/H] -> [B, N1, C]
        N = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # step 4: mlp stablization and dropout [B, N1, C] -> [B, N1, C]
        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x, q_shape


'''
# case 1
model = MultiScaleAttention(768)
x = torch.randn(1, 197, 768)
y, thw = model(x, [1, 14, 14])
print(y.shape)


# case 2
kernel_q = (1, 2, 2)
kernel_kv = (1, 2, 2)
stride_q = (1, 2, 2)
stride_kv = (1, 2, 2)
# MultiScaleAttention 中设计以半个kernal size进行padding，向下取整

model = MultiScaleAttention(768, kernel_q=kernel_q, kernel_kv=kernel_kv, stride_q=stride_q, stride_kv=stride_kv)
x = torch.randn(1, 197, 768)
y, thw = model(x, [1, 14, 14])

print(y.shape)  # 输出torch.Size([1, 65, 768])：不padding是7*7 由于padding变成8*8， 之后加上cls token
'''


class MultiScaleBlock(nn.Module):  # MViT Encoder
    """
    Attention module constructor

        input:
        tensor of (B, N, C)
        thw_shape: T, H, W  对应CNN的特征图形状（2D形状）T is video frams

        numpy.prob(T, H, W) == N(Num_patches) - 1 (cls token if it is there)

        output:
        tensor of (B, N_O, C)
        thw_shape: T_O, H_O, W_O

        :param dim: Transformer feature dim
        :param dim_out:

        :param num_heads: Transformer heads
        :param mlp_ratio: FFN hidden expansion
        :param qkv_bias: projecting bias
        :param drop_rate: dropout rate after attention calculation and mlp
        :param drop_path: dropout rate for SD
        :param act_layer: FFN act
        :param norm_layer: Pre Norm

        :param up_rate:
        :param kernel_q: pooling kernal size for q
        :param kernel_kv: pooling kernal size for k and v
        :param stride_q: pooling kernal stride for q
        :param stride_kv: pooling kernal stride for k and v

        :param has_cls_embed: if cls token is used
        :param mode: mode for attention pooling(downsampling) Options include `conv`, `avg`, and `max`.
        :param pool_first: process pooling(downsampling) before liner projecting

    """

    def __init__(
            self,
            dim,
            dim_out,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=False,
            drop_rate=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            up_rate=None,
            kernel_q=(1, 1, 1),
            kernel_kv=(1, 1, 1),
            stride_q=(1, 1, 1),
            stride_kv=(1, 1, 1),
            has_cls_embed=True,
            mode="conv",
            pool_first=False,
    ):
        super().__init__()

        self.has_cls_embed = has_cls_embed

        # step 1: Attention projecting
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)  # pre-norm

        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=self.has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            )

        self.drop_path = (DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

        # residual connection for Attention projecting
        kernel_skip = kernel_q  # fixme ori: [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]  # 以半个kernal size进行padding，向下取整

        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(kernel_skip) > 0
            else None)

        self.norm2 = norm_layer(dim)  # pre-norm

        # step 2: FFN projecting
        mlp_hidden_dim = int(dim * mlp_ratio)

        # here use FFN to encode feature into abstractive information in the dimension
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out

        self.mlp = FFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop=drop_rate,
        )

        # residual connection for FFN projecting
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x, thw_shape):
        # step 1: Attention projecting
        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        # residual connection for Attention projecting
        x_res, _ = attention_pool(x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed)
        x = x_res + self.drop_path(x_block)

        # step 2: FFN projecting
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        # residual connection for FFN projecting
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)

        return x, thw_shape_new


'''
# case 1
model = MultiScaleBlock(768,1024)
x = torch.randn(1, 197, 768)
y, thw = model(x, [1, 14, 14])
print(y.shape)  # torch.Size([1, 197, 1024])


# case 2
kernel_q = (1, 2, 2)
kernel_kv = (1, 2, 2)
stride_q = (1, 2, 2)
stride_kv = (1, 2, 2)
# MultiScaleAttention 中设计以半个kernal size进行padding，向下取整

model = MultiScaleBlock(768, 1024, kernel_q=kernel_q, kernel_kv=kernel_kv, stride_q=stride_q, stride_kv=stride_kv)
x = torch.randn(1, 197, 768)
y, thw = model(x, [1, 14, 14])

print(y.shape)  # 输出torch.Size([1, 65, 1024])：不padding是7*7 由于padding变成8*8， 之后加上cls token
'''


# TODO  框架的复现未完成
class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(img_size=spatial_size, patch_size=16, in_chans=in_chans, embed_dim=embed_dim)
        '''
            stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=slide_embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,)
        '''
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, embed_dim)
                )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim)
            )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        # TODO getthem
        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                                                           i
                                                       ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        self.blocks = nn.ModuleList()

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
            )
            self.blocks.append(attention_block)

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim)

        self.head = self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = x[0]
        x = self.patch_embed(x)

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x)
        if self.cls_embed_on:
            x = x[:, 0]
        else:
            x = x.mean(1)

        x = self.head(x)
        return x
