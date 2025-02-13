"""
Transformer blocks   Script  ver： Oct 29th 00:30

based on：timm
https://www.freeaihub.com/post/94067.html

[ ] A GNN to do the bag modeling:  -> [all cell nodes, feature_dim]

[ ] Bag MIL : A MIL to do the bag MIL modeling:  -> [batch, bag_num, bag_size (different), backbone_dim]
                                                 -> [batch, node_num, backbone_dim]

[X] Transformer do the relationship modeling of the bags: (batch, node_num + MTL_tokens, D)

[ ] MLP(MTL_heads) to do the MTL projection using MTL_tokens: (batch, MTL_tokens, Task_Dim)

"""
import os
import sys
from pathlib import Path

# For convenience, import all path to sys
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir))
sys.path.append(str(this_file_dir.parent))
sys.path.append(str(this_file_dir.parent.parent))
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

from collections import OrderedDict
from functools import partial

import timm

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
from ModelBase.WSI_models.WSI_Transformer.WSI_pos_embed import get_2d_sincos_pos_embed


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


class Attention(nn.Module):  # qkv Transform + MSA(MHSA) (Attention from timm)
    """
    qkv Transform + MSA(MHSA) (from timm)

    # input  x.shape = batch, patch_number, patch_dim
    # output  x.shape = batch, patch_number, patch_dim

    :param dim: dim=CNN feature dim, because the slide_feature size is 1x1
    :param num_heads:
    :param qkv_bias:
    :param qk_scale: by default head_dim ** -0.5  (squre root)
    :param attn_drop: dropout rate after MHSA
    :param proj_drop:

    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # input x.shape = batch, patch_number, patch_dim
        batch, patch_number, patch_dim = x.shape

        # mlp transform + head split [N, P, D] -> [N, P, 3D] -> [N, P, 3, H, D/H] -> [3, N, H, P, D/H]
        qkv = self.qkv(x).reshape(batch, patch_number, 3, self.num_heads, patch_dim //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # 3 [N, H, P, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # [N, H, P, D/H] -> [N, H, P, D/H]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)  # Dropout

        # head fusion [N, H, P, D/H] -> [N, P, H, D/H] -> [N, P, D]
        x = (attn @ v).transpose(1, 2).reshape(batch, patch_number, patch_dim)

        x = self.proj(x)
        x = self.proj_drop(x)  # mlp

        # output x.shape = batch, patch_number, patch_dim
        return x


class Encoder_Block(nn.Module):  # teansformer Block from timm

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        # input x.shape = batch, patch_number, patch_dim
        # output x.shape = batch, patch_number, patch_dim

        :param dim: dim
        :param num_heads:
        :param mlp_ratio: FFN
        :param qkv_bias:
        :param qk_scale: by default head_dim ** -0.5  (squre root)
        :param drop:
        :param attn_drop: dropout rate after Attention
        :param drop_path: dropout rate after sd
        :param act_layer: FFN act
        :param norm_layer: Pre Norm
        """
        super().__init__()
        # Pre Norm
        self.norm1 = norm_layer(dim)  # Transformer used the nn.LayerNorm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE from timm: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # stochastic depth

        # Add & Norm
        self.norm2 = norm_layer(dim)

        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


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


# Slide level modeling Transformer model
class Slide_Transformer_blocks(nn.Module):
    def __init__(self, ROI_feature_dim=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=None, act_layer=None, slide_pos=True,
                 MTL_token_num=0, slide_ngrids=1000, **kwargs):
        super().__init__()
        # Set Slide positional embedding
        self.slide_pos = slide_pos
        # Set MTL tokens (by right should be pass from the MTL wrapper framework
        self.MTL_token_num = MTL_token_num
        # put backbone MTL tokens if not given by wrapper framework
        if self.MTL_token_num > 0:
            self.MTL_tokens = nn.Parameter(torch.zeros(1, self.MTL_token_num, ROI_feature_dim))

        self.patch_embed = Slide_PatchEmbed(in_chans=ROI_feature_dim, embed_dim=embed_dim)

        if self.slide_pos:
            self.slide_ngrids = slide_ngrids
            num_patches = slide_ngrids ** 2
            self.register_buffer('pos_embed', torch.zeros(1, self.MTL_token_num + num_patches, embed_dim),
                                 persistent=False)  # fixed sin-cos embedding, top T positions are for MTL tokens

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Encoder_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                          attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # notice we have put multiple task tokens in the WSI model framework, here we just do encoders
        self.initialize_vit_weights()

    def coords_to_pos(self, coords):
        """
        This function is used to convert the coordinates to the positional indices

        Arguments:
            B is the batch size
            N is the number of tiles
        ----------
        coords: torch.Tensor
            The coordinates of the patches, of shape [B, N, 2]
        output: torch.Tensor
            The positional indices of the patches, of shape [B, N]
        """
        coords_ = torch.floor(coords / 256.0)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        return pos.long() + self.MTL_token_num  # add T value to each location (shift the positions by T)

    def initialize_vit_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.slide_ngrids, add_token=self.MTL_token_num)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if self.MTL_token_num > 0:
            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            torch.nn.init.normal_(self.MTL_tokens, std=0.02)

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

    def forward(self, x, coords=None, MTL_tokens=None):
        """
        :param x: [batch_size, tile_num, default_ROI_feature_dim]
        :param coords: [batch_size, tile_num, 2]
        :param MTL_tokens: None or (1, self.MTL_token_num, default_ROI_feature_dim) pass from MTL framework
        """
        if self.MTL_token_num > 0 and MTL_tokens is None:
            # put backbone MTL tokens if not given by wrapper framework
            MTL_tokens = self.MTL_tokens  # [1,T,D]

        # Feature embedding
        x = self.patch_embed(x)

        # Slide Pos Embedding
        if self.slide_pos:
            assert coords != None

            # get pos indices
            pos = self.coords_to_pos(coords)  # [B, N]  tile loc index that has been shift by T
            # [B,N,D] + ([1,B,N,D] -> [B,N,D])
            x = x + self.pos_embed[:, pos, :].squeeze(0)

            if self.MTL_token_num > 0:
                assert MTL_tokens != None
                assert MTL_tokens.shape[1] == self.MTL_token_num
                # top T positions are for MTL tokens
                MTL_tokens = MTL_tokens + self.pos_embed[:, :self.MTL_token_num, :]

        # Expand and put MTL tokens into data flow if its MTL model
        if self.MTL_token_num > 0:
            assert MTL_tokens.shape[1] == self.MTL_token_num
            # (batch, self.MTL_token_num, slide_embed_dim)
            MTL_tokens = MTL_tokens.expand(x.shape[0], -1, -1)
            # [batch, MTL_token_num + tile_num, default_ROI_feature_dim]
            x = torch.cat((MTL_tokens, x), dim=1)  # concatenate as the front tokens

        x = self.blocks(x)
        x = self.norm(x)
        # [batch, MTL_token_num + tile_num, default_ROI_feature_dim]
        return x


# Slide level modeling Transformer model with prompt tokens
class Prompt_Slide_Transformer_blocks(Slide_Transformer_blocks):
    def __init__(self, ROI_feature_dim=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, act_layer=None,
                 Prompt_Token_num=20, VPT_type="Shallow", basic_state_dict=None, MTL_token_num=0, **kwargs):

        # Recreate ViT
        super().__init__(ROI_feature_dim=ROI_feature_dim, MTL_token_num=MTL_token_num,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, norm_layer=norm_layer, act_layer=act_layer, **kwargs)

        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.VPT_type = VPT_type
        if self.VPT_type == "Deep":
            self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
        else:  # "Shallow"
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.Prompt_Tokens.requires_grad = True
        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass

    def UnFreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def obtain_prompt(self):
        prompt_state_dict = {'Prompt_Tokens': self.Prompt_Tokens}
        # print(prompt_state_dict)
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):

        if self.Prompt_Tokens.shape == prompt_state_dict['Prompt_Tokens'].shape:

            # device check
            Prompt_Tokens = nn.Parameter(prompt_state_dict['Prompt_Tokens'].cpu())
            Prompt_Tokens.to(torch.device(self.Prompt_Tokens.device))

            self.Prompt_Tokens = Prompt_Tokens

        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.Prompt_Tokens.shape)
            print('shape of model given prompt', prompt_state_dict['Prompt_Tokens'].shape)
            print('')

    def forward(self, x, coords=None, MTL_tokens=None):
        """
        :param x: [batch_size, tile_num, default_ROI_feature_dim]
        :param coords:
        :param MTL_tokens: None or (1, self.MTL_token_num, default_ROI_feature_dim) pass from MTL framework
        """
        if self.MTL_token_num > 0 and MTL_tokens == None:
            # put backbone MTL tokens if not given by wrapper framework
            MTL_tokens = self.MTL_tokens  # [1,T,D]

        # embed patches
        x = self.patch_embed(x)

        # Slide Pos Embedding
        if self.slide_pos:
            assert coords != None

            # get pos indices
            pos = self.coords_to_pos(coords)  # [B,N]
            # [B,N,D] + ([1,B,N,D] -> [B,N,D])
            x = x + self.pos_embed[:, pos, :].squeeze(0)
            if self.MTL_token_num > 0:
                assert MTL_tokens != None  # top T positions are for MTL tokens
                MTL_tokens = MTL_tokens + self.pos_embed[:, :self.MTL_token_num, :]

        # Expand and put MTL tokens into data flow if its MTL model
        if self.MTL_token_num > 0:
            assert MTL_tokens.shape[1] == self.MTL_token_num
            # (batch, self.MTL_token_num, default_ROI_feature_dim)
            MTL_tokens = MTL_tokens.expand(x.shape[0], -1, -1)
            # [batch, MTL_token_num + tile_num, default_ROI_feature_dim]
            x = torch.cat((MTL_tokens, x), dim=1)  # concatenate as the front tokens

        # x: [B, N (task tokens + tile tokens), D]

        if self.VPT_type == "Deep":
            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            for i in range(len(self.blocks)):
                # concatenate Prompt_Tokens
                Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                # firstly concatenate
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1]
                # lastly remove the prompt tokens, an interesting trick for VPT Deep
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        else:  # self.VPT_type == "Shallow"
            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            # concatenate Prompt_Tokens
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            num_tokens = x.shape[1]
            # Sequentially pass the encoders
            x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        # [batch, MTL_token_num + tile_num, default_ROI_feature_dim], MTL_token_num can be 0
        return x


if __name__ == '__main__':
    # cuda issue
    print('cuda availability:', torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MTL_token_num = 20
    default_ROI_feature_dim = 768
    slide_embed_dim = 768

    model = Prompt_Slide_Transformer_blocks(Prompt_Token_num=20, VPT_type="Deep",
                                            ROI_feature_dim=default_ROI_feature_dim,
                                            embed_dim=slide_embed_dim,
                                            MTL_token_num=MTL_token_num)
    '''
    model = Slide_Transformer_blocks(default_ROI_feature_dim=default_ROI_feature_dim,
                                     MTL_token_num=MTL_token_num)
    '''
    MTL_tokens = nn.Parameter(torch.zeros(1, MTL_token_num, slide_embed_dim)) if MTL_token_num > 0 else None
    MTL_tokens = MTL_tokens.to(dev) if MTL_token_num > 0 else None

    # Transferlearning on Encoders
    ViT_backbone_weights = timm.create_model('vit_base_patch16_224', pretrained=True).state_dict()
    del ViT_backbone_weights['patch_embed.proj.weight']
    del ViT_backbone_weights['patch_embed.proj.bias']
    model.load_state_dict(ViT_backbone_weights, strict=False)

    model.to(dev)

    # play data (to see that Transformer can do multiple length, therefore can do MTL easily)
    coords = torch.randn(2, 123, 2).to(dev)
    x = torch.randn(2, 123, default_ROI_feature_dim).to(dev)
    y = model(x, coords=coords, MTL_tokens=MTL_tokens)
    print(y.shape)  # [B,N_tsk+N_fea, D]
