"""
Slide MAE Model    Script  ver： Oct 26th 13:30

# References:
Based on MAE code.
https://github.com/facebookresearch/mae

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

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from ModelBase.WSI_models.WSI_Transformer.WSI_Transformer_blocks import Slide_PatchEmbed
from ModelBase.WSI_models.WSI_Transformer.WSI_pos_embed import get_2d_sincos_pos_embed


class SlideMaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, backbone: nn.Module, ROI_feature_dim=768, embed_dim=1024,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 basic_state_dict=None, decoder=None, decoder_rep_dim=None,
                 slide_pos=None, slide_ngrids=None,
                 **kwargs):

        super().__init__()
        # Encoder model to be trained
        self.backbone = backbone  # output of backbone model is tile tokens (remove the other tokens in the end)
        self.slide_pos = slide_pos or self.backbone.slide_pos
        self.slide_ngrids = slide_ngrids or self.backbone.slide_ngrids

        # MAE decoder specifics
        # --------------------------------------------------------------------------
        # notice the current version removed the CLS or MTL task tokens in the decoder
        # if the feature dimension of encoder and decoder are different, use decoder_embed to align them
        if embed_dim != decoder_embed_dim:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        else:
            self.decoder_embed = nn.Identity()

        if decoder is not None:
            assert decoder_rep_dim is not None
            self.decoder = decoder  # build outside the SlideMAE module
            # set mask_token (learnable mask token for reconstruction)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # Decoder use an FC to reconstruct image, unlike the Encoder which use a CNN to split slide_feature
            self.decoder_pred = nn.Linear(decoder_rep_dim, ROI_feature_dim, bias=True)  # decoder to tile_feature

        else:
            self.decoder = None  # build inside the SlideMAE module
            # set mask_token (learnable mask token for reconstruction)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            if self.slide_pos:
                # set and freeze decoder_pos_embed,  use the fixed sin-cos embedding for tokens + mask_token
                self.decoder_patch_embed = Slide_PatchEmbed(embed_dim, decoder_embed_dim)
                num_patches = self.slide_ngrids ** 2
                # fixed sin-cos embedding
                self.register_buffer('decoder_pos_embed', torch.zeros(1, num_patches, decoder_embed_dim),
                                     persistent=False)

            self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                                                       qkv_bias=True, norm_layer=norm_layer)
                                                 for i in range(decoder_depth)])
            # qk_scale=None fixme related to timm version
            self.decoder_norm = norm_layer(decoder_embed_dim)

            # Decoder use a FC to reconstruct image, unlike the Encoder which use a CNN to split slide_feature
            self.decoder_pred = nn.Linear(decoder_embed_dim, ROI_feature_dim, bias=True)  # decoder to tile_feature

        # --------------------------------------------------------------------------
        # whether or not to use norm_pix_loss
        self.norm_pix_loss = norm_pix_loss
        # parameter initialization
        self.initialize_weights()

        # load basic state_dict of backbone for Transfer-learning-based tuning
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

    def initialize_weights(self):
        # Encoder (backbone) initialization is done in the import backbone
        # Decoder init is done here:
        if self.decoder is None:  # build inside the SlideMAE module
            # initialize a 2d positional encoding of (slide_embed_dim, grid) by sin-cos embedding
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.slide_ngrids,
                                                        cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.decoder_patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            # initialize nn.Linear and nn.LayerNorm
            self.decoder_pred.apply(self._init_weights)
            self.decoder_norm.apply(self._init_weights)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        # initialize nn.Linear and nn.LayerNorm
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify_decoder(self, imgs, patch_size=None):  # TODO how should we pretrain decoder model if its seg model?
        """
        Break image to slide_feature tokens

        fixme, notice we take patch_size = self.decoder.patch_size[0]

        input:
        imgs: (B, CLS, H, W)

        output:
        x: (B, num_patches, -1) AKA [B, num_patches, -1]
        """
        # patch_size
        patch_size = self.decoder.patch_size[0] if patch_size is None else patch_size

        # assert H == W and image shape is divided-able by slide_feature
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0
        # slide_feature num in rol or column
        h = w = imgs.shape[2] // patch_size

        # use reshape to split slide_feature [B, C, H, W] -> [B, C, h_p, patch_size, w_p, patch_size]
        x = imgs.reshape(shape=(imgs.shape[0], -1, h, patch_size, w, patch_size))

        # ReArrange dimensions [B, C, h_p, patch_size, w_p, patch_size] -> [B, h_p, w_p, patch_size, patch_size, C]
        x = torch.einsum('nchpwq->nhwpqc', x)
        # ReArrange dimensions [B, h_p, w_p, patch_size, patch_size, C] -> [B, num_patches, flatten_dim]
        x = x.reshape(shape=(imgs.shape[0], h * w, -1))
        return x

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

    def random_masking(self, x, coords=None, mask_ratio=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.

        注意torch.argsort返回的是：
        在每个指定dim，按原tensor每个位置数值大小升序排列后，的原本位置的idx组成的矩阵

        input:
        x: [B, num_patches, D], sequence of Tokens

        output: x_remained, mask, ids_restore
        x_remained: [B, num_patches * (1-mask_ratio), D], sequence of Tokens
        mask: [B, num_patches], binary mask
        ids_restore: [B, num_patches], idx of restoring all position
        """
        assert mask_ratio is not None
        B, num_patches, D = x.shape  # batch, length, dim
        # 计算需要保留的位置的个数
        len_keep = int(num_patches * (1 - mask_ratio))
        # 做一个随机序列[B,num_patches]，用于做位置标号
        noise = torch.rand(B, num_patches, device=x.device)  # noise in [0, 1]

        # 在Batch里面每个序列上获得noise tensor经过升序排列后原本位置的idx矩阵  在batch内进行升序排列
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 再对idx矩阵继续升序排列可获得：原始noise tensor的每个位置的排序顺位
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # 设置需要的patch的索引
        # ids_keep.unsqueeze(-1).repeat(1, 1, D):
        # [B,num_patches] -> [B,keep_patches] -> [B,keep_patches,1] 每个位置数字为idx of ori slide_feature -> [B,keep_patches,D]

        # torch.gather 按照索引取值构建新tensor: x_remained [B,keep_patches,D] 表示被标记需要保留的位置, 原文是x_masked
        x_remained = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        if coords is not None:
            coords_remained = torch.gather(coords, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 2))
        else:
            coords_remained = None
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, num_patches], device=x.device)
        mask[:, :len_keep] = 0  # 设置mask矩阵，前len_keep个为0，后面为1

        # 按照noise tensor每个位置的大小顺序，来设置mask符号为0的位置，获得mask矩阵
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_remained, coords_remained, mask, ids_restore  # x_remained原文是x_masked

    def forward_encoder(self, image_features, coords=None, MTL_tokens=None, mask_ratio=None):
        """
        :param image_features: Tensor of shape [B, N, feature_dim],
                               where B is batch size, N is the number of patches/features per slide_feature.

        :param mask_ratio: mask_ratio

        :return: Encoder output: encoded tokens, mask position, restore idxs
        x: [B, self.MTL_token_num + num_patches * (1-mask_ratio), D], sequence of Tokens (including the MTL token)
        mask: [B, num_patches], binary mask
        ids_restore: [B, num_patches], idx of restoring all position
        """
        # masking: length -> length * (1-mask_ratio)
        # x_remained: [B, num_patches * (1-mask_ratio), D], sequence of Tokens
        x_remained, coords_remained, mask, ids_restore = self.random_masking(image_features, coords, mask_ratio)

        # apply Transformer Encoders
        if hasattr(self.backbone, "MTL_token_num") and self.backbone.MTL_token_num > 0:
            latent = self.backbone(x_remained, coords=coords_remained, MTL_tokens=MTL_tokens)
            # [B, MTL_token_num + num_patches, embed_dim] -> [B, num_patches, embed_dim]
            latent = latent[:, self.backbone.MTL_token_num:]  # (remove the other tokens in the end)
        else:
            # [B, num_patches, embed_dim]
            latent = self.backbone(x_remained, coords=coords_remained)

        # Encoder output: encoded latent tokens, coords_remained, mask position, restore idxs
        return latent, coords_remained, mask, ids_restore

    def forward_decoder(self, x, coords=None, ids_restore=None):
        """
        :param x: [B, 1 + num_patches * (1-mask_ratio), D], sequence of Tokens (including the cls token)
        :param ids_restore: [B, num_patches] all restore idxs for torch.gather(mask, dim=1, index=ids_restore)

        :return:
        Decoder output: reconstracted tokens
        x: [B, num_patches , D], sequence of Tokens
        """
        assert ids_restore is not None

        if self.decoder is None:
            # embed tokens: [B, num_encoded_tokens, slide_embed_dim] -> [B, num_encoded_tokens, D_Decoder]
            x = self.decoder_embed(x)

            # append mask tokens to sequence as placeholder: [B, num_patches - num_encoded_tokens, D_Decoder]
            # number of mask token need is the requirement to fulfill all the num_patches
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            # mask_tokens: [B, num_mask_patches(num_patches - num_visible_patches), 1]

            # -> [B, num_patches(num_visible_patches + num_mask_patches), D_Decoder]
            x_ = torch.cat([x, mask_tokens], dim=1)

            # unshuffle to restore the position of tokens
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            # torch.gather restore the correct location: x_rec [B,num_patches,D_Decoder] but here the value is empty

            # get pos indices for decoder
            pos = self.coords_to_pos(coords)
            x = x_ + self.decoder_pos_embed[:, pos, :].squeeze(0)

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # Reconstruction projection [B, num_patches, D_Decoder] -> [B, num_patches, p*p*3]
            x = self.decoder_pred(x)

        else:
            # append mask tokens to sequence as place holder: [B, num_patches - num_encoded_tokens, D]
            # number of mask token need is the requirement to fill the num_patches
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

            # -> [B, num_patches, D]
            x_ = torch.cat([x, mask_tokens], dim=1)

            # unshuffle to restore the position of tokens
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            # torch.gather restore the correct location: x_rec [B,num_patches,D_Decoder] but here the value is empty

            # embed tokens: [B, num_encoded_tokens, D_Encoder] -> [B, num_encoded_tokens, D_Decoder]
            x_ = self.decoder_embed(x_)

            # unpatchify to make image form [B, N, Enc] to [B,H,W,C]
            x = self.unpatchify(x_)  # restore image by Encoder

            # apply decoder module to segment the output of encoder
            x = self.decoder(x)  # [B, SEG_CLS, H, W]
            # the output of segmentation is transformed to  [B, N, Dec]
            x = self.patchify_decoder(x)  # TODO we need better design to handel SEG_CLS for certain decoders

            # Convert the number of channels to match image for loss function
            x = self.decoder_pred(x)  # [B, N, Dec] -> [B, N, p*p*3]

        return x

    def forward_loss(self, image_features, pred, mask):
        """
        MSE loss for all patches towards the ori image

        Input:
        image_features: [B, num_patches, ROI_feature_dim], Encoder input features
        pred: [B, num_patches, ROI_feature_dim], Decoder reconstructed image
        mask: [B, num_patches], 0 is keep, 1 is remove,

        """
        target = image_features

        if self.norm_pix_loss:  # standardization
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        # MSE loss (todo change to other loss like similarity)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per slide_feature

        # binary mask, 1 for removed patches
        loss = (loss * mask).sum() / mask.sum()  # here we only take mean loss on the reconstructed patches
        return loss

    def forward(self, image_features, coords=None, MTL_tokens=None, mask_ratio=0.75):
        # Encoder to obtain latent tokens
        if MTL_tokens is not None:
            latent, coords_remained, mask, ids_restore = (
                self.forward_encoder(image_features, coords=coords, MTL_tokens=MTL_tokens, mask_ratio=mask_ratio))
        else:
            latent, coords_remained, mask, ids_restore = (
                self.forward_encoder(image_features, coords=coords, mask_ratio=mask_ratio))

        # Decoder to obtain Reconstructed image patches (coords should be all)
        pred = self.forward_decoder(latent, coords=coords, ids_restore=ids_restore)  # [N, L, p*p*3]
        # MSE loss for all patches towards the ori image
        loss = self.forward_loss(image_features, pred, mask)
        # print(loss)  # todo should check loss scaling
        return loss, pred, mask


def SlideMAE_dec512d8b(backbone: nn.Module, ROI_feature_dim, slide_embed_dim, dec_idx=None, **kwargs):
    assert dec_idx is None or dec_idx == 'dec512d8b'
    print("Decoder:", dec_idx)

    model = SlideMaskedAutoencoder(backbone, decoder=None,
                                   ROI_feature_dim=ROI_feature_dim, embed_dim=slide_embed_dim,
                                   decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                   mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   **kwargs)
    return model


def SlideMAE_with_decoder(backbone: nn.Module, ROI_feature_dim, slide_embed_dim,
                          dec_idx=None, num_classes=3, img_size=224, **kwargs):
    # num_classes做的是one-hot seg但是不是做还原，我们得设计一下如何去做这个还原才能实现预训练

    if dec_idx == 'swin_unet':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        from Seg_structures.Swin_Unet_main.networks.vision_transformer import SwinUnet as ViT_seg
        decoder = ViT_seg(num_classes=num_classes, **kwargs)

    elif dec_idx == 'transunet':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        transunet_name = 'R50-ViT-B_16'
        transunet_patches_size = 16
        from Seg_structures.TransUNet_main.networks.vit_seg_modeling import CONFIGS as CONFIGS_Transunet_seg
        from Seg_structures.TransUNet_main.networks.vit_seg_modeling import VisionTransformer as Transunet_seg

        config_vit = CONFIGS_Transunet_seg[transunet_name]
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3

        if transunet_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / transunet_patches_size), int(img_size / transunet_patches_size))
        decoder = Transunet_seg(config_vit, num_classes=config_vit.n_classes)

    elif dec_idx == 'UTNetV2':
        decoder_embed_dim = 768
        decoder_rep_dim = 16 * 16 * 3

        from Seg_structures.UtnetV2.utnetv2 import UTNetV2 as UTNetV2_seg
        decoder = UTNetV2_seg(in_chan=3, num_classes=num_classes)

    else:
        print('no effective decoder!')
        return -1

    print('dec_idx: ', dec_idx)

    model = SlideMaskedAutoencoder(backbone, decoder=decoder,
                                   ROI_feature_dim=ROI_feature_dim, embed_dim=slide_embed_dim,
                                   decoder_embed_dim=decoder_embed_dim, decoder_rep_dim=decoder_rep_dim,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended structure
SlideMAE_dec512d8b = SlideMAE_dec512d8b  # decoder: 512 dim, 8 blocks
# Equipped with decoders
SlideMAE_with_decoder = SlideMAE_with_decoder  # decoder: 768 dim, with outside decoder
