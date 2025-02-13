# --------------------------------------------------------
# MUSK: A Vision-Language Foundation Model for Precision Oncology
# Published in Nature, 2025
# GitHub Repository: https://github.com/lilab-stanford/MUSK
# Copyright (c) 2025 Stanford University, by Jinxi Xiang
# Licensed under the CC-BY-NC-ND 4.0 License (https://creativecommons.org/licenses/by-nc-nd/4.0/)
# Please see LICENSE for additional details.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np
from typing import Optional, List, Tuple
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from .torchscale.model.BEiT3 import BEiT3
from .torchscale.architecture.config import EncoderConfig
import math
from .utils import MultiScaleForward


class TwoLayerMLP(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features,
            out_features,
            norm_layer,
            norm_input=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ModelWrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token',
                'beit3.encoder.embed_positions.A.weight',
                'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MUSK(ModelWrapper):
    def __init__(self, args, **kwargs):
        super().__init__(args=args)
        embed_dim = args.encoder_embed_dim

        # Define heads for vision and language
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)

        # Logit scale parameter initialization
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text_description: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
            return_global: bool = True,
            with_head: bool = True,
            out_norm: bool = True,
            ms_aug: bool = False,
            scales: Optional[List[int]] = None,
            max_split_size: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for vision-language model.
        Args:
            image: Input image tensor.
            text_description: Input text tokens.
            padding_mask: Padding mask for text.
            return_global: Whether to return global CLS token.
            with_head: Whether to apply linear heads.
            out_norm: Whether to normalize output embeddings.
            ms_aug: Enable multiscale feature augmentation.
            scales: List of scales for multiscale feature augmentation.
            max_split_size: Maximum split size for multiscale forward.

        Returns:
            vision_cls: Vision embeddings (normalized if out_norm).
            language_cls: Language embeddings (normalized if out_norm).
        """
        if scales is None:
            scales = [1, 2]  # Default scales

        # Process image input
        vision_cls = None
        if image is not None:
            if ms_aug:
                vision_cls = MultiScaleForward(
                    model=self,
                    input=image,
                    scales=scales,
                    max_split_size=max_split_size
                )
                if with_head:
                    vision_cls = self.vision_head(vision_cls[:, :1024])
            else:
                outputs = self.beit3(visual_tokens=image)
                x = outputs["encoder_out"]
                vision_cls = x[:, 0, :] if return_global else x
                if with_head:
                    vision_cls = self.vision_head(vision_cls)
            if out_norm:
                vision_cls = F.normalize(vision_cls, dim=-1)

        # Process text input
        language_cls = None
        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            language_cls = x[:, 0, :] if return_global else x
            if with_head:
                language_cls = self.language_head(language_cls)
            if out_norm:
                language_cls = F.normalize(language_cls, dim=-1)

        return vision_cls, language_cls


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def fix_huggingface_weight_MUSK(model, weight_path):
    from safetensors.torch import load_file

    # Load the model using timm and the downloaded weights
    checkpoint = load_file(weight_path)
    if 'model' in checkpoint:  # strip ddp
        checkpoint_model = checkpoint['model']
    elif 'module' in checkpoint:  # strip ddp
        checkpoint_model = checkpoint['module']
    else:
        checkpoint_model = checkpoint

    # fix state
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    for pos_embed_key in ("vision_pos_embed", "pos_embed", "beit3.encoder.embed_positions.A.weight"):
        if pos_embed_key in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model[pos_embed_key]
            embedding_size = pos_embed_checkpoint.shape[-1]
            if pos_embed_key == "beit3.encoder.embed_positions.A.weight":
                # being consistent with Fairseq, which starts from 2 for position embedding
                torchscale_model = True
                num_patches = model.beit3.vision_embed.num_patches
                num_extra_tokens = model.beit3.vision_embed.num_position_embeddings() + 2 - num_patches
            else:
                torchscale_model = False
                num_patches = model.patch_embed.num_patches
                num_extra_tokens = getattr(model, pos_embed_key).shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                if torchscale_model:
                    extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
                else:
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)

                # interpolate must be carried out on float
                pos_token_type = pos_tokens.dtype
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens.float(), size=(new_size, new_size), mode='bicubic', align_corners=False).to(
                    dtype=pos_token_type)

                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                if torchscale_model:
                    new_pos_embed = new_pos_embed.squeeze(0)
                checkpoint_model[pos_embed_key] = new_pos_embed

    return checkpoint_model

def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24,
        checkpoint_activations=checkpoint_activations,
    )

@register_model
def musk_large_patch16_384(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    model = MUSK(args, **kwargs)
    return model


class MUSK_vision_embedding_model(ModelWrapper):
    def __init__(self, args, **kwargs):
        """
        we stripe the MUSK model's vision parts to use as the embedding backbone for extracting features
        """
        super().__init__(args=args)
        self.embed_dim = args.encoder_embed_dim

        # Define heads for vision only
        self.vision_head = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # MUSK only use the retrieval head for image-text retrieval tasks.

        # Logit scale parameter initialization
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            with_head: bool = False,
            out_norm: bool = True,
            ms_aug: bool = False,
            scales: Optional[List[int]] = [1, 2],  # Default scales if need ms_aug
            max_split_size: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for vision-language model.
        Args:
            image: Input image tensor.
            with_head: Whether to apply linear heads. for CLIP tasks
            out_norm: Whether to normalize output embeddings.
            ms_aug: Enable multiscale feature augmentation.
            scales: List of scales for multiscale feature augmentation. [1, 2]  # Default scales
            max_split_size: Maximum split size for multiscale forward.

        Returns:
            vision_cls: Vision embeddings (normalized if out_norm).
            language_cls: Language embeddings (normalized if out_norm).
        """
        # Process image input
        vision_cls = None
        if image is not None:
            if ms_aug:
                vision_cls = MultiScaleForward(
                    model=self,
                    input=image,
                    scales=scales,
                    max_split_size=max_split_size
                )
                if with_head:
                    vision_cls = self.vision_head(vision_cls[:, :1024])
            else:
                outputs = self.beit3(visual_tokens=image)
                x = outputs["encoder_out"]
                vision_cls = x[:, 0, :]
                if with_head:
                    vision_cls = self.vision_head(vision_cls)

            if out_norm:
                vision_cls = F.normalize(vision_cls, dim=-1)

        return vision_cls


def get_MUSK_vision_embedding_model(pretrained=False,**kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    model = MUSK_vision_embedding_model(args, **kwargs)
    return model

