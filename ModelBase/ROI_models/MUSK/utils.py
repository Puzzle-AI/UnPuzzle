# --------------------------------------------------------
# MUSK: A Vision-Language Foundation Model for Precision Oncology
# Published in Nature, 2025
# GitHub Repository: https://github.com/lilab-stanford/MUSK
# Copyright (c) 2025 Stanford University, by Jinxi Xiang
# Licensed under the CC-BY-NC-ND 4.0 License (https://creativecommons.org/licenses/by-nc-nd/4.0/)
# Please see LICENSE for additional details.
# --------------------------------------------------------

import torch
import huggingface_hub
import os
from safetensors.torch import load_file
import math
import torch.nn.functional as F
from einops import rearrange
from timm.models import create_model

def xlm_tokenizer(tokens, tokenizer, max_len=100):
    tokens = tokenizer.encode(tokens)

    tokens = tokens[1:-1]  # remove eos and bos;
    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]

    tokens = [tokenizer.bos_token_id] + tokens[:] + [tokenizer.eos_token_id]
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)

    text_tokens = tokens + [tokenizer.pad_token_id] * (max_len - num_tokens)
    return text_tokens, padding_mask


def split_chessboard(x, num_split):
    """
        x: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0
    x_split = rearrange(x, 'b c (nh h) (nw w) -> (nh nw b) c h w', nh=num_split, nw=num_split)
    return x_split


def batched_forward(model, x, batch_size=-1):
    x_batched = x.split(batch_size)
    outs = []
    for x in x_batched:
        ret = model(
        image=x,
        out_norm=False,
        with_head=False,
        return_global=True,
        ms_aug=False
        )[0]
        outs.append(ret)
    return torch.cat(outs, dim=0)


"""
During MUSK pretraining, we used multi-scale image inputs, i.e., 
incorporating mixed magnifications. 
To align with this, we leverage MUSK's multi-scale capability during inference 
for linear probe and MIL tasks. Multi-scale was not applied to zero-shot tasks, 
as it is the CLS token that was used for modality alignment in contrastive learning.

The code implementation of MultiScaleForward() is derived from: ArXiv 2024 Vol. abs/2403.13043 
"""
def MultiScaleForward(
        model, 
        input, 
        scales=[1,2], 
        max_split_size=None, 
        ):
    
    assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
    assert input.shape[2] == input.shape[3], "Currently only square images are supported."

    b, c, input_size, _ = input.shape
    
    # image size for each scale
    img_sizes = [int(input_size * scale) for scale in scales]

    # prepare multiscale inputs
    max_split_size = max_split_size or input_size   # The maximum size of each split of image. Set as the input size by default
    num_splits = [math.ceil(size / max_split_size) for size in img_sizes]   # number of splits each scale
    input_multiscale = []
    for size, num_split in zip(img_sizes, num_splits):
        x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
        x = split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)

    # run feedforward on each scale
    outs_multiscale = [batched_forward(model, x, b) for x in input_multiscale]
    
    up_scale = rearrange(outs_multiscale[1], '(n b) c -> b n c', b=outs_multiscale[0].shape[0])
    out = torch.cat([outs_multiscale[0], up_scale.mean(1)], dim=-1)
    return out
