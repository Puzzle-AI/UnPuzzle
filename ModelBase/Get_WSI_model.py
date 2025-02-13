"""
Build WSI level models      Script  ver： Feb 1st 21:00
"""

import os
import sys
import timm
import torch
import torch.nn as nn
import huggingface_hub
from typing import List
from pathlib import Path
import pandas as pd

# Go up 1 level
sys.path.append(str(Path(__file__).resolve().parent))

# always import baseline model modules
from WSI_models.WSI_Transformer.WSI_Transformer_blocks import Slide_Transformer_blocks, Prompt_Slide_Transformer_blocks
from WSI_models.WSI_Transformer.Pooling_backbone import PoolingModel

from ModelBase.MTL_modules.modules import MTL_Model_builder, WSI_MTL_state_fixer


def build_WSI_backbone_model(model_name="gigapath", local_weight_path=None,
                             ROI_feature_dim=None, MTL_token_num=0, **kwargs):
    """
    the function call the implemented modules to build slide backbone model,
    the embedded tiles are therefore transferred into slide level tokens

    the backbone model return all the slide level tokens instead of only task tokens etc,
    the backbone model will be used in the SSL framework for pretraining
    or MTL framework for downstream task training
    
    :param model_name: slide-level model name
    :param local_weight_path: path of loading the local weight of slide-level model
    :param ROI_feature_dim: the dim of input ROI embedding 
    :param MTL_token_num: the structural task token number
    
    """
    # fixme internal token
    # Hugging Face API token
    os.environ["HF_TOKEN"] = "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"

    if model_name == "SlideViT":
        default_ROI_feature_dim = 768  # the embedding input size for vit_base_patch16_224
        slide_embed_dim = 768  # the embedding size of slide model

        slide_backbone = Slide_Transformer_blocks(
            ROI_feature_dim=default_ROI_feature_dim, embed_dim=slide_embed_dim,
            MTL_token_num=MTL_token_num,
            depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=None, act_layer=None,
            **kwargs)
        # Transferlearning on Encoders
        if local_weight_path is None:
            # download the weights
            print("Transferlearning on Encoders, with auto weights from online")
            ViT_backbone_weights = timm.create_model("vit_base_patch16_224", pretrained=False).state_dict()
            del ViT_backbone_weights["patch_embed.proj.weight"]
            del ViT_backbone_weights["patch_embed.proj.bias"]
            slide_backbone.load_state_dict(ViT_backbone_weights, False)
        elif local_weight_path is False:
            print("Pretrained weights not required. Randomly initialized the model! ")
        elif os.path.exists(local_weight_path):
            print("Transferlearning on Encoders, with auto weights at", local_weight_path)
            state_dict = torch.load(local_weight_path, map_location="cpu")["model"]
            slide_backbone.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"not a valid weights file at {local_weight_path}")

        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name == "SlideVPT":
        default_ROI_feature_dim = 768  # the embedding input size for vit_base_patch16_224
        slide_embed_dim = 768  # the embedding size of slide model

        slide_backbone = Prompt_Slide_Transformer_blocks(
            Prompt_Token_num=20, VPT_type="Deep",
            ROI_feature_dim=default_ROI_feature_dim, embed_dim=slide_embed_dim, MTL_token_num=MTL_token_num,
            depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0,
            attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=None, act_layer=None, **kwargs)
        slide_backbone.Freeze()
        # Transferlearning on Encoders
        if local_weight_path is None:
            # download the weights
            print("Transferlearning on Encoders, with auto weights from online")
            ViT_backbone_weights = timm.create_model("vit_base_patch16_224", pretrained=False).state_dict()
            del ViT_backbone_weights["patch_embed.proj.weight"]
            del ViT_backbone_weights["patch_embed.proj.bias"]
            slide_backbone.load_state_dict(ViT_backbone_weights, False)
        elif local_weight_path is False:
            print("Pretrained weights not required. Randomly initialized the model! ")
        elif os.path.exists(local_weight_path):
            print("Transferlearning on Encoders, with auto weights at", local_weight_path)
            state_dict = torch.load(local_weight_path, map_location="cpu")["model"]
            slide_backbone.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"not a valid weights file at {local_weight_path}")

        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name == "gigapath":
        from WSI_models.gigapath.slide_encoder import gigapath_slide_enc12l768d

        default_ROI_feature_dim = 1536  # the embedding input size
        slide_embed_dim = 768  # the embedding size of slide model

        slide_backbone = gigapath_slide_enc12l768d(in_chans=default_ROI_feature_dim, global_pool=False, **kwargs)
        print("Slide encoder param #", sum(p.numel() for p in slide_backbone.parameters()))

        # download the weights
        if local_weight_path is None:
            hf_hub = "hf_hub:prov-gigapath/prov-gigapath"
            hub_name = hf_hub.split(":")[1]
            local_dir = os.path.join(os.path.expanduser("~"), ".cache/")

            huggingface_hub.hf_hub_download(
                hub_name, filename="slide_encoder.pth", local_dir=local_dir, force_download=False
            )
            local_weight_path = os.path.join(local_dir, "slide_encoder.pth")

        # build weight for slide_feature level
        if local_weight_path is False:
            print("Pretrained weights not required. Randomly initialized the model! ")

        elif os.path.exists(local_weight_path):
            state_dict = torch.load(local_weight_path, map_location="cpu")["model"]

            missing_keys, unexpected_keys = slide_backbone.load_state_dict(state_dict, strict=False)
            if len(missing_keys) > 0:
                for k in missing_keys:
                    print("Missing ", k)

            if len(unexpected_keys) > 0:
                for k in unexpected_keys:
                    print("Unexpected ", k)

            print("\033[92m Successfully Loaded Pretrained GigaPath model from {} \033[00m".format(local_weight_path))
        else:
            print(
                "\033[93m Pretrained weights not found at {}. Randomly initialized the model! "
                "\033[00m".format(local_weight_path))

        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name == "LongNet":
        from WSI_models.gigapath.slide_encoder import LongNetMTL_backbone
        from functools import partial

        default_ROI_feature_dim = 1536  # the embedding input size
        slide_embed_dim = 768  # the embedding size of slide model

        slide_backbone = LongNetMTL_backbone(
            in_chans=default_ROI_feature_dim,
            embed_dim=slide_embed_dim,
            depth=12,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            MTL_token_num=MTL_token_num,
            **kwargs)

        print("Slide encoder param #", sum(p.numel() for p in slide_backbone.parameters()))

        # download the weights
        if local_weight_path is None:
            hf_hub = "hf_hub:prov-gigapath/prov-gigapath"
            hub_name = hf_hub.split(":")[1]
            local_dir = os.path.join(os.path.expanduser("~"), ".cache/")

            huggingface_hub.hf_hub_download(
                hub_name, filename="slide_encoder.pth", local_dir=local_dir, force_download=False
            )
            local_weight_path = os.path.join(local_dir, "slide_encoder.pth")

        # build weight for slide_feature level
        if local_weight_path is False:
            print("Pretrained weights not required. Randomly initialized the model! ")

        elif os.path.exists(local_weight_path):
            state_dict = torch.load(local_weight_path, map_location="cpu")["model"]
            MTL_state_dict = WSI_MTL_state_fixer(state_dict, new_MTL_num=MTL_token_num)
            missing_keys, unexpected_keys = slide_backbone.load_state_dict(MTL_state_dict, strict=False)
            if len(missing_keys) > 0:
                for k in missing_keys:
                    print("Missing ", k)

            if len(unexpected_keys) > 0:
                for k in unexpected_keys:
                    print("Unexpected", k)

            print(
                "\033[92m Successfully Loaded Pretrained GigaPath model to MTL LongNet "
                "from {} \033[00m".format(local_weight_path)
            )
        else:
            print(
                "\033[93m Pretrained weights not found at {}. Randomly initialized the model! \033[00m".format(
                    local_weight_path
                )
            )
        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name == "PathRWKV":
        from WSI_models.path_rwkv.v6 import PathRWKVv6 as PathRWKV

        default_ROI_feature_dim = 1024
        slide_embed_dim = 1024

        slide_backbone = PathRWKV(**kwargs)

        if local_weight_path:
            state_dict = torch.load(local_weight_path, map_location="cpu", weights_only=True)
            slide_backbone.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded pretrained model from {local_weight_path}")

        elif local_weight_path == False:
            pass

        else:
            print("Pretrained weights not found. Initializing model by default method.")

        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name == "CLAM":
        from WSI_models.clam.clam import CLAM

        default_ROI_feature_dim = ROI_feature_dim  # the embedding input size
        slide_embed_dim = ROI_feature_dim  # the embedding output size

        slide_backbone = CLAM(embed_dim=ROI_feature_dim, MTL_token_num=MTL_token_num, **kwargs)

        if local_weight_path:
            state_dict = torch.load(local_weight_path, map_location="cpu", weights_only=True)
            slide_backbone.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded pretrained model from {local_weight_path}")

        elif local_weight_path == False:
            pass

        else:
            print("Pretrained weights not found. Initializing model by default method.")

        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name[0:3] == "UNI" or model_name[0:5] == "ABMIL":
        from WSI_models.WSI_Transformer.ABMIL_backbone import AttentionMILModel

        # ABMIL
        default_ROI_feature_dim = ROI_feature_dim  # the embedding input size
        slide_embed_dim = 768  # the embedding output size

        slide_backbone = AttentionMILModel(
            in_features=ROI_feature_dim,
            L=slide_embed_dim,
            D=384,
            MTL_token_num=MTL_token_num,
            gated_attention=True,
            slide_pos=False,
            **kwargs,
        )
        # L & D as per Uni standard
        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name[0:5] == "DSMIL":
        from WSI_models.WSI_Transformer.DSMIL_backbone import DSMIL

        # DSMIL
        default_ROI_feature_dim = ROI_feature_dim  # the embedding input size
        slide_embed_dim = 768  # the embedding output size

        slide_backbone = DSMIL(in_features=ROI_feature_dim, feats_size=slide_embed_dim, num_classes=MTL_token_num)
        return slide_backbone, default_ROI_feature_dim, slide_embed_dim


    elif model_name[0:6] == "SETMIL":
        from WSI_models.setmil.SETMIL_backbone import SETMIL

        default_ROI_feature_dim = ROI_feature_dim  # the embedding input size
        slide_embed_dim = 768  # the embedding output size
        slide_backbone = SETMIL(
            in_chans=ROI_feature_dim,
            embed_dim=slide_embed_dim,
            **kwargs,
        )
        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name[0:8] == "TransMIL":
        from WSI_models.WSI_Transformer.TransMIL_backbone import TransMIL

        default_ROI_feature_dim = ROI_feature_dim  # the embedding input size
        slide_embed_dim = 768  # the embedding output size

        slide_backbone = TransMIL(
            in_chans=ROI_feature_dim,
            embed_dim=slide_embed_dim,
            n_classes=MTL_token_num,
            **kwargs,
        )

        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name[0:4] == "DTFD":
        from WSI_models.WSI_Transformer.DTFD_backbone import DTFDMIL

        default_ROI_feature_dim = ROI_feature_dim  # the embedding input size
        slide_embed_dim = 768  # the embedding output size

        slide_backbone = DTFDMIL(
            in_features=ROI_feature_dim,
            feats_size=slide_embed_dim,
            # num_classes = MTL_token_num,
            **kwargs
        )

        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name[0:8] == "SlideMax" or model_name[0:8] == "SlideAve":
        default_ROI_feature_dim = ROI_feature_dim  # the embedding input size
        slide_embed_dim = ROI_feature_dim  # the embedding output size
        slide_backbone = PoolingModel(
            pooling_methods=model_name[5:8], ROI_feature_dim=ROI_feature_dim, embed_dim=slide_embed_dim, slide_pos=False
        )
        return slide_backbone, default_ROI_feature_dim, slide_embed_dim

    elif model_name[0:7] == "Virchow":
        pass

    else:
        raise NotImplementedError


# ------------------- WSI VQA Image Encoder (ViT) -------------------
# Pre-processed image tensor is passed through WSI model to obtain image embedding (ViT CLS token)
class ImageEncoder(nn.Module):
    def __init__(self, WSI_Encoder, embed_size=768):
        super(ImageEncoder, self).__init__()

        self.Image_Encoder = WSI_Encoder
        self.embed_convert = (
            nn.Linear(self.Image_Encoder.embed_dim, embed_size)
            if self.Text_Encoder.embed_dim != embed_size
            else nn.Identity()
        )

    def forward(self, images):
        # Process image through Image_Encoder to get the embeddings
        Image_cls_embedding = self.Image_Encoder(images)  # CLS token output from ViT [B,D]
        return self.embed_convert(Image_cls_embedding)


class slide_embedding_model_builder(nn.Module):
    def __init__(self, backbone: nn.Module):
        """
        :param backbone: slide_feature-level modeling model
        """
        super().__init__()

        self.backbone = backbone  # the output feature is [B, slide_embed_dim]

    def forward(self, image_features, img_coords):
        """
        Forward pass for the MTL Transformer.

        :param image_features: Tensor of shape [B, N, feature_dim],
                               where B is batch size, N is the number of patches/features per slide_feature.
        :return: slide_latent has
                 shape [B, output_dim] (output_dim may vary depending on the task).
        """
        slide_latent = self.backbone(image_features, img_coords)

        return slide_latent


def build_WSI_task_model(
        MTL_module_name=None,
        model_name="gigapath",
        local_weight_path=None,
        ROI_feature_dim=None,
        MTL_heads_configs: List[int] = None,  # the list of multiple MTL task head dimension for each task
        latent_feature_dim=128,
        Froze_backbone=False,
        Head_strategy=None,
        bin_df=None,
        **kwargs,
):
    """
    MTL_token_design: default 'Through_out' for putting the tokens in each transformer layer,
                                else 'latent' convert from the slide model output
    """
    assert MTL_heads_configs is not None

    # here we need to set config limitation for certain models
    if model_name in ["gigapath", "SlideMax", "SlideAve", "SETMIL", "TransMIL", "PathRWKV", "DTFD"]:
        """
        for some models have No MTL tokens in backbone, we obtain the feature map and project to MTL tokens
        """
        MTL_token_design = None
    elif model_name in ["CLAM", "ABMIL", "UNI", "DSMIL"]:
        """
        for the MIL-based method, they reduce the features into several task tokens (originally 1)
        we call them as MTL_token_design == "MIL_to" in the MTL model building process

        in model design, we put the MTL_token_num to the task projection in their model
        """
        MTL_token_design = "MIL_to"
    else:
        """
        for other models we may have MIL_to or Through_out design of MTL tokens, we follow the input specification

        foe example, SlideViT and Slide VPT support both:
            when MTL_token_num = 0 it go with MTL_token_design = None
            when MTL_token_num > 0 it go with MTL_token_design = "Through_out"
        """
        if model_name in ["SlideViT", "SlideVPT"]:
            if len(MTL_heads_configs) == 0:
                MTL_token_design = None
            else:
                MTL_token_design = "Through_out"
        else:
            print('WSI-MTL with model_name of ', model_name,
                  ' do not have defined MTL_token_design, assigning None')
            MTL_token_design = None

    if MTL_token_design == "Through_out" or MTL_token_design == "MIL_to":
        if Head_strategy == 'expression_bin':
            MTL_token_num = len(pd.unique(bin_df['bin']))
        else:
            MTL_token_num = len(MTL_heads_configs)
    else:
        MTL_token_num = 0

    slide_backbone, default_ROI_feature_dim, slide_embed_dim = build_WSI_backbone_model(
        model_name, local_weight_path, ROI_feature_dim, MTL_token_num=MTL_token_num, **kwargs
    )

    ROI_feature_dim = ROI_feature_dim or default_ROI_feature_dim

    Input_embedding_converter = (
        nn.Linear(ROI_feature_dim, default_ROI_feature_dim) if ROI_feature_dim != default_ROI_feature_dim else None
    )

    MTL_Model = MTL_Model_builder(
        slide_backbone,
        MTL_module_name=MTL_module_name,
        MTL_token_design=MTL_token_design,
        MTL_heads_configs=MTL_heads_configs,
        Input_embedding_converter=Input_embedding_converter,
        embed_dim=slide_embed_dim,
        MTL_feature_dim=latent_feature_dim,
        Froze_backbone=Froze_backbone,
        Head_strategy=Head_strategy,  # fixme Tim testing parts
        bin_df=bin_df)

    return MTL_Model


def build_WSI_prob_embedding_model(model_name="gigapath", local_weight_path=None, ROI_feature_dim=None, **kwargs):
    slide_backbone, default_ROI_feature_dim, slide_embed_dim = build_WSI_backbone_model(
        model_name, local_weight_path, ROI_feature_dim, **kwargs
    )

    slide_embedding_model = slide_embedding_model_builder(slide_backbone)
    return slide_embedding_model, default_ROI_feature_dim


def build_WSI_MLLM_model(model_name="gigapath", local_weight_path=None, ROI_feature_dim=None, **kwargs):
    pass


if __name__ == "__main__":
    # cuda issue
    print("cuda availability:", torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ROI_feature_dim = 1536
    # MTL_token_design='Through_out' or None

    slide_task_model = build_WSI_task_model(
        MTL_module_name=None,
        model_name="SETMIL",
        local_weight_path=None,
        ROI_feature_dim=ROI_feature_dim,
        latent_feature_dim=128,
        MTL_heads_configs=[3, 4],
    )

    slide_task_model = slide_task_model.to(dev).half()

    # play data (to see that Transformer can do multiple length, therefore can do MTL easily)
    coords = torch.randn(1, 528, 2).to(dev).half()
    x = torch.randn(1, 528, ROI_feature_dim).to(dev).half()
    y = slide_task_model(x, coords=coords)
    print(y)
