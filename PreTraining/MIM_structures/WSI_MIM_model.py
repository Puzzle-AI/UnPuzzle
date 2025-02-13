"""
Slide MIM models builder   Script  ver: Nov 11th 15:00

# References:
Based on MAE code.
https://github.com/facebookresearch/mae

"""

import sys
import torch
from pathlib import Path

# Go up 3 levels
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from ModelBase.Get_WSI_model import build_WSI_backbone_model
from PreTraining.MIM_structures.WSI_models_mae import SlideMAE_dec512d8b, SlideMAE_with_decoder


def build_WSI_MIM_model(
    SSL_name="MAE",
    model_name="SlideViT",
    dec_idx=None,
    MTL_token_num=None,
    local_weight_path=None,
    ROI_feature_dim=None,
    **kwargs
):
    """
    Build a WSI MIM model

    firstly, a slide backbone model will be build (which returns all the slide tokens)
    then, the slide tokens are pass to the MIM framework for reconstruction

    :param SSL_name: 'MAE' or 'MAE_decoder' etc.
    :param model_name: model name of the slide backbone
    :param dec_idx:import a decoder model (a segmentation model) with its name,
                    default None will use Transformer as decoder
    :param MTL_token_num: for some model with MTL token, set MTL token_num,
                    default None, this should follow the model design
    :param local_weight_path: loading the local weight for model building,default None,
    :param ROI_feature_dim: input feature dim
    """

    slide_backbone, default_ROI_feature_dim, slide_embed_dim = build_WSI_backbone_model(
        model_name, local_weight_path, ROI_feature_dim, MTL_token_num=MTL_token_num, **kwargs
    )
    ROI_feature_dim = ROI_feature_dim or default_ROI_feature_dim

    if model_name in ["SlideViT", "SlideVPT", "gigapath", "LongNet", "PathRWKV"]:
        slide_pos = True
        slide_ngrids = slide_backbone.slide_ngrids
    else:
        slide_pos = None
        slide_ngrids = None

    if SSL_name.split("_")[0] == "MAE":
        if dec_idx == None:
            MIM_Model = SlideMAE_dec512d8b(
                slide_backbone,
                ROI_feature_dim,
                slide_embed_dim,
                dec_idx=None,
                slide_pos=slide_pos,
                slide_ngrids=slide_ngrids,
                **kwargs
            )
        else:
            MIM_Model = SlideMAE_with_decoder(
                slide_backbone,
                ROI_feature_dim,
                slide_embed_dim,
                dec_idx=dec_idx,
                slide_pos=slide_pos,
                slide_ngrids=slide_ngrids,
                **kwargs
            )
    else:
        raise NotImplementedError

    return MIM_Model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_name = "PathRWKV"
    # MTL_token_num = 10  # can be 0
    # default_ROI_feature_dim = 768

    # model_name = 'SlideVPT'
    # MTL_token_num = 10  # can be 0
    # default_ROI_feature_dim = 768

    # model_name = 'gigapath'
    # MTL_token_num = 0
    # default_ROI_feature_dim = 1536

    model_name = "SlideViT"
    MTL_token_num = 10  # can be 0
    default_ROI_feature_dim = 768
    # Show data to see that Transformer can do multiple length, therefore can do MTL easily
    coords = torch.randn(2, 528, 2).to(device)
    image_features = torch.randn(2, 528, default_ROI_feature_dim).to(device)

    model = build_WSI_MIM_model(
        model_name=model_name,
        dec_idx=None,
        MTL_token_num=MTL_token_num,
        local_weight_path=None,
        ROI_feature_dim=default_ROI_feature_dim,
    )
    model.to(device)

    loss, pred, mask_patch_indicators = model(image_features, coords, mask_ratio=0.75)

    print(loss, "\n")
    print(loss.shape, "\n")  # scaler has no shape
    print(pred.shape, "\n")
    print(mask_patch_indicators.shape, "\n")
