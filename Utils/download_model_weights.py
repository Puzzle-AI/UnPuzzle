"""
Download model weights  Script  ver： Dec 4th 12:20
uni, virchow, gigapath, puzzletuning are not included
"""
import os
import torch
import torch.nn as nn
from torchvision import models
import timm
from huggingface_hub import hf_hub_download

# model_save_path = '/data/hdd_1/model_weights/roi_models'
model_save_path = '/data/ssd_1/model_weights/wsi_models'
# os.environ["HF_TOKEN"] = "xxx"

# download from timm
model_name_list = [
    'vit_huge_patch14_224_in21k',
    'vit_large_patch16_224',
    'vit_large_patch16_384',
    'vit_small_patch16_224',
    'vit_small_patch16_384',
    'vit_tiny_patch16_224',
    'vit_tiny_patch16_384',
    'vit_base_patch16_224',
    'vit_base_patch16_384',
    'vgg16_bn',
    'vgg16',
    'vgg19_bn',
    'vgg19',
    'deit_base_patch16_224',
    'deit_base_patch16_384',
    'twins_pcpvt_base',
    'pit_b_224',
    'gcvit_base',
    'xcit_small_12_p16_224_dist',
    'xcit_small_12_p16_384_dist',
    'xcit_medium_24_p16_224_dist',
    'xcit_medium_24_p16_384_dist',
    'mvitv2_small_cls',
    'convit_base',
    'botnet26t_256',
    'densenet121',
    'xception',
    'pvt_v2_b0',
    'visformer_small',
    'coat_mini',
    'swin_base_patch4_window7_224',
    'swin_base_patch4_window12_384',
    'mobilenetv3_large_100',
    'mobilevit_s',
    'inception_v3',
    'crossvit_base_240',
    'efficientnet_b3',
    'efficientnet_b4',
    'vit_base_resnet50_384',
    'coat_lite_small'
]
for model_name in model_name_list:
    try:
        model = timm.create_model(model_name, pretrained=True)
        torch.save(model.state_dict(), f'{model_save_path}/{model_name}.pth')
    except Exception as e:
        print(e)

# download from torchvision.models
model = models.resnet18(pretrained=True)
torch.save(model.state_dict(), f'{model_save_path}/resnet18.pth')
model = models.resnet34(pretrained=True)
torch.save(model.state_dict(), f'{model_save_path}/resnet34.pth')
model = models.resnet50(pretrained=True)
torch.save(model.state_dict(), f'{model_save_path}/resnet50.pth')
model = models.resnet101(pretrained=True)
torch.save(model.state_dict(), f'{model_save_path}/resnet101.pth')


# # download from hf
# hf_hub_download("MahmoodLab/UNI", filename="UNI.pth", local_dir=model_save_path, force_download=True)