"""
Build ROI level models    Script  ver： Feb 9th 17:30
"""
import os
import sys
from pathlib import Path

# For convenience, import all path to sys
ROOT_PATH = Path(__file__).resolve().parent
sys.path.append(str(ROOT_PATH))

import timm
from pprint import pprint
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import huggingface_hub
from typing import List
import pandas as pd

import torch
import torch.nn as nn
import torchvision

from ROI_models.GetPromptModel import *
from ROI_models.Transformer_blocks import MTL_ViT_backbone
from ROI_models.VPT_structure import MTL_VPT_ViT_backbone
from MTL_modules.modules import MTL_Model_builder

def get_embedding_transform(model_name=None,edge_size=224):
    DEFAULT_CROP_PCT = 0.875
    DEFAULT_CROP_MODE = 'center'
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
    IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
    IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)
    OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    if model_name[0:4] == 'musk' or model_name[0:4] == "Musk" or model_name[0:4] == "MUSK":
        edge_size = 384
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(384, interpolation=3, antialias=True),
            torchvision.transforms.CenterCrop((384, 384)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
    elif model_name[0:3] == 'uni' or model_name[0:3] == "Uni" or model_name[0:3] == "UNI":
        edge_size = 224
        transform = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
    elif model_name[0:7] == 'Virchow' or model_name[0:8] == 'gigapath':
        edge_size = 224
        transform = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
    else:
        transform = transforms.Compose([
            transforms.Resize(edge_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

    return edge_size, transform

# get model
def build_ROI_backbone_model(num_classes=0, edge_size=224, model_name=None,
                             pretrained_backbone=True, disable_weight_check=False):
    """
    This build backbone model for ROI tasks

    :param num_classes: classification head design required by downstream,
    0 for taking the feature, -1 for taking the original feature map, >0 to be do classification

    :param edge_size: the input edge size of the dataloder
    :param model_name: the model we are going to use. by the format of Model_size_other_info

    :param pretrained_backbone: The backbone CNN is initiate randomly or by its official Pretrained models
    :param disable_weight_check: disable weight loading check for embedding model, default False

    :return: prepared ROI model
    """
    # fixme internal token
    # Hugging Face API token
    os.environ["HF_TOKEN"] = "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"
    # default transforms
    transforms = None
    expected_pred_code = None

    # set this mark as some model need tobe warpped
    load_at_backbone = False

    if pretrained_backbone == True or pretrained_backbone == False:
        pretrained_weight_path = None
        load_weight_online = pretrained_backbone

    elif pretrained_backbone == None:
        pretrained_weight_path = None
        load_weight_online = True

    else:  # str path
        if os.path.exists(pretrained_backbone):
            pretrained_weight_path = pretrained_backbone
        else:
            pretrained_weight_path = None
            raise  # the manual pretrained_weight_path is missing, check your file there
        load_weight_online = False

    if model_name[0:5] == 'ViT_h':
        # Transfer learning for ViT
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_huge_patch14_224_in21k', pretrained=load_weight_online,
                                      num_classes=num_classes)
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float()
        else:
            raise ValueError('not a available image size with', model_name)

    elif model_name[0:5] == 'ViT_l':
        # Transfer learning for ViT
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_large_patch16_224', pretrained=load_weight_online,
                                      num_classes=num_classes)
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([3, 0, 0, 3, 2, 1, 0, 0, -1, 0]).float()
        elif edge_size == 384:
            model = timm.create_model('vit_large_patch16_384', pretrained=load_weight_online,
                                      num_classes=num_classes)
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([3, 0, 0, 4, 1, 1, 0, 0, -2, 0]).float()
        else:
            raise NotImplementedError('not a available image size with', model_name)

    elif model_name[0:5] == 'ViT_s':
        # Transfer learning for ViT
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_small_patch16_224', pretrained=load_weight_online,
                                      num_classes=num_classes)
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([0, -1, 0, -1, -1, 0, 0, 0, -2, 0]).float()
        elif edge_size == 384:
            model = timm.create_model('vit_small_patch16_384', pretrained=load_weight_online,
                                      num_classes=num_classes)
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([0, 0, -1, 0, 0, 0, 0, 0, 0, 0]).float()
        else:
            raise NotImplementedError('not a available image size with', model_name)

    elif model_name[0:5] == 'ViT_t':
        # Transfer learning for ViT
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_tiny_patch16_224', pretrained=load_weight_online,
                                      num_classes=num_classes)
            expected_pred_code = torch.tensor([7, 1, 2, 0, 2, -7, 1, -1, -1, 0]).float()
        elif edge_size == 384:
            model = timm.create_model('vit_tiny_patch16_384', pretrained=load_weight_online,
                                      num_classes=num_classes)
            expected_pred_code = torch.tensor([5, 1, 4, 0, 3, -5, 1, 0, -1, 0]).float()
        else:
            raise NotImplementedError('not a available image size with', model_name)

    elif model_name[0:5] == 'ViT_b' or model_name[0:3] == 'ViT' or model_name[0:3] == 'vit':  # vit_base
        # ->  torch.Size([1, 768])
        # Transfer learning for ViT
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_base_patch16_224', pretrained=load_weight_online,
                                      num_classes=num_classes)
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([0, 0, 0, 0, -1, 0, -1, 0, 0, 1]).float()
        elif edge_size == 384:
            model = timm.create_model('vit_base_patch16_384', pretrained=load_weight_online,
                                      num_classes=num_classes)
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([1, -2, 0, -1, 0, 1, -2, 0, 0, 1]).float()
        else:
            raise NotImplementedError('not a available image size with', model_name)

    elif model_name[0:3] == 'vgg':
        # Transfer learning for vgg16_bn
        model_names = timm.list_models('*vgg*')
        pprint(model_names)
        if model_name[0:8] == 'vgg16_bn':
            model = timm.create_model('vgg16_bn', pretrained=load_weight_online, num_classes=num_classes)
        elif model_name[0:5] == 'vgg16':
            model = timm.create_model('vgg16', pretrained=load_weight_online, num_classes=num_classes)
        elif model_name[0:8] == 'vgg19_bn':
            model = timm.create_model('vgg19_bn', pretrained=load_weight_online, num_classes=num_classes)
        elif model_name[0:5] == 'vgg19':
            model = timm.create_model('vgg19', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:3] == 'uni' or model_name[0:3] == "Uni" or model_name[0:3] == "UNI":
        assert edge_size == 224
        #  ->  torch.Size([1, 1024])
        if load_weight_online:
            # fixme this somehow failed, we build UNI with manual config and huggingface
            '''
            model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                                      init_values=1e-5, dynamic_img_size=True)
            '''
            from huggingface_hub import hf_hub_download
            # Download the model weights
            pretrained_weight_path = hf_hub_download(repo_id="MahmoodLab/UNI", filename="pytorch_model.bin",
                                                     use_auth_token=os.environ["HF_TOKEN"])

            # Load the model using timm and the downloaded weights
            # Create the ViT model using the configuration read from Hugging Face
            model = timm.create_model(
                "vit_large_patch16_224",  # Model architecture
                pretrained=False,  # Load pretrained weights
                img_size=224,  # Image size is 224x224
                num_classes=num_classes,  # No classification head (feature extractor mode)
                init_values=1.0,  # Layer scale initialization value
                global_pool="token",  # Use token pooling (default for ViT)
                dynamic_img_size=True  # Allow dynamic image size
            )
            model.load_state_dict(torch.load(pretrained_weight_path), False)
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 1]).float()
        else:
            model = timm.create_model(
                "vit_large_patch16_224",  # Model architecture
                pretrained=True,  # Load pretrained weights
                img_size=224,  # Image size is 224x224
                num_classes=num_classes,  # No classification head (feature extractor mode)
                init_values=1.0,  # Layer scale initialization value
                global_pool="token",  # Use token pooling (default for ViT)
                dynamic_img_size=True  # Allow dynamic image size
            )

    # VPT feature embedding
    elif model_name[0:3] == 'VPT' or model_name[0:3] == 'vpt':
        # ->  torch.Size([1, 768])
        if load_weight_online:
            from huggingface_hub import hf_hub_download
            # Define the repo ID
            repo_id = "Tianyinus/PuzzleTuning_VPT"

            # Download the base state dictionary file
            base_state_dict_path = hf_hub_download(repo_id=repo_id,
                                                   filename="PuzzleTuning/Archive/ViT_b16_224_Imagenet.pth")

            # Download the prompt state dictionary file
            prompt_state_dict_path = hf_hub_download(repo_id=repo_id,
                                                     filename="PuzzleTuning/Archive/ViT_b16_224_timm_PuzzleTuning_SAE_CPIAm_Prompt_Deep_tokennum_20_E_199_promptstate.pth")

            # Load these weights into your model
            base_state_dict = torch.load(base_state_dict_path)
            prompt_state_dict = torch.load(prompt_state_dict_path)
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([0, 0, 0, 0, 1, -1, 0, 0, 0, 1]).float()

        else:
            prompt_state_dict = None
            base_state_dict = 'timm'

        # Build your model using the loaded state dictionaries
        model = build_promptmodel(prompt_state_dict=prompt_state_dict,
                                  base_state_dict=base_state_dict,
                                  num_classes=num_classes)

    elif model_name[0:4] == 'deit':  # Transfer learning for DeiT
        model_names = timm.list_models('*deit*')
        pprint(model_names)
        if edge_size == 384:
            model = timm.create_model('deit_base_patch16_384',
                                      pretrained=load_weight_online, num_classes=num_classes)
        elif edge_size == 224:
            model = timm.create_model('deit_base_patch16_224',
                                      pretrained=load_weight_online, num_classes=num_classes)
        else:
            pass

    elif model_name[0:4] == 'musk' or model_name[0:4] == "Musk" or model_name[0:4] == "MUSK":
        '''
        # GitHub Repository: https://github.com/lilab-stanford/MUSK
        '''
        assert num_classes == 0, 'we only use MUSK as encoder '
        assert edge_size == 384
        #  ->  torch.Size([1, 1024])
        # Create the ViT model using the code
        from ROI_models.MUSK.modeling import get_MUSK_vision_embedding_model,fix_huggingface_weight_MUSK
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

        model = get_MUSK_vision_embedding_model()
        fix_weight_state = None

        if pretrained_weight_path is not None:
            fix_weight_state = fix_huggingface_weight_MUSK(model, pretrained_weight_path)

        if load_weight_online:
            from huggingface_hub import hf_hub_download
            # Download the model weights
            online_weight_path = hf_hub_download(repo_id="xiangjx/musk", filename="model.safetensors",
                                                 use_auth_token=os.environ["HF_TOKEN"])
            fix_weight_state = fix_huggingface_weight_MUSK(model, online_weight_path)

        if fix_weight_state is not None:
            model.load_state_dict(fix_weight_state, False)
            load_at_backbone = True
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float()
        else:
            pass

        # official image transform from MUSK
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(384, interpolation=3, antialias=True),
            torchvision.transforms.CenterCrop((384, 384)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    elif model_name[0:5] == 'conch' or model_name[0:5] == "Conch" or model_name[0:5] == "CONCH":
        '''
        # GitHub Repository: https://github.com/mahmoodlab/CONCH
        '''
        assert num_classes==0, 'we only use CONCH as encoder '
        # here we fetch the CoCa.visual as CONCH_vision_embedding_model
        #  ->  torch.Size([1, 512])
        from ROI_models.CONCH import get_CONCH_vision_embedding_model
        force_image_size = edge_size if edge_size != 448 else None

        if load_weight_online:
            model, transforms = get_CONCH_vision_embedding_model(checkpoint_path="hf_hub:MahmoodLab/conch",
                                                                 force_image_size=force_image_size,
                                                                 hf_auth_token=os.environ["HF_TOKEN"])
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float()
            load_at_backbone = True
        elif pretrained_weight_path is not None:
            model, transforms = get_CONCH_vision_embedding_model(checkpoint_path=pretrained_weight_path,
                                                                 force_image_size=force_image_size)
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float()
            load_at_backbone = True
        else:
            model, transforms = get_CONCH_vision_embedding_model(checkpoint_path=None,
                                                                 force_image_size=force_image_size)


    elif model_name[0:5] == 'twins':  # Transfer learning for twins

        model_names = timm.list_models('*twins*')
        pprint(model_names)
        model = timm.create_model('twins_pcpvt_base', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:5] == 'pit_b':  # Transfer learning for PiT
        assert edge_size == 224
        model_names = timm.list_models('*pit*')
        pprint(model_names)
        model = timm.create_model('pit_b_224', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:5] == 'gcvit':  # Transfer learning for gcvit
        assert edge_size == 224
        model_names = timm.list_models('*gcvit*')
        pprint(model_names)
        model = timm.create_model('gcvit_base', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:6] == 'xcit_s':  # Transfer learning for XCiT
        model_names = timm.list_models('*xcit*')
        pprint(model_names)
        if edge_size == 384:
            model = timm.create_model('xcit_small_12_p16_384_dist', pretrained=load_weight_online,
                                      num_classes=num_classes)
        elif edge_size == 224:
            model = timm.create_model('xcit_small_12_p16_224_dist', pretrained=load_weight_online,
                                      num_classes=num_classes)
        else:
            pass

    elif model_name[0:6] == 'xcit_m':  # Transfer learning for XCiT
        model_names = timm.list_models('*xcit*')
        pprint(model_names)
        if edge_size == 384:
            model = timm.create_model('xcit_medium_24_p16_384_dist', pretrained=load_weight_online,
                                      num_classes=num_classes)
        elif edge_size == 224:
            model = timm.create_model('xcit_medium_24_p16_224_dist', pretrained=load_weight_online,
                                      num_classes=num_classes)
        else:
            pass

    elif model_name[0:6] == 'mvitv2':  # Transfer learning for MViT v2 small  fixme bug in model!
        model_names = timm.list_models('*mvitv2*')
        pprint(model_names)
        model = timm.create_model('mvitv2_small_cls', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:6] == 'convit':  # Transfer learning for ConViT fixme bug in model!
        assert edge_size == 224
        model_names = timm.list_models('*convit*')
        pprint(model_names)
        model = timm.create_model('convit_base', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:6] == 'swin_b':  # Transfer learning for Swin Transformer (swin_b_384)
        model_names = timm.list_models('*swin*')
        pprint(model_names)  # swin_base_patch4_window12_384  swin_base_patch4_window12_384_in22k
        if edge_size == 384:
            model = timm.create_model('swin_base_patch4_window12_384', pretrained=load_weight_online,
                                      num_classes=num_classes)
        elif edge_size == 224:
            model = timm.create_model('swin_base_patch4_window7_224', pretrained=load_weight_online,
                                      num_classes=num_classes)
        else:
            raise NotImplementedError(model_name)

    elif model_name[0:6] == 'ResNet':  # Transfer learning for the ResNets
        if model_name[0:8] == 'ResNet18':
            model = torchvision.models.resnet18(pretrained=load_weight_online)
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([0, 0, 0, 1, 0, 0, 1, 1, 0, 0]).float()
        elif model_name[0:8] == 'ResNet34':
            model = torchvision.models.resnet34(pretrained=load_weight_online)
            expected_pred_code = torch.tensor([0, 0, 1, 0, 0, 1, 0, 0, 1, 0]).float()
        elif model_name[0:8] == 'ResNet50':
            model = torchvision.models.resnet50(pretrained=load_weight_online)
            expected_pred_code = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float()
        elif model_name[0:9] == 'ResNet101':
            model = torchvision.models.resnet101(pretrained=load_weight_online)
            expected_pred_code = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float()
        else:
            print('this model is not defined in get model')
            return -1

        # Custom Flatten module
        class Feature_Flatten(nn.Module):
            def forward(self, x):
                # Flatten the tensor while keeping the batch dimension
                return x.view(x.size(0), -1)

        num_ftrs = model.fc.in_features
        if num_classes > 0:
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif num_classes == 0:
            model.fc = Feature_Flatten()
            # ResNet18 ->  torch.Size([1, 512])
            # ResNet34 ->  torch.Size([1, 512])
            # ResNet50 ->  torch.Size([1, 2048])
            # ResNet101 ->  torch.Size([1, 2048])
        elif num_classes == -1:  # call for original feature shape
            model.fc = nn.Identity()
        else:
            raise NotImplementedError

    elif model_name[0:7] == 'bot_256' :  # Model: BoT
        assert edge_size == 256
        model_names = timm.list_models('*bot*')
        pprint(model_names)
        # NOTICE: we find no weight for BoT in timm
        # ['botnet26t_256', 'botnet50ts_256', 'eca_botnext26ts_256']
        model = timm.create_model('botnet26t_256', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:7] == 'Virchow':
        """
        Notice that the Virchow models are a bit different as they need to condatenate the features after the 
        Transformer backbone. therefore, we need to warp encoders as backbone and load the weights to it.
        """
        assert edge_size == 224
        # -> [1, 2560]
        class VirchowTaskModel(nn.Module):
            """
            A model that integrates the Virchow backbone and extracts a class token or a full tile embedding based on the version.

            Args:
                Virchow_backbone_model: The pre-trained Virchow backbone model.
                Virchow_version: The version of Virchow model being used ('1' or '2').
                num_classes: If non-zero, returns only the class token (used for classification tasks).
                             If zero, concatenates the class token and mean patch token for feature extraction.
            """

            def __init__(self, Virchow_backbone_model, Virchow_version='1', num_classes=0):
                super(VirchowTaskModel, self).__init__()
                assert Virchow_version in ['1', '2'], "Virchow_version must be either '1' or '2'"

                self.Virchow_backbone_model = Virchow_backbone_model
                self.Virchow_version = Virchow_version
                self.num_classes = num_classes
                if self.num_classes > 0:
                    self.embed_dim = Virchow_backbone_model.embed_dim
                else:
                    self.embed_dim = 2 * Virchow_backbone_model.embed_dim

            def forward(self, x):
                """
                Memo from original authors:
                    output = model(image)  # size: 1 x 257 (or 261 for Virchow2) x 1280

                    class_token = output[:, 0]    # size: 1 x 1280
                    patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280
                    # for Virchow 2 its:
                    # patch_tokens = output[:, 5:]  # size: 1 x 260 x 1280, tokens 1-4 are register tokens so we ignore those

                    # concatenate class token and average pool of patch tokens
                    embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
                    # We concatenate the class token and the mean patch token to create the final tile embedding.

                    Notice:
                    In more resource constrained settings, one can experiment with using just class token or the mean patch
                    token. For downstream tasks with dense outputs (i.e. segmentation), the 256 x 1280 tensor of patch
                    tokens can be used.
                """
                # Pass image through the backbone model
                output = self.Virchow_backbone_model(x)  # size: 1 x 257 (or 261 for Virchow2) x 1280

                # Extract class token
                class_token = output[:, 0]  # size: 1 x 1280 or 1 x num_classes (if used for classification task)

                # If num_classes is non-zero, return only the class token (for classification tasks)
                if self.num_classes > 0:
                    return class_token

                # For Virchow2, ignore the first 4 tokens (register tokens)
                if self.Virchow_version == '2':
                    patch_tokens = output[:, 5:]  # size: 1 x 260 x 1280
                else:
                    patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

                # Compute final embedding by concatenating class token and mean of patch tokens
                embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

                return embedding

        from timm.layers import SwiGLUPacked

        if model_name[0:8] == 'Virchow2':
            Virchow_version = '2'
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([1, 0, 0, 0, 1, 0, 0, -1, 0, 0]).float()
            # Model configuration from JSON
            config = {
                "architecture": "vit_huge_patch14_224",
                "model_args": {
                    "img_size": 224,
                    "init_values": 1e-5,
                    "num_classes": 0,
                    "reg_tokens": 4,
                    "mlp_ratio": 5.3375,
                    "global_pool": "",
                    "dynamic_img_size": True
                },
                "pretrained_cfg": {
                    "tag": "virchow_v2",
                    "custom_load": False,
                    "input_size": [3, 224, 224],
                    "fixed_input_size": False,
                    "interpolation": "bicubic",
                    "crop_pct": 1.0,
                    "crop_mode": "center",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "num_classes": 0,
                    "pool_size": None,
                    "first_conv": "patch_embed.proj",
                    "classifier": "head",
                    "license": "CC-BY-NC-ND-4.0"
                }
            }
            # Create the model using timm
            Virchow_backbone_model = timm.create_model(
                config['architecture'],
                pretrained=False,
                reg_tokens=config['model_args']['reg_tokens'],
                img_size=config['model_args']['img_size'],
                init_values=config['model_args']['init_values'],
                mlp_ratio=config['model_args']['mlp_ratio'],
                num_classes=num_classes,
                global_pool=config['model_args']['global_pool'],
                dynamic_img_size=config['model_args']['dynamic_img_size'],
                mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

        else:  # Virchow1
            Virchow_version = '1'
            # Define the expected embedding tensor
            expected_pred_code = torch.tensor([0, 0, -2, -1, 0, -2, 0, 0, 0, 0]).float()
            # Model configuration from JSON
            config = {
                "architecture": "vit_huge_patch14_224",
                "model_args": {
                    "img_size": 224,
                    "init_values": 1e-5,
                    "num_classes": 0,
                    "mlp_ratio": 5.3375,
                    "global_pool": "",
                    "dynamic_img_size": True
                },
                "pretrained_cfg": {
                    "tag": "virchow_v1",
                    "custom_load": False,
                    "input_size": [3, 224, 224],
                    "fixed_input_size": False,
                    "interpolation": "bicubic",
                    "crop_pct": 1.0,
                    "crop_mode": "center",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "num_classes": 0,
                    "pool_size": None,
                    "first_conv": "patch_embed.proj",
                    "classifier": "head",
                    "license": "Apache 2.0"
                }
            }

            # Create the model using timm
            Virchow_backbone_model = timm.create_model(
                config['architecture'],
                pretrained=False,
                img_size=config['model_args']['img_size'],
                init_values=config['model_args']['init_values'],
                mlp_ratio=config['model_args']['mlp_ratio'],
                num_classes=num_classes,
                global_pool=config['model_args']['global_pool'],
                dynamic_img_size=config['model_args']['dynamic_img_size'],
                mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

        # ref: https://huggingface.co/paige-ai/Virchow
        if load_weight_online:
            from huggingface_hub import hf_hub_download
            if model_name[0:8] == 'Virchow2':
                Virchow_version = '2'
                # Download the model weights
                online_weight_path = hf_hub_download(repo_id="paige-ai/Virchow2", filename="pytorch_model.bin",
                                                     use_auth_token=os.environ["HF_TOKEN"])
            else:  # Virchow1
                Virchow_version = '1'
                # Download the model weights
                online_weight_path = hf_hub_download(repo_id="paige-ai/Virchow", filename="pytorch_model.bin",
                                                     use_auth_token=os.environ["HF_TOKEN"])
            transforms = create_transform(**resolve_data_config(Virchow_backbone_model.pretrained_cfg,
                                                                model=Virchow_backbone_model))
            Virchow_backbone_model.load_state_dict(torch.load(online_weight_path), False)
        elif pretrained_weight_path:
            Virchow_backbone_model.load_state_dict(torch.load(pretrained_weight_path), False)
            load_at_backbone = True
        # Virchow 1 and 2 are special as we are loading the backbone and stack the model here,
        # if loading their weights, we should load the weights to the warped VirchowTaskModel
        print('Virchow 1/2 are special as we are warpping and loading to the backbone')

        model = VirchowTaskModel(Virchow_backbone_model=Virchow_backbone_model,
                                 Virchow_version=Virchow_version,
                                 num_classes=num_classes)  # <=0 to be set to feature extracting or as MTL backbone

    # prov-gigapath feature embedding ViT
    elif model_name[0:8] == 'gigapath':
        assert edge_size == 224
        # ->  torch.Size([1, 1536])
        # ref: https://www.nature.com/articles/s41586-024-07441-w
        # Model configuration from your JSON
        config = {
            "architecture": "vit_giant_patch14_dinov2",
            "num_classes": num_classes,  # original is 0
            "num_features": 1536,
            "global_pool": "token",
            "model_args": {
                "img_size": 224,
                "in_chans": 3,
                "patch_size": 16,
                "slide_embed_dim": 1536,
                "depth": 40,
                "num_heads": 24,
                "init_values": 1e-05,
                "mlp_ratio": 5.33334,
                "num_classes": 0
            }
        }

        # Create the model using timm
        model = timm.create_model(
            config['architecture'],
            pretrained=False,  # Set to True if you want to load pretrained weights
            img_size=config['model_args']['img_size'],
            in_chans=config['model_args']['in_chans'],
            patch_size=config['model_args']['patch_size'],
            embed_dim=config['model_args']['slide_embed_dim'],
            depth=config['model_args']['depth'],
            num_heads=config['model_args']['num_heads'],
            init_values=config['model_args']['init_values'],
            mlp_ratio=config['model_args']['mlp_ratio'],
            num_classes=config['model_args']['num_classes']
        )
        # Define the expected embedding tensor
        expected_pred_code = torch.tensor([-1, 0, -1, 0, -1, 1, 0, 0, 0, 0]).float()

        if load_weight_online:
            from huggingface_hub import hf_hub_download
            # fixme if "HF_TOKEN" failed, use your own hugging face token and register for the project gigapath
            # Download the model weights
            online_weight_path = hf_hub_download(repo_id="prov-gigapath/prov-gigapath",
                                                 filename="pytorch_model.bin",
                                                 use_auth_token=os.environ["HF_TOKEN"])
            model.load_state_dict(torch.load(online_weight_path), False)

    elif model_name[0:8] == 'densenet':  # Transfer learning for densenet
        model_names = timm.list_models('*densenet*')
        pprint(model_names)
        model = timm.create_model('densenet121', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:8] == 'xception':  # Transfer learning for Xception
        model_names = timm.list_models('*xception*')
        pprint(model_names)
        model = timm.create_model('xception', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:9] == 'pvt_v2_b0':  # Transfer learning for PVT v2 (todo not applicable with torch summary)
        model_names = timm.list_models('*pvt_v2*')
        pprint(model_names)
        model = timm.create_model('pvt_v2_b0', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:9] == 'visformer':  # Transfer learning for Visformer
        assert edge_size == 224
        model_names = timm.list_models('*visformer*')
        pprint(model_names)
        model = timm.create_model('visformer_small', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:9] == 'coat_mini':  # Transfer learning for coat_mini
        assert edge_size == 224
        model_names = timm.list_models('*coat*')
        pprint(model_names)
        model = timm.create_model('coat_mini', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:11] == 'mobilenetv3':  # Transfer learning for mobilenetv3
        model_names = timm.list_models('*mobilenet*')
        pprint(model_names)
        model = timm.create_model('mobilenetv3_large_100', pretrained=load_weight_online,
                                  num_classes=num_classes)

    elif model_name[0:11] == 'mobilevit_s':  # Transfer learning for mobilevit_s
        model_names = timm.list_models('*mobilevit*')
        pprint(model_names)
        model = timm.create_model('mobilevit_s', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:11] == 'inceptionv3':  # Transfer learning for Inception v3
        model_names = timm.list_models('*inception*')
        pprint(model_names)
        model = timm.create_model('inception_v3', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:13] == 'crossvit_base':  # Transfer learning for crossvit_base  (todo not okey with torch summary)
        model_names = timm.list_models('*crossvit_base*')
        pprint(model_names)
        model = timm.create_model('crossvit_base_240', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:14] == 'efficientnet_b':  # Transfer learning for efficientnet_b3,4
        model_names = timm.list_models('*efficientnet*')
        pprint(model_names)
        model = timm.create_model(model_name[0:15], pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:14] == 'ResN50_ViT_384':  # ResNet+ViT融合模型384
        model_names = timm.list_models('*vit_base_resnet*')
        pprint(model_names)
        model = timm.create_model('vit_base_resnet50_384', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:15] == 'coat_lite_small':  # Transfer learning for coat_lite_small
        assert edge_size == 224
        model_names = timm.list_models('*coat*')
        pprint(model_names)
        model = timm.create_model('coat_lite_small', pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:17] == 'efficientformer_l':  # Transfer learning for efficientnet_b3,4
        assert edge_size == 224
        model_names = timm.list_models('*efficientformer*')
        pprint(model_names)
        model = timm.create_model(model_name[0:18], pretrained=load_weight_online, num_classes=num_classes)

    elif model_name[0:10] == 'VisionRWKV':
        from ROI_models.vrwkv.v6 import VisionRWKV

        if num_classes == 0:
            model = VisionRWKV(img_size=edge_size)  # output: (B, N, D)

        elif num_classes == 1:
            model = VisionRWKV(img_size=edge_size, cls_token=1, output_cls_token=True)  # output: (B, D)

        else:
            model = VisionRWKV(img_size=edge_size, cls_token=num_classes)  # output: (B, N, num_classes)

        # Transferlearning on Encoders
        if pretrained_backbone:
            if not pretrained_weight_path:  # fixme change to use 'if load_weight_online'
                online_weight_path = huggingface_hub.hf_hub_download("OpenGVLab/Vision-RWKV",
                                                                         filename="vrwkv6_b_in1k_224.pth")

            state_dict = torch.load(online_weight_path, weights_only=True)["state_dict"]
            model_state_dict = model.state_dict()
            filtered_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()
                                   if k.replace("backbone.", "") in model_state_dict}
            model.load_state_dict(filtered_state_dict, strict=False)

    else:
        raise NotImplementedError('\nThe model'+ model_name+ 'with the edge size of'
                                  + edge_size+"is not defined in the script!!\n")

    # load state
    try:
        if pretrained_weight_path is None and load_weight_online:
            print('pretrained_weight_path: already load online weight')
        else:
            print('pretrained_weight_path:', pretrained_weight_path)

        if pretrained_weight_path is None:
            pass
        else:
            if os.path.exists(pretrained_weight_path) and not load_at_backbone:
                missing_keys, unexpected_keys = model.load_state_dict(torch.load(pretrained_weight_path), False)
                # missing_keys, unexpected_keys = model.load_state_dict(pretrained_weight_path, False)
                if len(missing_keys) > 0:
                    for k in missing_keys:
                        print("Missing ", k)

                if len(unexpected_keys) > 0:
                    for k in unexpected_keys:
                        print("Unexpected ", k)
    except Exception as e:
        print(e)
        print(f'Warning: loading pretrained_weight failed')

    try:
        # Print the model to verify
        print(model)
        # Move model to dev for testing tensor inference
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(dev)

        # Define a fixed, specific pseudo image tensor
        img = torch.tensor(  # Shape: (1, 3, edge_size, edge_size)
            [[[[0.1 * (i + j + k) % 1.0 for i in range(edge_size)] for j in range(edge_size)]
              for k in range(3)] for _ in range(1)], dtype=torch.float32, device=dev)
        # put the results to cpu for evaluation
        preds = model(img).to('cpu')  # (1, class_number or embedding dim)
        # put model back to cpu for clean the gpu (we put stacked model to GPU after building task model)
        model.to("cpu")

        print('Build ROI model with in/out shape: ', img.shape, ' -> ', preds.shape)
        print("ROI model param size #", sum(p.numel() for p in model.parameters()))
    except:
        print("Problem exist in the model defining process！！")
        raise  # Problem exist in the model defining process
    else:
        if (load_weight_online or pretrained_weight_path) and num_classes == 0:
            # Convert the model's prediction to integers for comparison
            model_pred_code = preds[0][:10].int().float()  # (with top-10 features as int code)

            if expected_pred_code is None:
                print('model_pred_code:', model_pred_code)
                print('The model is not recorded with embedding check, pass')
            elif disable_weight_check:
                print('The model weight loadding checking is disabled, '
                      'pls ensure the weight path is as expacted')
            else:
                print('model_pred_code:', model_pred_code)
                print('expected_pred_code:', expected_pred_code)

                # Calculate loss
                loss = nn.MSELoss()(model_pred_code, expected_pred_code)

                # Check if loss is zero (exact match with expected embedding)
                if loss.item() == 0:
                    print('The embedding is checked for model weight loadding')
                else:
                    raise ValueError("The model prediction does not match the expected embedding! "
                                     "Please check if you have load the intented weight for the ROI model \n "
                                     "or if you are using finetuned weight, pls set disable_weight_check to True")

        print('model is ready now!')
        return model  # todo for future maybe we return model, transforms


# ------------------- ROI VQA Image Encoder (ViT) -------------------
# Pre-processed image tensor is passed through the Vision Transformer (ViT), to obtain image embedding (ViT CLS token)
class ImageEncoder(nn.Module):
    def __init__(self, model_name='uni', embed_size=768):
        super(ImageEncoder, self).__init__()

        from timm.layers import SwiGLUPacked
        self.Image_Encoder = build_ROI_backbone_model(model_name=model_name, num_classes=0)

        self.embed_convert = nn.Linear(self.Image_Encoder.embed_dim, embed_size) \
            if self.Text_Encoder.embed_dim != embed_size else nn.Identity()

    def forward(self, images):
        # Process image through Image_Encoder to get the embeddings
        Image_cls_embedding = self.Image_Encoder(images)  # CLS token output from ViT [B,D]
        return self.embed_convert(Image_cls_embedding)


class CLSModel(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes=0):
        super(CLSModel, self).__init__()

        self.backbone = backbone
        self.num_classes = num_classes

        self.head = nn.Linear(embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

    def forward(self, images):
        # Process image through Image_Encoder to get the embeddings
        Image_cls_embedding = self.backbone(images)  # CLS token output from ViT [B,D]

        logit = self.head(Image_cls_embedding)
        return logit


def build_MTL_backbone_model(model_name, MTL_token_num=None, edge_size=224, convert_embed_dim=768,
                             pretrained_backbone=True, **kwargs):
    if model_name == "MTL_ViT":
        Tile_backbone = MTL_ViT_backbone(MTL_token_num=MTL_token_num, img_size=edge_size)
        if pretrained_backbone:
            # Transferlearning on Encoders
            ViT_backbone_weights = timm.create_model('vit_base_patch16_224', pretrained=True).state_dict()
            del ViT_backbone_weights['patch_embed.proj.weight']
            del ViT_backbone_weights['patch_embed.proj.bias']
            Tile_backbone.load_state_dict(ViT_backbone_weights, strict=False)

    elif model_name == "MTL_VPT":
        Tile_backbone = MTL_VPT_ViT_backbone(MTL_token_num=MTL_token_num, img_size=edge_size, **kwargs)
        if pretrained_backbone:
            # Transferlearning on Encoders
            ViT_backbone_weights = timm.create_model('vit_base_patch16_224', pretrained=True).state_dict()
            del ViT_backbone_weights['patch_embed.proj.weight']
            del ViT_backbone_weights['patch_embed.proj.bias']
            Tile_backbone.load_state_dict(ViT_backbone_weights, strict=False)

    else:
        Tile_backbone = build_ROI_backbone_model(num_classes=0, model_name=model_name, edge_size=edge_size,
                                                 pretrained_backbone=pretrained_backbone, **kwargs)

    default_ROI_feature_dim = (3, edge_size, edge_size)

    if hasattr(Tile_backbone, 'embed_dim'):
        # fixme ensure all the backbone module has this record
        embed_dim = Tile_backbone.embed_dim
    else:
        Tile_backbone = build_ROI_backbone_model(num_classes=convert_embed_dim, model_name=model_name,
                                                 edge_size=edge_size, pretrained_backbone=pretrained_backbone, **kwargs)
        embed_dim = convert_embed_dim

    return Tile_backbone, default_ROI_feature_dim, embed_dim


# ------------------- ROI MTL model -------------------
def build_ROI_task_model(
        model_name="vit", edge_size=224,
        MTL_heads_configs: List[int] = None,  # the list of multiple MTL task head dimension for each task
        latent_feature_dim=128,
        Froze_backbone=False,
        convert_embed_dim=768,
        Head_strategy=None,
        bin_df=None,
        pretrained_backbone=True):
    assert MTL_heads_configs is not None

    # here we need to set config limitation for certain models
    if model_name in ["MTL_ViT", "MTL_VPT"]:
        MTL_token_design = "Through_out"
    else:
        MTL_token_design = None

    if MTL_token_design == "Through_out" or MTL_token_design == "MIL_to":
        if Head_strategy == 'expression_bin':
            MTL_token_num = len(pd.unique(bin_df['bin']))
        else:
            MTL_token_num = len(MTL_heads_configs)
    else:
        MTL_token_num = 0

    Tile_backbone, default_ROI_feature_dim, embed_dim = build_MTL_backbone_model(model_name, MTL_token_num,
                                                                                 edge_size=edge_size,
                                                                                 convert_embed_dim=convert_embed_dim,
                                                                                 pretrained_backbone=pretrained_backbone)

    MTL_Model = MTL_Model_builder(
        Tile_backbone,
        MTL_module_name=None,
        MTL_token_design=MTL_token_design,
        MTL_heads_configs=MTL_heads_configs,
        Input_embedding_converter=None,
        embed_dim=embed_dim,
        MTL_feature_dim=latent_feature_dim,
        Froze_backbone=Froze_backbone,
        Head_strategy=Head_strategy,  # fixme Tim testing parts
        bin_df=bin_df)

    return MTL_Model


# ------------------- ROI CLS model -------------------
def build_CLS_with_backbone_model(model_name, num_classes=0, edge_size=224, pretrained_backbone=True):
    MTL_token_num = 1  # CLS token
    Tile_backbone, default_ROI_feature_dim, embed_dim = (
        build_MTL_backbone_model(model_name=model_name, MTL_token_num=MTL_token_num,
                                 edge_size=edge_size, pretrained_backbone=pretrained_backbone))

    model = CLSModel(Tile_backbone, embed_dim, num_classes)
    return model


def build_ROI_CLS_model(model_name, num_classes=0, edge_size=224, pretrained_backbone=True):
    # MTL and MLLM models need to use backbone to build CLS-finetuning model
    if ('MTL' in model_name.split('_')
            or 'MUSK' in model_name.split('_')
            or 'Musk' in model_name.split('_')
            or 'musk' in model_name.split('_')
            or 'CONCH' in model_name.split('_')
            or 'Conch' in model_name.split('_')
            or 'conch' in model_name.split('_')):
        # build ROI model with MTL framework with single task token (for future adapting models)
        model = build_CLS_with_backbone_model(model_name, num_classes, edge_size, pretrained_backbone)
        print('build_ROI_CLS_model with ', model_name, ' as backbone_model')
        return model

    else:
        # build the ROI CLS model with backbone directly
        try:
            model = build_ROI_backbone_model(num_classes=num_classes, edge_size=edge_size,
                                             model_name=model_name, pretrained_backbone=pretrained_backbone)
        except AssertionError:
            print('if the model is encoundtering num_classes=0 issue, its becauese the model should be only '
                  'used as encoder backbone, check build_ROI_CLS_model to use the build_CLS_with_backbone_model')
            raise AssertionError('the model is encoundtering num_classes=0')
        else:
            return model

if __name__ == "__main__":
    # cuda issue
    print("cuda availability:", torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ROI_feature_dim = 768
    '''
    # MTL_token_design='Through_out' or None
    model = build_ROI_task_model(
        model_name="MTL_VPT",
        latent_feature_dim=128,
        MTL_heads=[3,4],
    )
    '''
    model = build_ROI_backbone_model(model_name='ResNet18', pretrained_backbone=True, edge_size=224, num_classes=0)
    model = model.to(dev).half()

    # play data (to see that Transformer can do multiple length, therefore can do MTL easily)
    x = torch.randn(2, 3, 224, 224).to(dev).half()
    y = model(x)
    print(y)  # list of T elements, each is a tensor of [B, D_T], D_T is the task predication dimension
