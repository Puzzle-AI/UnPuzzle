"""
MTL blocks    Script  ver： Feb 4th 12:30       MTL modules
"""
import os
import sys
import timm
import torch
import torch.nn as nn
import huggingface_hub
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np


class MTL_module_baseline(nn.Module):
    """
    this one project [B,D] -> [B,T*D]
    """

    def __init__(self, MTL_token_num, latent_feature_dim):
        super().__init__()
        self.MTL_token_num, self.latent_feature_dim = MTL_token_num, latent_feature_dim
        self.layer = nn.Linear(latent_feature_dim, MTL_token_num * latent_feature_dim)

    def forward(self, latent_features):
        MTL_tokens = self.layer(latent_features).view(-1, self.MTL_token_num, self.latent_feature_dim)
        return MTL_tokens


# TODO
class MTL_module_xxxx(nn.Module):
    """
    this one is the MTL module that project [B,D] -> [B,T,D] for task heads
    """

    def __init__(self, MTL_token_num, latent_feature_dim):
        super().__init__()
        self.MTL_token_num, self.latent_feature_dim = MTL_token_num, latent_feature_dim
        self.layer = nn.Linear(latent_feature_dim, MTL_token_num * latent_feature_dim)

    def forward(self, latent_features):
        MTL_tokens = self.layer(latent_features).view(-1, self.MTL_token_num, self.latent_feature_dim)
        return MTL_tokens


def MTL_module_block_builder(MTL_token_num, MTL_feature_dim, MTL_module_name=None, MTL_token_design=None):
    """
    this one is the MTL module builder, ensuring the features to be [B,T,D] for task heads
    
    [B,T,D] -> [B,T,D] for task heads
    [B,D] -> [B,T,D] for task heads
    
    """
    if MTL_module_name is not None:
        pass  # todo CSC building blocks MTL_module_xxxx
    else:
        if MTL_token_design == "Through_out" or MTL_token_design == "MIL_to":
            # [B,T,D] -> [B,T,D]
            MTL_module = nn.Identity()  # future design of MTL modules
        else:
            '''
            for the pure feature extracting backbone, they extract the features into [B,D] or [B,N,D]
            we project them from [B,D] or [B,N,D] to [B,D] and expand to [B,T*D] -> [B,T,D] for MTL
            '''
            MTL_module = MTL_module_baseline(MTL_token_num, MTL_feature_dim)

    return MTL_module


class MTL_heads_baseline(nn.Module):
    """
    Task head for all tasks

    this one project [B,T,D] -> output list [T, B, K] and the K is different for different T

    :param MTL_heads_configs: the list of multiple MTL task head dimension for each task
    :param MTL_feature_dim: the feature dim of MTL tokens

    this build the output to be a list size of [T, B, K]
        tensor k = MTL_tasks_pred[task_idx][batch_idx]

    """

    def __init__(self, MTL_heads_configs: List[int] = None, MTL_feature_dim=128):
        super().__init__()

        MTL_heads_list = [nn.Linear(MTL_feature_dim, config_dim) for config_dim in MTL_heads_configs]

        self.MTL_heads = nn.ModuleList(MTL_heads_list)

        self.MTL_head_initialization()

    def MTL_head_initialization(self):
        for head in self.MTL_heads:
            self._initialize_module(head)

    def _initialize_module(self, module):
        """
        Initialize weights for the given module if it's not Identity.
        """
        for m in module.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, MTL_tokens):
        # a list of T tasks, each element is batch output of task (size of K) and the K is different for different T
        MTL_tasks_pred = []

        for idx, head in enumerate(self.MTL_heads):
            task_pred = head(MTL_tokens[:, idx])
            if torch.isnan(task_pred).any() or torch.isinf(task_pred).any():
                raise ValueError("Detected NaN or Inf in MTL head output! "
                                 "pls check backbone model or MTL framework model")
            MTL_tasks_pred.append(task_pred)
        return MTL_tasks_pred


class MTL_binning_heads(nn.Module):
    """
    Task head for binned tasks

    this one project [B, bin_no, Dim] -> output list [bin_no, B, G]. G is the number of genes in each bin

    :param MTL_heads_configs: the list of multiple MTL task head dimension for each task
    :param MTL_feature_dim: the feature dim of MTL tokens

    this build the output to be a list size of [bin_no, B, G]
        genePredsInABin = MTL_tasks_pred[bin_idx][batch_idx]
    """

    def __init__(self, bin_df, MTL_feature_dim=128, MTL_token_num=None, activation=None):
        super().__init__()
        self.bin_df = bin_df
        assert MTL_token_num is not None

        MTL_heads_list = []
        for i in range(MTL_token_num):
            genesInBin = len(self.bin_df[self.bin_df['bin'] == i])
            MTL_heads_list.append(nn.Linear(MTL_feature_dim, genesInBin))

        self.MTL_heads = nn.ModuleList(MTL_heads_list)
        self.MTL_head_initialization()

        self.activation = activation
        if self.activation:
            exec(f'self.activ_func = nn.ReLU()')

    def MTL_head_initialization(self):
        for head in self.MTL_heads:
            self._initialize_module(head)

    def _initialize_module(self, module):
        """
        Initialize weights for the given module if it's not Identity.
        """
        for m in module.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, MTL_tokens):
        # a list of T tasks, each element is batch output of task (size of K) and the K is different for different T
        MTL_tasks_pred_binned = []

        for idx, head in enumerate(self.MTL_heads):
            task_pred = head(MTL_tokens[:, idx])
            if self.activation:
                activ_func = eval('self.activ_func')
                task_pred = activ_func(task_pred)

            if torch.isnan(task_pred).any() or torch.isinf(task_pred).any():
                raise ValueError("Detected NaN or Inf in MTL head output! "
                                 "pls check backbone model or MTL framework model")
            MTL_tasks_pred_binned.append(task_pred)

        # decompose binning to real task number  fixme Naman pls check this
        single_element_pred = []
        for tensor in MTL_tasks_pred_binned:
            for value in tensor.view(-1):  # Flatten the tensor and iterate over values
                single_element_pred.append(value.view(1, 1))  # Reshape each value to 1x1 tensor

        return single_element_pred


# making bin_df
def bin_MTL_tasks(WSI_task_description_csv, WSI_task_idx_or_name_list, MTL_token_bins_num, method='expression_bin'):
    """
    group MTL tasks to a smaller number of bins to reduce GPU load when the number of tasks is large
    """
    if method == 'expression_bin':
        mean_genes = WSI_task_description_csv[WSI_task_idx_or_name_list].mean().sort_values()
        bin_borders = np.linspace(0, len(mean_genes), MTL_token_bins_num + 1, dtype=int)
        bin_df = pd.DataFrame()
        bin_df['values'] = mean_genes
        bin_df['bin'] = -1
        bin_df.reset_index(inplace=True, names='gene')
        # fill up the bin column
        for i in range(len(bin_borders) - 1):
            bin_df.iloc[bin_borders[i]:bin_borders[i + 1], -1] = i

    else:
        raise NotImplementedError()
    return bin_df


def get_MTL_heads(Head_strategy=None, MTL_heads_configs=None, MTL_feature_dim=128, bin_df=None):
    """
    This one host different MTL heads design to reduce the feature dimnesion of MTL

    :param Head_strategy: Head binning strategy to reduce GPU memory load
                1. expression value
                2. pca (not yet implemented)
    :param MTL_heads_configs: the list of multiple MTL task head dimension for each task
    :param MTL_feature_dim: feature dim for MTL modeling
    :param bin_df: dataframe of genes with their respective bin numbers 
    
    """
    assert MTL_heads_configs is not None

    if Head_strategy == 'expression_bin':
        assert bin_df is not None
        MTL_token_num = len(pd.unique(bin_df['bin']))
        MTL_heads = MTL_binning_heads(bin_df, MTL_feature_dim, MTL_token_num, None)

    else:
        MTL_token_num = len(MTL_heads_configs)
        MTL_heads = MTL_heads_baseline(MTL_heads_configs, MTL_feature_dim)

    return MTL_token_num, MTL_heads


class MTL_Model_builder(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            MTL_module_name: str = None,
            MTL_token_design=None,
            MTL_heads_configs: List[int] = None,
            Head_strategy=None,
            bin_df=None,
            Input_embedding_converter: Optional[nn.Module] = None,
            embed_dim: int = 768,
            MTL_feature_dim: int = 128,
            Froze_backbone=False):
        """
        This builder take the MTL tokens or the features into a MTL module and stack with MTL downstream heads

            :param backbone: backbone model for WSI or ROI feature extraction

            :param MTL_module_name: MTL model name

            :param MTL_token_design: default 'Through_out' for putting the tokens in each transformer layer,
                                else 'MIL_to' means the MTL tokens are obtained through slide MIL
                                else 'latent' convert from the slide model output
            :param MTL_heads_configs: the list of multiple MTL task head dimension for each task
            
            :param Head_strategy: xxxx todo 
                1. xxx
                2. xxx
            :param bin_df:  todo tim
            
            :param Input_embedding_converter: sometimes we need to convert the input feature to the backbone model's input

            :param embed_dim: feature dim for backbone output

            :param MTL_feature_dim: feature dim for MTL modeling

            :param Froze_backbone:

        Output:
        this build the output to be a list size of [T, B, K]
        tensor k = MTL_tasks_pred[task_idx][batch_idx]
        """
        super().__init__()
        assert isinstance(backbone, nn.Module), "Backbone must be an instance of nn.Module."
        assert MTL_heads_configs is not None

        self.backbone = backbone  # the output feature is [B, slide_embed_dim]
        self.MTL_token_design = MTL_token_design

        self.Input_embedding_converter = Input_embedding_converter or nn.Identity()

        if MTL_feature_dim == embed_dim:
            self.MTL_embedding_converter = nn.Identity()
        else:
            self.MTL_embedding_converter = nn.Linear(embed_dim, MTL_feature_dim)

        self.MTL_token_num, self.MTL_heads = \
            get_MTL_heads(Head_strategy=Head_strategy,
                          MTL_heads_configs=MTL_heads_configs,
                          MTL_feature_dim=MTL_feature_dim,
                          bin_df=bin_df)

        assert self.MTL_token_num > 0, "MTL_heads cannot be empty."

        # MTL_token_design: default 'Through_out' for putting the tokens in the first layer like cls token
        # and then pass through all transformer layer,
        # else None, convert from the slide model output to MTL token
        if self.MTL_token_design == "Through_out":
            # Embedding the MTL tokens
            self.MTL_tokens = nn.Parameter(torch.zeros(1, self.MTL_token_num, embed_dim))
        elif self.MTL_token_design == "MIL_to":
            # the MTL tokens are obtained through MIL modeling
            self.MTL_tokens = None
        else:
            '''
            for the pure feature extracting backbone, they extract the features into [B,D] or [B,N,D]
            without MTL_module, we project them from [B,D] or [B,N,D] to [B,D] and expand to [B,T*D] for MTL
            '''
            self.MTL_tokens = None

        self.MTL_module_name = MTL_module_name
        self.MTL_module = MTL_module_block_builder(self.MTL_token_num,
                                                   MTL_feature_dim,
                                                   MTL_module_name=self.MTL_module_name,
                                                   MTL_token_design=self.MTL_token_design)

        # initialize the MTL framework modules
        self.MTL_initialization()

        if Froze_backbone:
            self.Freeze_backbone()

    def MTL_initialization(self):
        if isinstance(self.Input_embedding_converter, nn.Module):
            self._initialize_module(self.Input_embedding_converter)
        if isinstance(self.MTL_embedding_converter, nn.Module):
            self._initialize_module(self.MTL_embedding_converter)
        if self.MTL_tokens is not None:
            torch.nn.init.normal_(self.MTL_tokens, std=0.02)  # Initialize tokens
        if isinstance(self.MTL_module, nn.Module):
            self._initialize_module(self.MTL_module)

    def _initialize_module(self, module):
        """
        Initialize weights for the given module if it's not Identity.
        """
        for m in module.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def Freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def UnFreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, image_features, coords=None):
        """
        Forward pass for the MTL dataflow:
        due to the different backbone design with/without the MTL Tokens,
            MTL_token_design == "Through_out" should have [B, T+N, embed_dim] or [B, T, embed_dim] from backbone
                we need to put the MTL tokens into the backbone as they are hosted in this builder block
                we take MTL_tokens = slide_latent[:, : self.MTL_token_num]
            MTL_token_design == "MIL_to" use backbone to model the features into [B, T, embed_dim] from backbone
            MTL_token_design == None will use backbone extract features into [B, N, embed_dim] or [B, embed_dim]
                we take MTL_tokens to [B,embed_dim] and project to [B, T, embed_dim] with MTL module (set outside)
                        we convert [B, N, embed_dim] to [B,embed_dim] with # fixme temp: global average pooling
        Batch size: B
        MTL_token_num: T
        tile_num: N the number of patches/features/tokens

        :param image_features: Tensors, WSI: [B, N, feature_dim], or ROI: [B,C,H,W]
        :param coords: Tensors, WSI: [B, N, 2]: [Y,X], default is None

        :return: List of task predictions, where each element in the list corresponds to a task and has
                 shape [B, output_dim] (output_dim may vary depending on the task).
        """
        # for WSI [B, N, ROI_feature_dim] -> [B, N, default_ROI_feature_dim]
        # for ROI [B,C,H,W] still unchanged
        image_features = self.Input_embedding_converter(image_features)

        if self.MTL_token_design == "Through_out":
            # WSI: [B, N, default_ROI_feature_dim]+[B, T, default_ROI_feature_dim] -> [B, T+N, embed_dim] or [B, T, embed_dim]
            # ROI: [B,C,H,W]+[B, T, default_ROI_feature_dim] -> [B, T+N, embed_dim] or [B, T, embed_dim]
            backbone_latent = self.backbone(image_features, coords=coords, MTL_tokens=self.MTL_tokens)

            # take MTL_tokens: [B, T, embed_dim]
            MTL_tokens = backbone_latent[:, : self.MTL_token_num]

            # MTL module: [B,T,MTL_feature_dim]  -> [B, MTL_token_num, MTL_feature_dim]
            MTL_tokens = self.MTL_module(self.MTL_embedding_converter(MTL_tokens))

        elif self.MTL_token_design == "MIL_to":
            # WSI: [B, N, default_ROI_feature_dim] -> [B, T, embed_dim]  likely MIL methods
            # ROI: [B,C,H,W] -> [B, T, embed_dim]  certain feature reduction methods
            MTL_tokens = self.backbone(image_features, coords=coords)

            # MTL module: [B,T,MTL_feature_dim]  -> [B, MTL_token_num, MTL_feature_dim]
            MTL_tokens = self.MTL_module(self.MTL_embedding_converter(MTL_tokens))

        else:  # this design generate MTL tokens from latent feature
            # WSI: [B, N, default_ROI_feature_dim]
            # ROI: [B,C,H,W]
            # -> [B, N, embed_dim] or [B, embed_dim]
            if coords is None or not hasattr(self.backbone, 'slide_pos'):
                backbone_latent = self.backbone(image_features)
            else:
                backbone_latent = self.backbone(image_features, coords=coords)

            if self.MTL_module_name is None:  # by default we compress-to/take 1 token
                # [B, N, embed_dim] or [B, embed_dim] -> [B, embed_dim]
                MTL_tokens = backbone_latent.mean(dim=1) if len(backbone_latent.shape) == 3 \
                    else backbone_latent  # shape of [B,embed_dim], like taking the CLS tokens as feature
            else:
                MTL_tokens = backbone_latent  # shape of [B,N, embed_dim]

            # MTL_embedding_converter: ([B,T,embed_dim] or [B,embed_dim]) -> ([B,T,MTL_feature_dim] or [B,MTL_feature_dim])
            # MTL module: ([B,T,MTL_feature_dim] or [B,MTL_feature_dim])  -> [B, MTL_token_num, MTL_feature_dim]
            MTL_tokens = self.MTL_module(self.MTL_embedding_converter(MTL_tokens))

        # Predicate the Tasks with MTL task heads
        MTL_tasks_pred = self.MTL_heads(MTL_tokens)

        return MTL_tasks_pred


def WSI_MTL_state_fixer(state_dict, new_MTL_num=None):
    """
    :param state_dict: a loaded pytorch state dict

    :param new_MTL_num:

    """
    assert new_MTL_num is not None
    old_MTL_num = state_dict['MTL_token_num'] if 'MTL_token_num' in state_dict else 0
    old_CLS_num = 1 if 'cls_token' in state_dict else 0

    if old_MTL_num == new_MTL_num and old_CLS_num == 0:
        return state_dict
    elif old_MTL_num == 0 and old_CLS_num == new_MTL_num:
        if 'cls_token' in state_dict:
            state_dict["MTL_tokens"] = state_dict["cls_token"]
            del state_dict["cls_token"]
        return state_dict
    else:
        if 'cls_token' in state_dict:
            del state_dict["cls_token"]

        if 'pos_embed' in state_dict:
            old_pos_embed_tensor = state_dict['pos_embed']  # (1, old_MTL_num+old_CLS_num + num_patches, embed_dim)
            new_MTL_embed_tensor = torch.zeros(1, new_MTL_num, old_pos_embed_tensor.size(-1))
            # (1, new_MTL_num + num_patches, embed_dim)
            state_dict['pos_embed'] = torch.cat((new_MTL_embed_tensor,
                                                 old_pos_embed_tensor[:, old_MTL_num + old_CLS_num:, :]), dim=1)
        return state_dict
