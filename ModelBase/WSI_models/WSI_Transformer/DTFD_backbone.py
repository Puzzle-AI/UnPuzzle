# BigModel import
from __future__ import annotations
import os
import sys
# from pathlib import Path

# # For convenience, import all path to sys
# this_file_dir = Path(__file__).resolve().parent
# sys.path.append(str(this_file_dir))
# sys.path.append(str(this_file_dir.parent))
# sys.path.append(str(this_file_dir.parent.parent))
# sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

# DTFD import
import torch
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
import pickle
import random
import torch.nn.functional as F
import numpy as np


# from ModelBase.WSI_models.WSI_Transformer.WSI_pos_embed import get_2d_sincos_pos_embed
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, add_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, slide_embed_dim] or [1+grid_size*grid_size, slide_embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_token > 0:
        pos_embed = np.concatenate([np.zeros([add_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    slide_embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


from sklearn.metrics import roc_auc_score, roc_curve
import torch
import numpy as np


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def eval_metric(oprob, label):
    auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    prob = oprob > threshold
    label = label > threshold

    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()

    accuracy = torch.mean((TP + TN) / (TP + TN + FP + FN + 1e-12))
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    specificity = torch.mean(TN / (TN + FP + 1e-12))
    F1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    return accuracy, precision, recall, specificity, F1, auc


# network.py
class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)

    def forward(self, x):  ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x)  ## K x L
        pred = self.classifier(afeat)  ## K x num_cls
        return afeat


class DTFDMIL(nn.Module):
    def __init__(self,
                 in_features: int,  # in_chn
                 feats_size: int,
                 num_classes=2,

                 slide_pos: bool = False,
                 slide_ngrids: int = 1000,
                 dropout: float = 0.25,

                 # mDim = 512,  # 我猜是feats_size
                 droprate=0,
                 droprate_2=0,
                 numLayer_Res=0,

                 numGroup=3,
                 distill_type='AFS'
                 ):

        super(DTFDMIL, self).__init__()

        self.in_features = in_features
        self.feats_size = feats_size
        self.num_classes = num_classes
        self.slide_pos = slide_pos
        self.slide_ngrids = slide_ngrids
        self.dropout = dropout
        self.droprate = droprate
        self.droprate_2 = droprate_2
        self.numLayer_Res = numLayer_Res
        self.numGroup = numGroup
        self.distill_type = distill_type

        self.encoder = nn.Sequential(nn.Linear(in_features, feats_size),
                                     nn.ReLU(),
                                     nn.Dropout(dropout))

        self.dimReduction = DimReduction(in_features,
                                         feats_size,
                                         numLayer_Res=self.numLayer_Res)

        self.classifier = Classifier_1fc(feats_size,
                                         self.num_classes,
                                         self.droprate)
        self.attention = Attention_Gated(feats_size)
        self.UClassifier = Attention_with_Classifier(L=feats_size,
                                                     num_cls=self.num_classes,
                                                     droprate=self.droprate_2)
        self.attCls = Attention_with_Classifier(L=self.feats_size,
                                                num_cls=self.num_classes,
                                                droprate=self.droprate_2)

        # Optional position embeddings, aligned with Slide_Transformer_blocks
        if self.slide_pos:
            num_patches = self.slide_ngrids ** 2
            self.register_buffer('pos_embed', torch.zeros(1, num_patches, self.feats_size), persistent=False)

        # Initialize weights
        self.initialize_weights()

    def coords_to_pos(self, coords):
        """
        Convert coordinates to positional indices, similar to Slide_Transformer_blocks.

        Arguments:
            coords: torch.Tensor - Coordinates of the patches, shape [B, N, 2]
        Returns:
            torch.Tensor - Positional indices, shape [B, N]
        """
        coords_ = torch.floor(coords / 256.0)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        return pos.long()

    # initialize (and freeze) pos_embed by sin-cos embedding
    def initialize_weights(self):

        if self.slide_pos:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.slide_ngrids,
                                                add_token=0)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, coords=None):

        device = x.device
        # Feature encoding
        x = self.encoder(x)  # Output shape: [batch_size, tile_num, feats_size]

        # Apply positional embeddings if enabled
        if self.slide_pos:
            if coords is None:
                raise ValueError("Coordinates must be provided for positional embeddings when slide_pos is enabled.")

            pos = self.coords_to_pos(coords)
            x = x + self.pos_embed[:, pos, :].squeeze(0)

        total_instance = x.size(1)
        instance_per_group = total_instance // self.numGroup

        numIter = x.size(0)

        for idx in range(numIter):
            tfeat_tensor = x[idx, :, :]

            feat_index = list(range(tfeat_tensor.shape[0]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), self.numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            slide_pseudo_feat = []

            for tindex in index_chunk_list:

                subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=torch.LongTensor(tindex).to(device))
                # tmidFeat = self.dimReduction(subFeat_tensor)
                # tmidFeat = self.encoder(subFeat_tensor)
                # print("tmidFeat:",tmidFeat.shape)
                # tAA = self.attention(tmidFeat).squeeze(0)
                tmidFeat = subFeat_tensor
                tAA = self.attention(tmidFeat).squeeze(0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                tPredict = self.classifier(tattFeat_tensor)  ### 1 x 2

                patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

                MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)  ##########################
                max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                af_inst_feat = tattFeat_tensor

                if self.distill_type == 'MaxMinS':
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif self.distill_type == 'MaxS':
                    slide_pseudo_feat.append(max_inst_feat)
                elif self.distill_type == 'AFS':
                    slide_pseudo_feat.append(af_inst_feat)

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

            ## optimization for the second tier
            gSlidePred = self.UClassifier(slide_pseudo_feat)

        return slide_pseudo_feat


if __name__ == "__main__":
    # cuda issue
    print('cuda avaliablity:', torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MTL_token_num = 5
    ROI_feature_dim = 1024
    sllide_embed_dim = 512

    # slide_task_model = build_WSI_task_model(
    #     MTL_token_design=None,
    #     model_name="gigapath",
    #     local_weight_path=None,
    #     ROI_feature_dim=ROI_feature_dim,
    #     latent_feature_dim=128,
    #     MTL_heads=[nn.Linear(128, 3), nn.Linear(128, 4)],
    # )

    model = DTFDMIL(
        in_features=ROI_feature_dim,
        feats_size=sllide_embed_dim,
        num_classes=MTL_token_num
    )

    model.to(dev)
    # play data (to see that Transformer can do multiple length, therefore can do MTL easily)

    x = torch.randn(1, 30, ROI_feature_dim).to(dev)
    coords = torch.randn(1, 30, 2).to(dev)

    y = model(x, coords=coords)
    print("Shape of return_features:", y.shape)

    print("Test sucessful!")

    loss = y[0].sum()
    loss.backward()

'''
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)        
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()

'''
