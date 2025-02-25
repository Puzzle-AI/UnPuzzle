"""
Online Augmentations    Script  ver： Nov 25th 15:45
ref:
CutOut, Mixup, CutMix based on
https://blog.csdn.net/cp1314971/article/details/106612060
"""
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from scipy.special import perm
from torchvision.transforms import Resize
from torchvision.transforms import ToPILImage, ToTensor

from .visual_usage import patchify, unpatchify
from .fmix import sample_mask, FMixBase  # Fmix


# generate random bounding box
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# augmentation methods
class Cutout(object):
    def __init__(self, alpha=2, shuffle_p=1.0, class_num=2, batch_size=4, device='cpu'):
        """
        Cutout augmentation arXiv:1708.04552
        :param alpha: alpha
        :param shuffle_p: chance of trigger augmentation
        :param class_num: number of classification categories
        :param batch_size: batch_size of training
        :param device: CUDA or CPU
        """
        self.alpha = alpha
        self.class_num = class_num
        self.batch_size = batch_size
        self.p = shuffle_p
        self.device = torch.device(device)

    def __call__(self, inputs, labels, act=True):
        labels = torch.eye(self.class_num).to(self.device)[labels, :]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)  # duplicate inputs for ori inputs
        cutout_inputs = inputs.clone().to(self.device)  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio

        for i in range(self.batch_size):

            if np.random.randint(0, 101) > 100 * self.p or (not act):
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            lam = np.random.beta(self.alpha, self.alpha)
            bbx1, bby1, bbx2, bby2 = rand_bbox(ori_inputs.size(), lam)  # get random bbox

            cutout_inputs[i, :, bbx1:bbx2, bby1:bby2] = 0

            # update the ratio of (area of ori_image on new masked image) for soft-label
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (ori_inputs.size()[2] * ori_inputs.size()[3]))
            lam_list.append(lam)

        long_label = labels.argmax(dim=1)

        # NOTICE cutout use long label and ori_crossentropy instead of soft-label and soft-label_crossentropy
        return cutout_inputs, labels, long_label


class CutMix(object):
    def __init__(self, alpha=2, shuffle_p=1.0, class_num=2, batch_size=4, device='cpu'):
        """
        CutMix augmentation arXiv:1905.04899
        :param alpha: alpha
        :param shuffle_p: chance of trigger augmentation
        :param class_num: number of classification categories
        :param batch_size: batch_size of training
        :param device: CUDA or CPU
        """
        self.alpha = alpha
        self.class_num = class_num
        self.batch_size = batch_size

        # calibrate the trigger chance of p, new ratio is the change of operation occur in each batch
        self.p = shuffle_p * (perm(self.batch_size, self.batch_size)
                              / (perm(self.batch_size, self.batch_size) -
                                 perm(self.batch_size - 1, self.batch_size - 1)))
        self.device = torch.device(device)

    def __call__(self, inputs, labels, act=True):

        labels = torch.eye(self.class_num).to(self.device)[labels, :]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)  # duplicate inputs for ori inputs
        cutmix_inputs = inputs.clone().to(self.device)  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio
        indices = torch.randperm(self.batch_size, device=self.device)  # shuffle indices
        shuffled_inputs = inputs[indices].to(self.device)
        shuffled_labels = labels[indices].to(self.device)

        for i in range(self.batch_size):

            if np.random.randint(0, 101) > 100 * self.p or (not act):
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            lam = np.random.beta(self.alpha, self.alpha)
            bbx1, bby1, bbx2, bby2 = rand_bbox(ori_inputs.size(), lam)  # get random bbox

            cutmix_inputs[i, :, bbx1:bbx2, bby1:bby2] = \
                shuffled_inputs[i, :, bbx1:bbx2, bby1:bby2]

            # update the ratio of (area of ori_image on new image) for soft-label
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (ori_inputs.size()[2] * ori_inputs.size()[3]))
            lam_list.append(lam)
            labels[i] = labels[i] * lam + shuffled_labels[i] * (1 - lam)

        long_label = labels.argmax(dim=1)
        return cutmix_inputs, labels, long_label


class Mixup(object):
    def __init__(self, alpha=2, shuffle_p=1.0, class_num=2, batch_size=4, device='cpu'):
        """
        Mixup augmentation arXiv:1710.09412
        :param alpha: alpha
        :param shuffle_p: chance of trigger augmentation
        :param class_num: number of classification categories
        :param batch_size: batch_size of training
        :param device: CUDA or CPU
        """
        self.alpha = alpha
        self.class_num = class_num
        self.batch_size = batch_size
        # calibrate the trigger chance of p, new ratio is the change of operation occur in each batch
        self.p = shuffle_p * (perm(self.batch_size, self.batch_size)
                              / (perm(self.batch_size, self.batch_size) -
                                 perm(self.batch_size - 1, self.batch_size - 1)))
        self.device = torch.device(device)

    def __call__(self, inputs, labels, act=True):
        labels = torch.eye(self.class_num).to(self.device)[labels, :]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)  # duplicate inputs for ori inputs
        mixup_inputs = inputs.clone().to(self.device)  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio
        indices = torch.randperm(self.batch_size, device=self.device)  # shuffle indices
        shuffled_inputs = inputs[indices].to(self.device)
        shuffled_labels = labels[indices].to(self.device)

        for i in range(self.batch_size):
            if np.random.randint(0, 101) > 100 * self.p or (not act):
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            lam = np.random.beta(self.alpha, self.alpha)
            lam_list.append(lam)
            mixup_inputs[i] = ori_inputs[i] * lam + shuffled_inputs[i] * (1 - lam)
            labels[i] = labels[i] * lam + shuffled_labels[i] * (1 - lam)

        long_label = labels.argmax(dim=1)
        return mixup_inputs, labels, long_label


class SaliencyMix(object):
    def __init__(self, alpha=1, shuffle_p=1.0, class_num=2, batch_size=4, device='cpu'):
        """
        SaliencyMix augmentation arXiv:2006.01791
        :param alpha: alpha
        :param shuffle_p: chance of trigger augmentation
        :param class_num: number of classification categories
        :param batch_size: batch_size of training
        :param device: CUDA or CPU
        """
        # ori batch_size=128
        self.alpha = alpha
        self.class_num = class_num
        self.batch_size = batch_size
        # calibrate the trigger chance of p, new ratio is the change of operation occur in each batch
        self.p = shuffle_p
        self.device = torch.device(device)

    def __call__(self, inputs, labels, act=True):
        labels = torch.eye(self.class_num).to(self.device)[labels, :]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)  # duplicate inputs for ori inputs
        saliencymix_inputs = inputs.clone().to(self.device)  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio
        indices = torch.randperm(self.batch_size, device=self.device)  # shuffle indices
        shuffled_inputs = inputs[indices].to(self.device)
        shuffled_labels = labels[indices].to(self.device)

        for i in range(self.batch_size):
            if np.random.randint(0, 101) > 100 * self.p or (not act) or self.alpha <= 0:
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            lam = np.random.beta(self.alpha, self.alpha)
            bbx1, bby1, bbx2, bby2 = saliency_bbox(shuffled_inputs[i], lam)  # get random bbox

            saliencymix_inputs[i, :, bbx1:bbx2, bby1:bby2] = \
                shuffled_inputs[i, :, bbx1:bbx2, bby1:bby2]

            # update the ratio of (area of ori_image on new image) for soft-label
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (ori_inputs.size()[2] * ori_inputs.size()[3]))
            lam_list.append(lam)
            labels[i] = labels[i] * lam + shuffled_labels[i] * (1 - lam)

        long_label = labels.argmax(dim=1)
        return saliencymix_inputs, labels, long_label


class ResizeMix(object):
    def __init__(self, shuffle_p=1.0, class_num=2, batch_size=4, device='cpu'):
        """
        ResizeMix augmentation arXiv:2012.11101
        :param shuffle_p: chance of trigger augmentation
        :param class_num: number of classification categories
        :param batch_size: batch_size of training
        :param device: CUDA or CPU
        """
        # ori batch_size=512
        self.class_num = class_num
        self.batch_size = batch_size
        # calibrate the trigger chance of p, new ratio is the change of operation occur in each batch
        self.p = shuffle_p
        self.device = torch.device(device)

    def __call__(self, inputs, labels, alpha=0.1, beta=0.8, act=True):
        labels = torch.eye(self.class_num).to(self.device)[labels, :]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)  # duplicate inputs for ori inputs
        resizemix_inputs = inputs.clone().to(self.device)  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio
        indices = torch.randperm(self.batch_size, device=self.device)  # shuffle indices
        shuffled_inputs = inputs[indices].to(self.device)
        shuffled_labels = labels[indices].to(self.device)

        for i in range(self.batch_size):
            if np.random.randint(0, 101) > 100 * self.p or (not act):
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            lam = np.random.uniform(alpha, beta)
            # lam = 1 - lam
            bbx1, bby1, bbx2, bby2 = rand_bbox(ori_inputs.size(), lam)  # get random bbox

            # resizer by torchvision
            torch_resize = Resize([bbx2 - bbx1, bby2 - bby1])

            # Tensor -> PIL -> resize -> Tensor
            re_pil_image = torch_resize(ToPILImage()(shuffled_inputs[i]))
            resizemix_inputs[i, :, bbx1:bbx2, bby1:bby2] = ToTensor()(re_pil_image)

            # update the ratio of (area of ori_image on new image) for soft-label
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (ori_inputs.size()[2] * ori_inputs.size()[3]))
            lam_list.append(lam)
            labels[i] = labels[i] * lam + shuffled_labels[i] * (1 - lam)

        long_label = labels.argmax(dim=1)
        return resizemix_inputs, labels, long_label


class FMix(FMixBase):

    def __init__(self, shuffle_p=1.0, class_num=2, batch_size=4, decay_power=3, alpha=1, size=(32, 32),
                 max_soft=0.0, reformulate=False, device='cpu'):
        """
        FMix augmentation arXiv:2002.12047
        :param shuffle_p: chance of trigger augmentation
        :param class_num: number of classification categories
        :param batch_size: batch_size of training

        :param decay_power: decay_power
        :param alpha: alpha
        :param size: size of patch
        :param max_soft: max_soft
        :param reformulate: reformulate

        :param device: CUDA or CPU
        """
        # ori batch_size=128
        super().__init__(decay_power, alpha, size, max_soft, reformulate)
        self.class_num = class_num
        self.batch_size = batch_size
        self.p = shuffle_p
        self.device = torch.device(device)

    def __call__(self, inputs, labels, alpha=1, act=True):
        # Sample mask and generate random permutation
        lam, mask = sample_mask(self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate)
        mask = torch.from_numpy(mask).float().to(self.device)

        labels = torch.eye(self.class_num).to(self.device)[labels, :]  # one-hot hard label
        ori_inputs = inputs.clone().to(self.device)
        fmix_inputs = inputs.clone().to(self.device)  # duplicate inputs for outputs
        lam_list = []  # a list to record operating ratio
        indices = torch.randperm(self.batch_size, device=self.device)  # shuffle indices
        shuffled_inputs = inputs[indices].to(self.device)
        shuffled_labels = labels[indices].to(self.device)

        for i in range(self.batch_size):
            if np.random.randint(0, 101) > 100 * self.p or (not act):
                # trigger the augmentation operation
                lam_list.append(-1)
                continue

            x1 = mask * ori_inputs[i]
            x2 = (1 - mask) * shuffled_inputs[i]
            fmix_inputs[i] = x1 + x2

            lam_list.append(lam)
            labels[i] = labels[i] * lam + shuffled_labels[i] * (1 - lam)

        long_label = labels.argmax(dim=1)
        # print('lam:', lam)
        return fmix_inputs, labels, long_label


# CellMix
class CellMix(object):
    def __init__(self, shuffle_p=1.0, class_num=2, strategy='In-place', group_shuffle_size=-1, device='cpu'):
        """
        CellMix augmentation arXiv:2301.11513
        :param shuffle_p: chance of trigger augmentation
        :param class_num: number of classification categories
        :param strategy: 'In-place' or 'Random' to shuffle the relation patches within the batch
        :param group_shuffle_size: the size of shuffling group in the batch, -1 to all
        :param device: CUDA or CPU
        """
        self.p = shuffle_p
        self.CLS = class_num  # classification category number of the task
        self.device = device
        self.strategy = strategy  # 'In-place' or 'Random'
        self.group_shuffle_size = group_shuffle_size  # -1 for whole batch

    def __call__(self, inputs, labels, fix_position_ratio=0.5, puzzle_patch_size=32, act=True):
        """
        Fix-position in-place shuffling
        Perform cross-sample random selection to fix some patches in each image of the batch
        After selection, the fixed patches are reserved, the rest patches are batch wise
        in-place shuffled and then regrouped with the fixed patches.
        cross-sample selection is done by argsort random noise in dim 1 and apply to all image within the batch.
        in-place batch-wise shuffle operation is done by argsort random noise in dim 0.
        grouped-in-place batch-wise shuffle operation is done by argsort random noise in the batch dimension

        :param inputs: input image tensor, size of [B, 3, H, W],
        :param labels: long-int encoded cls idxes for a batch [B], value of CLS idx
        :param fix_position_ratio:  float ratio of the least remaining part of patches
        :param puzzle_patch_size:  int patch size of shuffle
        :param act: set to be False to force not triggering CellMix in validation, set to True to trigger by chance p

        output: x, soft_label, long_label
        x : [B, 3, H, W] re-grouped image after cellmix augmentation
        soft_label : [B, CLS], soft-label of the class distribution
        long_label : [B] hard long-label for general discribe
        """
        if np.random.randint(0, 101) > 100 * self.p or (not act):
            soft_label = torch.eye(self.CLS).to(self.device)[labels, :]  # one-hot hard label
            return inputs, soft_label, labels

        # Break img into puzzle patches with the size of puzzle_patch_size  [B, num_patches, D]
        inputs = patchify(inputs, puzzle_patch_size)
        B, num_patches, D = inputs.shape

        # generate the persudo-mask: in cls dim only the k dim is
        mask = torch.zeros([B, num_patches, self.CLS], device=inputs.device, requires_grad=False)  # no grad
        # mask of patches: (B, num_patches, cls)  (cls)=[0,mask_area,0,....]

        # transform to persudo-mask
        B_idx = range(B)
        mask[B_idx, :, labels] = 1

        # num of fix_position puzzle patches
        len_fix_position = int(num_patches * fix_position_ratio)

        # create a noise tensor to prepare shuffle idx of puzzle patches
        noise = torch.rand(1, num_patches, device=self.device)
        noise = torch.repeat_interleave(noise, repeats=B, dim=0)

        # based on the batch sequence's shape, the noise tensor get a series idx matrix by sort
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # sort the idx matrix again, we can obtain the original location idx matrix before assignment
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_fix = ids_shuffle[:, :len_fix_position]  # [B,num_patches] -> [B,fix_patches]
        ids_puzzle = ids_shuffle[:, len_fix_position:]  # [B,num_patches] -> [B,puzzle_patches]

        # set puzzle patch
        # ids_?.unsqueeze(-1).repeat(1, 1, D)
        # [B,?_patches] -> [B,?_patches,1] (at each place with the idx of ori patch) -> [B,?_patches,D]

        # torch.gather to select patche groups x_fixed of [B,fix_patches,D] and x_puzzle of [B,puzzle_patches,D]
        x_fixed = torch.gather(inputs, dim=1, index=ids_fix.unsqueeze(-1).repeat(1, 1, D))
        x_puzzle = torch.gather(inputs, dim=1, index=ids_puzzle.unsqueeze(-1).repeat(1, 1, D))
        mask_fixed = torch.gather(mask, dim=1, index=ids_fix.unsqueeze(-1).repeat(1, 1, self.CLS))
        mask_puzzle = torch.gather(mask, dim=1, index=ids_puzzle.unsqueeze(-1).repeat(1, 1, self.CLS))

        if self.strategy == 'In-place' or self.strategy == 'Random':
            # the In-place strategy shuffles the relation patches within their location, among the batch index
            B, num_shuffle_patches, D = x_puzzle.shape

            # create a noise tensor to prepare shuffle idx of puzzle patches
            # [B, num_shuffle_patches] noise in [0, 1]
            noise = torch.rand(B, num_shuffle_patches, device=self.device)

            if self.group_shuffle_size == -1 or self.group_shuffle_size == B:  # CellMix-Split
                # sort the noise matrix, obtain a index assignment for shuffle,
                # shuffle dim 0 of entire noise (among all the batch)
                in_place_shuffle_indices = torch.argsort(noise, dim=0)

            else:  # CellMix-Group
                assert B > self.group_shuffle_size > 0 and B % self.group_shuffle_size == 0
                grouped_indices_list = []
                for group_idx in range(B // self.group_shuffle_size):
                    # group the noise by self.group_shuffle_size: [group_shuffle_size,N]
                    grouped_noise = noise[group_idx * self.group_shuffle_size:
                                          group_idx * self.group_shuffle_size + self.group_shuffle_size, :]
                    # sort each grouped_noise matrix, obtain a index assignment for shuffle,
                    # now the shuffle dim is 0 (among the batch within the group)
                    grouped_indices = torch.argsort(grouped_noise, dim=0)
                    # put grouped_noise matrix into the list
                    grouped_indices_list.append(grouped_indices + self.group_shuffle_size * group_idx)
                # stack(cat) the group indices(from list) back to tensor
                in_place_shuffle_indices = torch.cat(grouped_indices_list, dim=0)

            # torch.gather to achieve shuffle (taking all the idx base on a shuffled indices)
            x_puzzle = torch.gather(x_puzzle, dim=0, index=in_place_shuffle_indices.unsqueeze(-1).repeat(1, 1, D))
            mask_puzzle = torch.gather(mask_puzzle, dim=0,
                                       index=in_place_shuffle_indices.unsqueeze(-1).repeat(1, 1, self.CLS))
        else:
            print('not a valid CellMix strategy')

        # pack up all puzzle patches
        inputs = torch.cat([x_fixed, x_puzzle], dim=1)
        mask = torch.cat([mask_fixed, mask_puzzle], dim=1)

        # unshuffle to restore the fixed positions
        inputs = torch.gather(inputs, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        # torch.gather to generate restored binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, self.CLS))

        # CellMix random strategy randomly shuffle the image patches (after cellmix in-place shuffle)
        if self.strategy == 'Random':
            B, num_patches, D = inputs.shape
            # create a noise tensor to prepare shuffle idx of puzzle patches
            noise = torch.rand(B, num_patches, device=self.device)  # [num_patches,B] noise in [0, 1]
            # sort the noise matrix, obtain a index assignment for shuffle, now the shuffle dim is 1 (with the batch)
            all_shuffle_indices = torch.argsort(noise, dim=1)
            # ids_shuffle shape of [B,N], in N is idx
            # torch.gather to shuffle
            inputs = torch.gather(inputs, dim=1, index=all_shuffle_indices.unsqueeze(-1).repeat(1, 1, D))
            # no need to torch the mask, because its patch-wise shuffle within each sample
        else:  # when strategy == 'In-place'
            pass

        # unpatchify to obtain puzzle images and their mask
        inputs = unpatchify(inputs, puzzle_patch_size)  # restore to image size：B,3,224,224/ B,3,384,384

        # transform soft-mask to soft-label
        # calaculate a composed label with a conjugate design
        # [B, num_patches, CLS]->(B, CLS)
        soft_label = mask.sum(dim=1)  # (B, CLS)
        soft_label = soft_label / num_patches
        # long_label, as a data-augmentation requirement
        long_label = soft_label.argmax(dim=1)

        return inputs, soft_label, long_label


# ask func
def get_online_augmentation(augmentation_name, p=0.5, class_num=2, batch_size=4, edge_size=224, device='cpu'):
    """
    :param augmentation_name: name of data-augmentation method
    :param p: chance of triggering
    :param class_num: classification task num
    :param batch_size: batch size
    :param edge_size: edge size of img

    :param device: cpu or cuda

    其中augmentation_name, class_num, batch_size, edge_size必须提供

    return Augmentation
    """
    if augmentation_name == 'CellMix-Group':  # Pair the images and in-place swap the relation tokens between each pair
        Augmentation = CellMix(shuffle_p=p, class_num=class_num, strategy='In-place', group_shuffle_size=2,
                               device=device)

    elif augmentation_name == 'CellMix-Group4':
        # Group 4 images and in-place swap the relation tokens within this group of 4 images
        Augmentation = CellMix(shuffle_p=p, class_num=class_num, strategy='In-place', group_shuffle_size=4,
                               device=device)

    elif augmentation_name == 'CellMix-Split':  # In-place shuffle the relation tokens among the whole batch
        Augmentation = CellMix(shuffle_p=p, class_num=class_num, strategy='In-place', group_shuffle_size=-1,
                               device=device)

    elif augmentation_name == 'CellMix-Random':  # Pair the images, shuffle the relation tokens among the pair,
        # the location can be different instead of in-place
        Augmentation = CellMix(shuffle_p=p, class_num=class_num, strategy='Random', group_shuffle_size=2,
                               device=device)

    elif augmentation_name == 'CellMix-Random4':  # Group 4 images and shuffle the relation tokens among them,
        # the location can be different instead of in-place
        Augmentation = CellMix(shuffle_p=p, class_num=class_num, strategy='Random', group_shuffle_size=4,
                               device=device)

    elif augmentation_name == 'CellMix-Self':  # Shuffle the relation tokens within the same image
        Augmentation = CellMix(shuffle_p=p, class_num=class_num, strategy='Random', group_shuffle_size=1,
                               device=device)

    elif augmentation_name == 'CellMix-All':  # Shuffle the relation tokens among the whole batch
        Augmentation = CellMix(shuffle_p=p, class_num=class_num, strategy='Random', group_shuffle_size=-1,
                               device=device)

    elif augmentation_name == 'Cutout':
        Augmentation = Cutout(alpha=2, shuffle_p=p, class_num=class_num, batch_size=batch_size, device=device)

    elif augmentation_name == 'CutMix':
        Augmentation = CutMix(alpha=2, shuffle_p=p, class_num=class_num, batch_size=batch_size, device=device)

    elif augmentation_name == 'Mixup':
        Augmentation = Mixup(alpha=2, shuffle_p=p, class_num=class_num, batch_size=batch_size, device=device)

    elif augmentation_name == 'SaliencyMix':
        Augmentation = SaliencyMix(alpha=1, shuffle_p=p, class_num=class_num, batch_size=batch_size,
                                   device=device)  # alpha实际为源代码中beta

    elif augmentation_name == 'ResizeMix':
        Augmentation = ResizeMix(shuffle_p=p, class_num=class_num, batch_size=batch_size, device=device)

    elif augmentation_name == 'FMix':
        # FMIX p=1.0 beacuse the chance of trigger is determined inside its own design
        Augmentation = FMix(shuffle_p=1.0, class_num=class_num, batch_size=batch_size,
                            size=(edge_size, edge_size), device=device)

    elif augmentation_name == 'PuzzleMix':
        Augmentation = None
        # fixme: all related parts have been taken out separately
        # Augmentation = PuzzleMix(alpha=2, shuffle_p=p, class_num=class_num, batch_size=batch_size, device=device)
        # return Augmentation

    elif augmentation_name == 'CoMix':
        # TODO CoMix
        Augmentation = None

    elif augmentation_name == 'RandomMix':
        # TODO RandomMix
        Augmentation = None

    else:
        raise NotImplementedError

    if Augmentation is not None:
        print('\nUsing online-data augmentation methods of ', augmentation_name)
        # Print all attributes and configurations of the Augmentation instance
        print("Online-data augmentation configurations:")
        for attr, value in Augmentation.__dict__.items():
            print(f"  {attr}: {value}")
        print()
        return Augmentation
    else:
        print('\nno valid counterparts augmentation selected\n')
        return None


if __name__ == '__main__':
    '''
    Augmentation = get_online_augmentation('CellMix-Split', p=0.5, class_num=2)
    output, labels, GT_labels = Augmentation(x, label, fix_position_ratio=0.5, puzzle_patch_size=32, act=True)

    print(labels, GT_labels)

    '''

    x = torch.load("./temp-tensors/warwick.pt")
    # print(x.shape)
    label = torch.load("./temp-tensors/warwick_labels.pt")
    # print(label)

    # Augmentation = get_online_augmentation('ResizeMix', p=0.5, class_num=2)
    # output, labels, GT_labels = Augmentation(x, label, act=True)
    Augmentation = get_online_augmentation('CellMix-Group', p=1, class_num=2)
    output, labels, GT_labels = Augmentation(x, label, fix_position_ratio=0.5, puzzle_patch_size=32, act=True)

    print(labels, GT_labels)

    composed_img = ToPILImage()(output[0])
    composed_img.show()
    composed_img = ToPILImage()(output[1])
    composed_img.show()
    composed_img = ToPILImage()(output[2])
    composed_img.show()
    composed_img = ToPILImage()(output[3])
    composed_img.show()
