"""
Training       Script  ver： Feb 9th 16:30
dataset structure: ImageNet
image folder dataset is used.
"""
from __future__ import print_function, division

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import sys
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# For convenience, import all path to sys
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir))
sys.path.append(str(this_file_dir.parent))
sys.path.append(str(this_file_dir.parent.parent))
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

from ModelBase.Get_ROI_model import build_ROI_CLS_model, build_promptmodel
from Utils.Offline_augmentation_dataset import AmbiguousImageFolderDataset
from Utils.data_augmentation import data_augmentation
from Utils.SoftCrossEntropyLoss import SoftlabelCrossEntropy
from Utils.Online_augmentations import get_online_augmentation
from Utils.visual_usage import visualize_check, check_SAA
from Utils.tools import setup_seed, del_file, FixStateDict
from Utils.schedulers import patch_scheduler, ratio_scheduler
from DownStream.ROI_finetune.TASK_NAME import TASK_NAME
from Utils.check_log_json import CLS_JSON_logger, calculate_summary

import argparse
import copy
import json
import time
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from torchsummary import summary


def safe_model_summary(model, input_size, device="cuda"):
    """
    Runs torchsummary.summary on a deep copy of the model without affecting the original model.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.
        input_size (tuple): The expected input size (excluding batch size).
        device (str or torch.device): The device to move the copied model to. Default is "cuda".

    Returns:
        None
    """
    # Ensure device is a torch.device object
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model_copy = None  # Initialize to prevent reference errors in `finally`

    try:
        # Make a deep copy of the model and move it to the specified device
        model_copy = copy.deepcopy(model).to(device)

        # Run summary without modifying the original model
        summary(model_copy, input_size=input_size)

    except Exception as e:
        print(f"Error during model summary: {e}")

    finally:
        # Remove all hooks from both the original and copied model
        for m in model.modules():  # Clear hooks from the original model
            if hasattr(m, "_forward_hooks"):
                m._forward_hooks.clear()
            if hasattr(m, "_backward_hooks"):
                m._backward_hooks.clear()

        if model_copy is not None:
            for m in model_copy.modules():  # Clear hooks from the copied model
                if hasattr(m, "_forward_hooks"):
                    m._forward_hooks.clear()
                if hasattr(m, "_backward_hooks"):
                    m._backward_hooks.clear()

            # Free the copied model and GPU memory
            del model_copy
            torch.cuda.empty_cache()

# Training Tools
def better_performance(temp_acc, temp_vac, best_acc, best_vac):  # determin which epoch have the best model

    if temp_vac >= best_vac and temp_acc >= best_acc:
        return True
    elif temp_vac > best_vac:
        return True
    else:
        return False

def report_best_training(best_epoch_idx, best_acc, best_vac, class_names, best_log_dic):
    assert best_log_dic is not None
    print('Best epoch idx: ', best_epoch_idx)
    print('Best epoch train Acc: {:4f}'.format(best_acc))
    print('Best epoch val Acc: {:4f}'.format(best_vac))
    for cls_idx in range(len(class_names)):
        precision, recall = calculate_summary(best_log_dic, class_names, cls_idx)


# Training Loop
def train_model(model, dataloaders, criterion, optimizer, class_names, dataset_sizes, confusing_training=False,
                Augmentation=None, fix_position_ratio_scheduler=None, puzzle_patch_size_scheduler=None, edge_size=384,
                model_name=None, num_epochs=25, intake_epochs=0, check_minibatch=100, scheduler=None, device=None,
                draw_path='../imagingresults', enable_attention_check=False, enable_visualize_check=False,
                enable_sam=False, writer=None):
    """
    Training iteration
    :param model: model object
    :param dataloaders: 2 dataloader(train and val) dict
    :param criterion: loss func obj
    :param optimizer: optimizer obj
    :param class_names: The name of classes for priting
    :param dataset_sizes: size of datasets
    :param confusing_training: offline augmentation status
    :param Augmentation: augmentation methods
    :param fix_position_ratio_scheduler: Online augmentation fix_position_ratio_scheduler
    :param puzzle_patch_size_scheduler: Online augmentation puzzle_patch_size_scheduler
    :param edge_size: image size for the input image
    :param model_name: model idx for the getting pre-setted model
    :param num_epochs: total training epochs
    :param intake_epochs: number of skip over epochs when choosing the best model
    :param check_minibatch: number of skip over minibatch in calculating the criteria's results etc.
    :param scheduler: scheduler is an LR scheduler object from torch.optim.lr_scheduler.
    :param device: cpu/gpu object
    :param draw_path: path folder for output pic
    :param enable_attention_check: use attention_check to show the pics of models' attention areas
    :param enable_visualize_check: use visualize_check to show the pics
    :param enable_sam: use SAM training strategy
    :param writer: attach the records to the tensorboard backend
    """

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    # for saving the best model state dict
    best_model_wts = copy.deepcopy(model.state_dict())  # deepcopy
    # initiate the empty json dict
    json_logger = CLS_JSON_logger(class_names, draw_path, model_name)

    # initial best performance
    best_acc = 0.0
    best_vac = 0.0
    temp_acc = 0.0
    temp_vac = 0.0
    best_epoch_idx = 1
    best_log_dic = None

    epoch_loss = 0.0  # initial value for loss-drive

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # record json log, initially empty
        json_logger.init_epoch(epoch=str(epoch + 1))

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Val']:  # alternatively train/val

            index = 0
            check_index = -1  # set a visualize check at the end of each epoch's train and val
            model_time = time.time()

            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # criteria, initially empty
            running_loss = 0.0
            log_running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:  # use different dataloder in different phase
                # Obtain data
                inputs = inputs.to(device)  # print('inputs[0]',type(inputs[0]))

                # Obtain correct label
                if confusing_training:  # use off-line augmented dataset in training instead of regular data
                    if phase == 'Train':
                        # in this case the labels is a list of [soft-label, long-int-encoded-cls-idx]
                        GT_long_labels = labels[1].to(device)  # long-int
                        # soft_labels, will be replaced if using online augmentation later
                        labels = labels[0].to(device)

                        if Augmentation is not None:
                            # cellmix controlled by schedulers
                            if fix_position_ratio_scheduler is not None and puzzle_patch_size_scheduler is not None:
                                # loss-drive
                                fix_position_ratio = fix_position_ratio_scheduler(epoch, epoch_loss)
                                puzzle_patch_size = puzzle_patch_size_scheduler(epoch, epoch_loss)

                                inputs, labels, GT_long_labels = Augmentation(inputs, GT_long_labels,
                                                                              fix_position_ratio,
                                                                              puzzle_patch_size)
                            # Counterpart augmentations
                            else:
                                inputs, labels, GT_long_labels = Augmentation(inputs, GT_long_labels)
                    else:  # in val its normal dataset!
                        # convert to soft-label encoding, which is a same output with/without augmentation
                        GT_long_labels = labels.to(device)  # long-int
                        labels = torch.eye(len(class_names)).to(device)[GT_long_labels, :]  # one-hot hard label

                else:
                    # NOTICE in CLS task the labels' type is long tensor([B])，not one-hot ([B,CLS])
                    labels = labels.to(device)

                    # Online Augmentations or no Augmentations, on device
                    if Augmentation is not None:
                        if phase == 'Train':
                            # cellmix controlled by schedulers
                            if fix_position_ratio_scheduler is not None and puzzle_patch_size_scheduler is not None:
                                # loss-drive
                                fix_position_ratio = fix_position_ratio_scheduler(epoch, epoch_loss)
                                puzzle_patch_size = puzzle_patch_size_scheduler(epoch, epoch_loss)

                                inputs, labels, GT_long_labels = Augmentation(inputs, labels,
                                                                              fix_position_ratio,
                                                                              puzzle_patch_size)
                            # Counterpart augmentations
                            else:
                                inputs, labels, GT_long_labels = Augmentation(inputs, labels)

                        else:  # Val, put Augmentations  act=False
                            inputs, labels, GT_long_labels = Augmentation(inputs, labels, act=False)
                    else:
                        GT_long_labels = labels  # store ori_label on CPU

                # zero the parameter gradients
                if not enable_sam:
                    optimizer.zero_grad()

                # forward
                with torch.amp.autocast("cuda"):  # automatic mix precision training
                    # track grad if only in train!
                    with torch.set_grad_enabled(phase == 'Train'):

                        outputs = model(inputs)  # pred outputs of confidence: [B,CLS]
                        _, preds = torch.max(outputs, 1)  # idx outputs: [B] each is a idx
                        loss = criterion(outputs, labels)  # cross entrphy of one-hot outputs: [B,CLS] and idx label [B]

                        # backward + optimize only if in training phase
                        if phase == 'Train':
                            if enable_sam:
                                loss.backward()
                                # first forward-backward pass
                                optimizer.first_step(zero_grad=True)

                                # second forward-backward pass
                                loss2 = criterion(model(inputs), labels)  # SAM need another model(inputs)
                                loss2.backward()  # make sure to do a full forward pass when using SAM
                                optimizer.second_step(zero_grad=True)
                            else:
                                loss.backward()
                                optimizer.step()

                # log criteria: update
                log_running_loss += loss.item()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.cpu() == GT_long_labels.cpu().data)

                # Compute precision and recall for each class.
                json_logger.update_confusion_matrix(preds, GT_long_labels)

                # attach the records to the tensorboard backend
                if writer is not None:
                    # ...log the running loss
                    writer.add_scalar(phase + ' minibatch loss', float(loss.item()),
                                      epoch * len(dataloaders[phase]) + index)
                    writer.add_scalar(phase + ' minibatch ACC',
                                      float(torch.sum(preds.cpu() == GT_long_labels.cpu().data) / inputs.size(0)),
                                      epoch * len(dataloaders[phase]) + index)

                # at a checking minibatch in the middle of epoch
                if index % check_minibatch == check_minibatch - 1:
                    model_time = time.time() - model_time

                    check_index = index // check_minibatch + 1

                    epoch_idx = epoch + 1
                    print('Epoch:', epoch_idx, '   ', phase, 'index of ' + str(check_minibatch) + ' minibatch:',
                          check_index, '     time used:', model_time)

                    print('minibatch AVG loss:', float(log_running_loss) / check_minibatch)

                    if enable_visualize_check:
                        visualize_check(inputs, GT_long_labels, model, class_names, num_images=-1,
                                        pic_name='Visual_' + phase + '_E_' + str(epoch_idx) + '_I_' + str(index + 1),
                                        draw_path=draw_path, writer=writer)

                    if enable_attention_check:
                        try:
                            check_SAA(inputs, GT_long_labels, model, model_name, edge_size, class_names, num_images=1,
                                      pic_name='GradCAM_' + phase + '_E_' + str(epoch_idx) + '_I_' + str(index + 1),
                                      draw_path=draw_path, writer=writer)
                        except:
                            print('model:', model_name, ' with edge_size', edge_size, 'is not supported yet')
                    else:
                        pass

                    model_time = time.time()
                    log_running_loss = 0.0

                index += 1

            if phase == 'Train':
                if scheduler is not None:  # lr scheduler: update
                    scheduler.step()

            # at the last of train/val in each epoch, if no check has been triggered
            if check_index == -1:
                epoch_idx = epoch + 1
                if enable_visualize_check:
                    visualize_check(inputs, GT_long_labels, model, class_names, num_images=-1,
                                    pic_name='Visual_' + phase + '_E_' + str(epoch_idx),
                                    draw_path=draw_path, writer=writer)

                if enable_attention_check:
                    try:
                        check_SAA(inputs, GT_long_labels, model, model_name, edge_size, class_names, num_images=1,
                                  pic_name='GradCAM_' + phase + '_E_' + str(epoch_idx),
                                  draw_path=draw_path, writer=writer)
                    except:
                        print('model:', model_name, ' with edge_size', edge_size, 'is not supported yet')
                else:
                    pass

            # log criteria: print
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100
            print('\nEpoch: {}  {} \nLoss: {:.4f}  Acc: {:.4f}'.format(epoch + 1, phase, epoch_loss, epoch_acc))

            if phase == 'Train' and fix_position_ratio_scheduler is not None \
                    and puzzle_patch_size_scheduler is not None:
                print('\nEpoch: {}, Fix_position_ratio: {}, Puzzle_patch_size: '
                      '{}'.format(epoch + 1, fix_position_ratio, puzzle_patch_size))

            # attach the records to the tensorboard backend
            if writer is not None:
                # ...log the running loss
                writer.add_scalar(phase + ' loss', float(epoch_loss), epoch + 1)
                writer.add_scalar(phase + ' ACC', float(epoch_acc), epoch + 1)

            # calculating the confusion matrix for phase
            log_dict = json_logger.update_epoch_phase(epoch=str(epoch + 1), phase=phase, tensorboard_writer=writer)

            if phase == 'Val':
                temp_vac = epoch_acc
            else:
                temp_acc = epoch_acc  # not useful actually

            # deep copy the model
            if phase == 'Val' and better_performance(temp_acc, temp_vac, best_acc, best_vac) and epoch >= intake_epochs:
                # what is better? we now use the wildly used method only
                best_epoch_idx = epoch + 1
                best_acc = temp_acc
                best_vac = temp_vac
                best_model_wts = copy.deepcopy(model.state_dict())
                best_log_dic = log_dict

            print('\n')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    report_best_training(best_epoch_idx, best_acc, best_vac, class_names, best_log_dic)

    # attach the records to the tensorboard backend
    if writer is not None:
        writer.close()

    # load best model weights as final model training result
    model.load_state_dict(best_model_wts)
    # save json_log  indent=2 for better view
    json_logger.dump_json()

    return model


def main(args):
    if args.paint:
        # use Agg kernel, not painting in the front-desk
        import matplotlib
        matplotlib.use('Agg')

    enable_notify = args.enable_notify  # True
    enable_tensorboard = args.enable_tensorboard  # True
    enable_attention_check = args.enable_attention_check  # False   'CAM' 'SAA'
    enable_visualize_check = args.enable_visualize_check  # False

    enable_sam = args.enable_sam  # False

    data_augmentation_mode = args.data_augmentation_mode  # 0

    linearprobing = args.linearprobing  # False

    Pre_Trained_model_path = args.Pre_Trained_model_path  # None
    Prompt_state_path = args.Prompt_state_path  # None

    # Prompt
    PromptTuning = args.PromptTuning  # None  "Deep" / "Shallow"
    Prompt_Token_num = args.Prompt_Token_num  # 20
    PromptUnFreeze = args.PromptUnFreeze  # False

    gpu_idx = args.gpu_idx  # GPU idx start with0, -1 to use multipel GPU

    # pretrained_backbone
    pretrained_backbone = False if args.backbone_PT_off else True

    # classification required number of your dataset
    num_classes = args.num_classes  # default 0 for auto-fit
    # image size for the input image
    edge_size = args.edge_size  # 224 384 1000

    # batch info
    batch_size = args.batch_size  # 8
    num_workers = args.num_workers  # main training num_workers 4

    num_epochs = args.num_epochs  # 50
    intake_epochs = args.intake_epochs  # 0
    check_minibatch = args.check_minibatch if args.check_minibatch is not None else 512 // batch_size

    lr = args.lr  # 0.000007
    lrf = args.lrf  # 0.0

    opt_name = args.opt_name  # 'Adam'

    # model info
    # Fixme we removed args.task_name, now we use run name as wapper folder
    model_name = args.model_name  # the model we are going to use.
    tag = '' if args.tag is None else "_" + args.tag
    run_name = 'CLS_' + model_name + tag

    # PATH info
    runs_path = args.runs_path  # root path of saving the multiple experiments runs
    save_model_path = args.save_model_path or args.runs_path  # root path to saving models, if none, will go to draw root
    data_path = args.data_path  # path to a dataset

    # we use the run name as a warp folder for both Train and Test
    os.makedirs(os.path.join(runs_path), exist_ok=True)
    os.makedirs(os.path.join(save_model_path), exist_ok=True)
    os.makedirs(os.path.join(runs_path, run_name), exist_ok=True)
    os.makedirs(os.path.join(save_model_path, run_name), exist_ok=True)
    runs_path = os.path.join(runs_path, run_name, 'Train')
    save_model_path = os.path.join(save_model_path, run_name, run_name + '.pth')

    # fixme flush the output runs folder, NOTICE this may be DANGEROUS
    if os.path.exists(runs_path):
        del_file(runs_path)
    else:
        os.makedirs(runs_path)

    # Train Augmentation
    augmentation_name = args.augmentation_name  # None
    confusing_training = args.confusing_training  # False

    # Data Augmentation
    data_transforms = data_augmentation(data_augmentation_mode, edge_size=edge_size)

    # locate dataset
    if os.path.exists(os.path.join(data_path, 'train')):
        train_data_path = os.path.join(data_path, 'train')
    elif os.path.exists(os.path.join(data_path, 'Train')):
        train_data_path = os.path.join(data_path, 'Train')
    else:
        raise ValueError('train_data_path not available')
    if os.path.exists(os.path.join(data_path, 'val')):
        val_data_path = os.path.join(data_path, 'val')
    elif os.path.exists(os.path.join(data_path, 'Val')):
        val_data_path = os.path.join(data_path, 'Val')
    else:
        raise ValueError('val_data_path not available')

    if confusing_training:
        datasets = {'Train': AmbiguousImageFolderDataset(train_data_path, data_transforms['Train']),
                    'Val': torchvision.datasets.ImageFolder(val_data_path, data_transforms['Val'])}
    else:
        datasets = {'Train': torchvision.datasets.ImageFolder(train_data_path, data_transforms['Train']),
                   'Val': torchvision.datasets.ImageFolder(val_data_path, data_transforms['Val'])}

    dataset_sizes = {x: len(datasets[x]) for x in ['Train', 'Val']}  # size of each dataset

    dataloaders = {'Train': torch.utils.data.DataLoader(datasets['Train'], batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers, drop_last=True),
                   # colab suggest 2 workers
                   'Val': torch.utils.data.DataLoader(datasets['Val'], batch_size=batch_size, shuffle=False,
                                                      num_workers=num_workers // 4 + 1, drop_last=True)}
    # we decide class name by validate set
    class_names = [d.name for d in os.scandir(val_data_path) if d.is_dir()]
    class_names.sort()
    if num_classes == 0:
        print("class_names:", class_names)
        num_classes = len(class_names)
    else:
        if len(class_names) == num_classes:
            print("class_names:", class_names)
        else:
            print('classification number of the model mismatch the dataset requirement of:', len(class_names))
            return -1

    # use notifyemail to send the record to somewhere
    if enable_notify:
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='tum9598@163.com', mail_pass='xxx',
                       default_reciving_list=['tum9598@163.com'],  # change here if u want to use notify
                       log_root_path='log', max_log_cnt=5)

        if enable_tensorboard:
            notify.add_text('update to the tensorboard')
        else:
            notify.add_text('not update to the tensorboard')

        notify.add_text('  ')

        notify.add_text('model name ' + str(model_name))
        notify.add_text('  ')

        notify.add_text('GPU idx: ' + str(gpu_idx))
        notify.add_text('  ')

        notify.add_text('cls number ' + str(num_classes))
        notify.add_text('class_names =' + str(class_names))
        notify.add_text('edge size ' + str(edge_size))
        notify.add_text('  ')
        notify.add_text('batch_size ' + str(batch_size))
        notify.add_text('num_epochs ' + str(num_epochs))
        notify.add_text('lr ' + str(lr))
        notify.add_text('opt_name ' + str(opt_name))
        notify.add_text('  ')
        notify.add_text('enable_sam ' + str(enable_sam))
        notify.add_text('augmentation_name ' + str(augmentation_name))
        notify.add_text('data_augmentation_mode ' + str(data_augmentation_mode))

        notify.send_log()

    print("*********************************{}*************************************".format('tasks_to_run'))
    print(args)

    # start tensorboard backend
    if enable_tensorboard:
        writer = SummaryWriter(runs_path)
    else:
        writer = None
    # if u run locally
    # nohup tensorboard --logdir=/home/MSHT/runs --host=0.0.0.0 --port=7777 &
    # tensorboard --logdir=/home/ZTY/runs --host=0.0.0.0 --port=7777

    if gpu_idx == -1:  # use all cards
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            gpu_use = gpu_idx
        else:
            print('we dont have more GPU idx here, try to use gpu_idx=0')
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # tasks_to_run k for: only card idx k is sighted for this code
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")
                gpu_use = 'cpu'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_idx = "cuda:" + str(gpu_idx).strip()
        gpu_use = gpu_idx
        device = torch.device(device_idx if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    # else:
    #     # Decide which device we want to run on
    #     try:
    #         # tasks_to_run k for: only card idx k is sighted for this code
    #         os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    #         gpu_use = gpu_idx
    #     except:
    #         print('we dont have that GPU idx here, try to use gpu_idx=0')
    #         try:
    #             # tasks_to_run 0 for: only card idx 0 is sighted for this code
    #             os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #             gpu_use = 0
    #         except:
    #             print("GPU distributing ERRO occur use CPU instead")
    #             gpu_use = 'cpu'

    # device enviorment
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # get model
    if PromptTuning is not None:
        print('PromptTuning of ', model_name)
        print('Prompt VPT type:', PromptTuning)

        # initialize the model backbone:
        if Pre_Trained_model_path is None or Pre_Trained_model_path == 'timm':
            base_state_dict = 'timm'
            print('backbone base_state_dict of timm')
        elif Pre_Trained_model_path is not None and os.path.exists(Pre_Trained_model_path):
            print('backbone base_state_dict at: ', Pre_Trained_model_path)
            base_state_dict = torch.load(Pre_Trained_model_path)
        else:
            print('invalid Pre_Trained_model_path for prompting at: ', Pre_Trained_model_path)
            raise

        # put the additional prompt tokens to model:
        if Prompt_state_path is None:
            prompt_state_dict = None
            print('prompting with empty prompt_state:  prompt_state of None')
        elif Prompt_state_path is not None and os.path.exists(Prompt_state_path):
            print('prompting with prompt_state at: ', Prompt_state_path)
            prompt_state_dict = torch.load(Prompt_state_path)
        else:
            print('invalid prompt_state_dict for prompting, path at:', Prompt_state_path)
            raise

        model = build_promptmodel(num_classes, edge_size, model_name, Prompt_Token_num=Prompt_Token_num,
                                  VPT_type=PromptTuning, prompt_state_dict=prompt_state_dict,
                                  base_state_dict=base_state_dict)
        # Use FineTuning with prompt tokens (when PromptUnFreeze==True)
        if PromptUnFreeze:
            model.UnFreeze()
            print('prompt tuning with all parameaters un-freezed')

    else:
        # get model: randomly initiate model, except the backbone CNN(when pretrained_backbone is True)
        model = build_ROI_CLS_model(model_name=model_name, num_classes=num_classes, edge_size=edge_size,
                                    pretrained_backbone=pretrained_backbone)

        # Manually get the model pretrained on the Imagenet1000
        if Pre_Trained_model_path is not None:
            if os.path.exists(Pre_Trained_model_path):
                state_dict = FixStateDict(torch.load(Pre_Trained_model_path), remove_key_head='head')
                model.load_state_dict(state_dict, False)
                print('Specified backbone model weight loaded:', Pre_Trained_model_path)
            else:
                print('Specified Pre_Trained_model_path:' + Pre_Trained_model_path, ' is NOT avaliable!!!!\n')
                raise
        else:
            print('building model (no-prompt) with pretrained_backbone status:', pretrained_backbone)
            if pretrained_backbone is True:
                print('timm loaded')

        if linearprobing:
            # Only tuning the last FC layer for CLS task
            module_all = 0
            for child in model.children():  # find all nn.modules
                module_all += 1

            for param in model.parameters():  # freeze all parameters
                param.requires_grad = False

            for module_idx, child in enumerate(model.children()):
                if module_idx == module_all:  # Unfreeze the parameters of the last FC layer
                    for param in child.parameters():
                        param.requires_grad = True

    print('GPU:', gpu_use)

    if gpu_use == -1:
        model = nn.DataParallel(model)

    model.to(device)

    safe_model_summary(model, input_size=(3, edge_size, edge_size), device=device)
    print("model :", model_name)

    # Augmentation
    Augmentation = get_online_augmentation(augmentation_name, p=0.5, class_num=num_classes,
                                           batch_size=batch_size, edge_size=edge_size, device=device)

    if augmentation_name != 'CellMix-Split' and augmentation_name != 'CellMix-Group' \
            and augmentation_name != 'CellMix-Random':
        puzzle_patch_size_scheduler = None
        fix_position_ratio_scheduler = None

    else:
        # tasks_to_run puzzle_patch_size and fix_position_ratio schedulers
        puzzle_patch_size_scheduler = patch_scheduler(total_epoches=num_epochs,
                                                      warmup_epochs=0,
                                                      edge_size=edge_size,
                                                      basic_patch=16,
                                                      strategy=args.patch_strategy,  # 'random', 'linear' or 'loop'
                                                      loop_round_epoch=args.loop_round_epoch
                                                      if args.patch_strategy == 'loop' or
                                                         args.patch_strategy == 'loss_back' or
                                                         args.patch_strategy == 'loss_hold' else 1,
                                                      fix_patch_size=args.fix_patch_size,  # 16,32,48,64,96,128,192
                                                      patch_size_jump=args.patch_size_jump)  # 'odd' or 'even'

        fix_position_ratio_scheduler = ratio_scheduler(total_epoches=num_epochs,
                                                       warmup_epochs=0,
                                                       basic_ratio=0.5,
                                                       strategy=args.ratio_strategy,  # 'linear'
                                                       fix_position_ratio=args.fix_position_ratio,
                                                       loop_round_epoch=args.loop_round_epoch)

    # Default cross entropy of one-hot outputs: [B,CLS] and idx label [B] long tensor
    # confusing_training or have online augmentation, we use the loss of SoftlabelCrossEntropy
    if confusing_training or Augmentation is not None:
        criterion = SoftlabelCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()

    if opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.005)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 15 0.1  default SGD StepLR scheduler
    elif opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = None
    else:
        print('no optimizer')
        raise

    if enable_sam:
        from Utils.sam import SAM

        if opt_name == 'SGD':
            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.8)
            scheduler = None
        elif opt_name == 'Adam':
            base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=0.01)
        else:
            print('no optimizer')
            raise

    if lrf > 0:  # use cosine learning rate schedule
        import math
        # cosine Scheduler by https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # train
    model_ft = train_model(model, dataloaders, criterion, optimizer, class_names, dataset_sizes,
                           fix_position_ratio_scheduler=fix_position_ratio_scheduler,
                           puzzle_patch_size_scheduler=puzzle_patch_size_scheduler,
                           confusing_training=confusing_training, Augmentation=Augmentation,
                           edge_size=edge_size, model_name=model_name, num_epochs=num_epochs,
                           intake_epochs=intake_epochs, check_minibatch=check_minibatch,
                           scheduler=scheduler, device=device, draw_path=runs_path,
                           enable_attention_check=enable_attention_check,
                           enable_visualize_check=enable_visualize_check,
                           enable_sam=enable_sam, writer=writer)
    # save model
    if gpu_use == -1:
        # if on multi-GPU, save model as a single GPU model by stripe 'module'
        if PromptTuning is None:
            torch.save(model_ft.module.state_dict(), save_model_path)
        else:
            if PromptUnFreeze:
                torch.save(model_ft.module.state_dict(), save_model_path)
            else:
                prompt_state_dict = model_ft.module.obtain_prompt()
                # fixme maybe bug at DP module.obtain_prompt, just model.obtain_prompt is enough
                torch.save(prompt_state_dict, save_model_path)

        print('model trained by multi-GPUs has its single GPU copy saved at ', save_model_path)

    else:
        if PromptTuning is None:
            torch.save(model_ft.state_dict(), save_model_path)
        else:
            if PromptUnFreeze:
                torch.save(model_ft.state_dict(), save_model_path)
            else:
                prompt_state_dict = model_ft.obtain_prompt()
                torch.save(prompt_state_dict, save_model_path)

        print('model trained by GPU (idx:' + str(gpu_use) + ') has been saved at ', save_model_path)

    # after training, generate model config
    try:
        task_spec_name = TASK_NAME[args.task_name]
        one_hot_table = {task_spec_name: {}}
        len_classes = len(class_names)
        onehot_list = np.eye(len_classes, dtype=int).tolist()
        for index, class_name in enumerate(class_names):
            one_hot_table[task_spec_name][class_name] = onehot_list[index]
        model_config = {
            'all_task_dict': {task_spec_name: 'float'},
            'dataset': args.data_path,
            'model_struct': args.model_name,
            'edge_size': args.edge_size,
            'task_name': task_spec_name,
            'one_hot_table': one_hot_table
        }
        save_config_path = os.path.join(os.path.dirname(save_model_path), 'model_config.yaml')
        with open(save_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f)
        print(f'saved model_config.yaml at {save_config_path}')
    except Exception as e:
        print(f'saving model config file occupied with error: {e}')


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Task Name
    parser.add_argument('--task_name', type=str, default=None, help='name of task')

    # Model Name or index
    parser.add_argument('--model_name', default='ViT', type=str, help='Model Name or index')
    # Model tag (for example k-fold)
    parser.add_argument("--tag", default=None, type=str, help="Model tag (for example 5-fold)")

    # backbone_PT_off  by default is false, in default tasks_to_run the backbone weight is required
    parser.add_argument('--backbone_PT_off', action='store_true', help='use a fresh backbone weight in training')

    # Environment parameters
    parser.add_argument('--gpu_idx', default=-1, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Path parameters
    parser.add_argument('--data_path', default='/data/MIL_Experiment/dataset/ROSE_CLS',
                        help='path to dataset')
    parser.add_argument('--save_model_path', default=None, help='path to save model state-dict')
    parser.add_argument('--runs_path', default='./runs',
                        help='path to draw and save tensorboard output')

    # Help tool parameters
    parser.add_argument('--paint', action='store_false', help='paint in front desk')  # matplotlib.use('Agg')
    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    # check tool parameters
    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')
    parser.add_argument('--enable_attention_check', action='store_true', help='check and save attention map')
    parser.add_argument('--enable_visualize_check', action='store_true', help='check and save pics')

    # Tuning tasks_to_run
    # PromptTuning
    parser.add_argument('--PromptTuning', default=None, type=str,
                        help='use Prompt Tuning strategy instead of Finetuning')
    # Prompt_Token_num
    parser.add_argument('--Prompt_Token_num', default=20, type=int, help='Prompt_Token_num')

    # PromptUnFreeze
    parser.add_argument('--PromptUnFreeze', action='store_true', help='prompt tuning with all parameters un-freezed')

    # linear-probing
    parser.add_argument('--linearprobing', action='store_true', help='use linear-probing tuning')

    # Finetuning a Pretrained model at PATH
    # '/home/MIL_Experiment/saved_models/Hybrid2_384_PreTrain_000.pth'
    parser.add_argument('--Pre_Trained_model_path', default=None, type=str,
                        help='Finetuning a trained model in this dataset')
    # Prompt_state_path
    parser.add_argument('--Prompt_state_path', default=None, type=str,
                        help='Prompt_state_path for prompt tokens')

    # Training status parameters
    # SAM
    parser.add_argument('--enable_sam', action='store_true', help='use SAM strategy in training')

    # Online augmentation_name
    parser.add_argument('--augmentation_name', default=None, type=str, help='Online augmentation name')
    # confusing_training
    parser.add_argument('--confusing_training', action='store_true',
                        help='use Offline generated ambiguous samples and soft-label')

    # CellMix ablation: loss_drive strategy
    parser.add_argument('--ratio_strategy', default=None, type=str, help='CellMix ratio scheduler strategy')
    parser.add_argument('--patch_strategy', default=None, type=str, help='CellMix patch scheduler strategy')
    parser.add_argument('--loop_round_epoch', default=4, type=int, help='CellMix loss_drive_threshold is designed to '
                                                                        'record the epoch bandwidth for epochs at '
                                                                        'the same patch size')

    # CellMix ablation: fix_patch_size  patch_size_jump
    parser.add_argument('--fix_position_ratio', default=0.5, type=float, help='CellMix ratio scheduler strategy')
    parser.add_argument('--fix_patch_size', default=None, type=int, help='CellMix ablation using fix_patch_size')
    parser.add_argument('--patch_size_jump', default=None, type=str, help='CellMix patch_size_jump strategy')

    # Dataset based parameters
    parser.add_argument('--num_classes', default=0, type=int, help='classification number, default 0 for auto-fit')
    parser.add_argument('--edge_size', default=224, type=int, help='edge size of input image')  # 224 256 384 1000
    # Dataset specific augmentations in dataloader
    parser.add_argument('--data_augmentation_mode', default=-1, type=int, help='data_augmentation_mode')

    # Training setting parameters
    parser.add_argument('--batch_size', default=8, type=int, help='Training batch_size default 8')
    parser.add_argument('--num_epochs', default=50, type=int, help='training epochs')
    parser.add_argument('--intake_epochs', default=0, type=int, help='only save model at epochs after intake_epochs')
    parser.add_argument('--lr', default=0.00001, type=float, help='learing rate')
    parser.add_argument('--lrf', type=float, default=0.0,
                        help='learning rate decay rate, default 0(not enabled), suggest 0.1 and lr=0.00005')
    parser.add_argument('--opt_name', default='Adam', type=str, help='optimizer name Adam or SGD')

    # check_minibatch for painting pics
    parser.add_argument('--check_minibatch', default=None, type=int, help='check batch_size')
    parser.add_argument('--num_workers', default=2, type=int, help='use CPU num_workers , default 2 for colab')

    return parser


if __name__ == '__main__':
    # tasks_to_run up the random seed
    setup_seed(42)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

