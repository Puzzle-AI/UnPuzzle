"""
ROI MTL Train       Script  ver: Feb 8th 01:00

flexible to multiple-tasks and missing labels

we have enabled multiple samples training by controlling the gradient in different task labels
we break the process of controlling when calculating the gradient, and
we use loss-aggregate technique to combine each sample for back-propagation
"""

import os
import sys
import json
import copy
import time
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from tensorboardX import SummaryWriter

# Go up 3 levels
ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_PATH))

from DataPipe.dataset_framework import Bulk_ROI_Dataset, MTL_collate_fn
from DownStream.MTL.task_settings import task_filter_auto
from ModelBase.Get_ROI_model import build_ROI_task_model
from Utils.MTL_plot_json import check_json_with_plot
from Utils.tools import setup_seed, del_file
from Utils.data_augmentation import data_augmentation


def ROI_MTL_train(model, dataloaders, dataset_sizes, criterions, optimizer, LR_scheduler,
                  loss_weight, task_dict, task_describe, num_epochs=25, accum_iter_train=1,
                  check_minibatch=5, intake_epochs=1, runs_path="./", writer=None, device=torch.device("cpu")):
    assert accum_iter_train >= 1  # accum batch should be >= 1
    since = time.time()

    task_num = len(task_dict)
    running_key_CLS = []
    running_key_REG = []
    running_cls_task_name_dict = {}
    for key in task_dict:
        if task_dict[key] == list:  # CLS sign
            running_cls_task_name_dict[key] = task_describe[key]
            running_key_CLS.append(key)
            running_key_REG.append("not applicable")
        else:  # REG sigh
            running_key_CLS.append("not applicable")
            running_key_REG.append(key)

    # log dict
    log_dict = {}

    # loss scaler
    Scaler = torch.cuda.amp.GradScaler()
    # for recording and epoch selection
    temp_best_epoch_loss = 10000000  # this one should be very big
    best_epoch_idx = 1

    epoch_average_sample_loss = 0.0  # api variable for future analysis

    for epoch in range(num_epochs):

        for phase in ["Train", "Val"]:

            if phase == "Train":
                accum_iter = accum_iter_train
                model.train()  # Set model to training mode
            else:
                accum_iter = 1
                model.eval()  # Set model to evaluate mode

            # Track information
            epoch_time = time.time()
            model_time = time.time()
            index = 0  # check minibatch index

            loss = 0.0  # for back propagation, assign as float, if called it will be tensor (Train)
            accum_average_sample_loss = 0.0  # for bag analysis

            failed_sample_count = 0  # default: the whole batch is available by dataloader

            phase_running_loss = [0.0 for _ in range(task_num)]  # phase-Track on loss, separate by tasks, not weighted
            temp_running_loss = [0.0 for _ in range(task_num)]  # temp record on loss, for iteration updating
            accum_running_loss = [0.0 for _ in range(task_num)]  # accumulated record on loss, for iteration updating

            running_measurement = [0.0 for _ in range(task_num)]  # track on measurement(ACC/L1 loss etc)
            temp_running_measurement = [0.0 for _ in range(task_num)]  # temp record for iteration updating
            accum_running_measurement = [0.0 for _ in range(task_num)]  # accumulated record for iteration updating

            epoch_preds = [[] for _ in range(task_num)]  # pred in current epoch
            epoch_labels = [[] for _ in range(task_num)]  # label in current epoch

            # missing count (track and record for validation check)
            minibatch_missing_task_sample_count = [0 for _ in range(task_num)]  # for accum iter and minibatch
            total_missing_task_sample_count = [0 for _ in range(task_num)]  # for phase

            # all matrix dict (initialize them with 0 values for each task)
            epoch_cls_task_matrix_dict = {}
            for cls_task_name in running_cls_task_name_dict:
                class_names = [key for key in running_cls_task_name_dict[cls_task_name]]
                # initiate the empty matrix_log_dict
                matrix_log_dict = {}
                for cls_idx in range(len(class_names)):
                    # only float type is allowed in json, set to float inside
                    matrix_log_dict[class_names[cls_idx]] = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
                epoch_cls_task_matrix_dict[cls_task_name] = matrix_log_dict

            # data loop for a epoch
            for data_iter_step, sample in enumerate(dataloaders[phase]):

                if sample is None:
                    # jump the batch if it cannot correct by WSI_collate_fn in dataloader
                    failed_sample_count += dataloaders[phase].batch_size
                    continue
                else:
                    # take data and task_description_list from sample
                    # image_features is a tensor of [B,N,D],
                    image_features = sample['image_features'].to(device)
                    # task_description_list [task, batch_size] batch-stacked tensors, element of long-int or float
                    task_description_list = sample['task_description_list']

                    # coords_yx is tensor of [B,N,2], yx
                    # coords_yx = sample['coords_yx']
                    # a list of names
                    # slide_id = sample['slide_id']

                # count failed samples in dataloader (should be 0, normally)
                # default B - B = 0, we don't have the last batch issue 'drop last batch in training code'
                failed_sample_count += dataloaders[phase].batch_size - len(task_description_list[0])  # batch size

                # Tracking the loss for loss-drive bag settings update and also recordings, initialize with 0.0
                running_loss = 0.0  # all tasks loss over a batch

                # zero the parameter gradients if its accumulated enough wrt. accum_iter
                if data_iter_step % accum_iter == 0:
                    optimizer.zero_grad()

                # fixme should have phase == 'Train', but val is too big for gpu
                with torch.amp.autocast("cuda"):
                    y = model(image_features)

                    # we calculate the all-batch loss for each task, and do backward
                    for task_idx in range(task_num):
                        head_loss = 0.0  # head_loss for a task_idx with all samples in a same batch
                        # for back propagation, assign as float, if called it will be tensor (Train)

                        # task_description_list[task_idx].shape = [B] as long-int (CLS) or float tensor (REG)
                        # for each sample in the batch, todo this need further optimization for bug batch size
                        for batch_idx, label_value in enumerate(task_description_list[task_idx]):
                            if label_value >= 99999999:  # stop sign
                                # Task track on missing or not
                                total_missing_task_sample_count[task_idx] += 1
                                minibatch_missing_task_sample_count[task_idx] += 1
                                continue  # record jump to skip this missing task

                            # take corresponding task criterion for task_description_list and predict output
                            # y[task_idx][batch_idx] we have [bag_size]:batch inside model pred (y)
                            output = y[task_idx][batch_idx].unsqueeze(0)  # [1,CLS] conf. or [1] reg.
                            label = label_value.to(device).unsqueeze(0)  # [1] long-int value out of k or [1] reg.
                            head_loss += criterions[task_idx](output, label)  # calculate B[1] loss and aggregate

                            # calculate and note down the measurement
                            if running_key_CLS[task_idx] != "not applicable":
                                # for CLS task, record the measurement in ACC
                                task_name = running_key_CLS[task_idx]
                                class_names = [key for key in running_cls_task_name_dict[task_name]]

                                _, preds = torch.max(output.cpu().data, 1)
                                long_labels = label.cpu().data
                                # check the tp for running_measurement
                                running_measurement[task_idx] += torch.sum(preds == long_labels)

                                # record pred and label for auc calc
                                pred_one_hot = torch.softmax(output.clone().detach(), dim=1).cpu().numpy()
                                label_one_hot = label.detach().cpu().numpy()
                                epoch_preds[task_idx].append(pred_one_hot)
                                epoch_labels[task_idx].append(label_one_hot)

                                # Compute tp tn fp fn for each class.
                                for cls_idx in range(len(epoch_cls_task_matrix_dict[task_name])):
                                    tp = np.dot(
                                        (long_labels == cls_idx).numpy().astype(int),
                                        (preds == cls_idx).cpu().numpy().astype(int),
                                    )
                                    tn = np.dot(
                                        (long_labels != cls_idx).numpy().astype(int),
                                        (preds != cls_idx).cpu().numpy().astype(int),
                                    )
                                    fp = np.sum((preds == cls_idx).cpu().numpy()) - tp
                                    fn = np.sum((long_labels == cls_idx).numpy()) - tp
                                    # epoch_cls_task_matrix_dict[task_name][cls_idx] = {'tp': 0.0, 'tn': 0.0,...}
                                    epoch_cls_task_matrix_dict[task_name][class_names[cls_idx]]["tp"] += tp
                                    epoch_cls_task_matrix_dict[task_name][class_names[cls_idx]]["tn"] += tn
                                    epoch_cls_task_matrix_dict[task_name][class_names[cls_idx]]["fp"] += fp
                                    epoch_cls_task_matrix_dict[task_name][class_names[cls_idx]]["fn"] += fn
                            else:  # for REG tasks
                                running_measurement[task_idx] += head_loss.item()
                                # record pred and label for auc calc
                                epoch_preds[task_idx].append(float(output))
                                epoch_labels[task_idx].append(float(label))

                        # build up loss for bp
                        # phase_running_loss will always record the loss on each sample
                        if type(head_loss) == float:  # the loss is not generated: maintains as float
                            pass  # they are all missing task_description_list for this task over the whole batch
                        else:
                            # Track the task wrt. the phase
                            phase_running_loss[task_idx] += head_loss.item()
                            # Track a task's loss (over a batch) into running_loss (all tasks loss / a batch)
                            if phase == "Train":
                                # todo in the future make a scheduler for loss_weight
                                running_loss += head_loss * loss_weight[task_idx]
                                # accum the loss for bag analysis
                                accum_average_sample_loss += running_loss.item()
                            else:  # val
                                running_loss += head_loss.item() * loss_weight[task_idx]
                                # accum the loss for bag analysis, no need gradient if not at training
                                accum_average_sample_loss += running_loss

                    # accum the bp loss from all tasks (over accum_iter of batches), if called it will be tensor (Train)
                    loss += running_loss / accum_iter  # loss for minibatch, remove the influence by loss-accumulate

                    if data_iter_step % accum_iter == accum_iter - 1:
                        index += 1  # index start with 1

                        # backward + optimize only if in training phase
                        if phase == "Train":
                            # in all-tasks in the running of batch * accum_iter,
                            if type(loss) == float:  # the loss is not generated: maintains as float
                                pass  # minor issue, just pass (only very few chance)
                            else:
                                Scaler.scale(loss).backward()
                                Scaler.step(optimizer)
                                Scaler.update()
                            # flush loss (accum_iter) only for train, but for val it will be compared later
                            loss = 0.0  # for back propagation, re-assign as float, if called it will be tensor (Train)
                        else:
                            # for val we keep the loss to find the best epochs
                            pass

                        # triggering minibatch (over accum_iter * batch) check (updating the recordings)
                        if index % check_minibatch == 0:

                            check_time = time.time() - model_time
                            model_time = time.time()
                            check_index = index // check_minibatch

                            print(f"In epoch: {epoch + 1} {phase}   index of {accum_iter} * {check_minibatch} "
                                  f"minibatch: {check_index}     time used: {check_time} seconds")

                            check_minibatch_results = []
                            for task_idx in range(task_num):
                                # temp loss sum values for check, accum loss is the accum loss until the previous time
                                temp_running_loss[task_idx] = (phase_running_loss[task_idx]
                                                               - accum_running_loss[task_idx])
                                # update accum
                                accum_running_loss[task_idx] += temp_running_loss[task_idx]
                                # update average running
                                valid_num = (accum_iter * check_minibatch * len(task_description_list[0])
                                             - minibatch_missing_task_sample_count[task_idx])
                                # assert valid_num != 0 # whole check runs should have at least 1 sample with 1 label
                                if valid_num == 0:
                                    task_name = running_key_CLS[task_idx] if running_key_CLS[
                                                                                 task_idx] != "not applicable" else \
                                    running_key_REG[task_idx]
                                    print(f"Warning: task {task_name} does not have valid sample in current check!")
                                else:
                                    temp_running_loss[task_idx] /= valid_num

                                    # create value
                                    temp_running_measurement[task_idx] = (running_measurement[task_idx]
                                                                          - accum_running_measurement[task_idx])
                                    # update accum
                                    accum_running_measurement[task_idx] += temp_running_measurement[task_idx]

                                    # CLS
                                    if running_key_CLS[task_idx] != "not applicable":
                                        check_minibatch_acc = temp_running_measurement[task_idx] / valid_num * 100

                                        # TP int(temp_running_measurement[task_idx])
                                        temp_running_results = (running_key_CLS[task_idx], float(check_minibatch_acc))
                                        check_minibatch_results.append(temp_running_results)
                                    # REG
                                    elif running_key_REG[task_idx] != "not applicable":
                                        temp_running_results = (running_key_REG[task_idx],
                                                                float(temp_running_measurement[task_idx]) / valid_num)
                                        check_minibatch_results.append(temp_running_results)
                                    else:
                                        print("record error in task_idx", task_idx)
                            print("Average_sample_loss:", temp_running_loss, "\n", check_minibatch_results, "\n")
                            # clean the missing count
                            minibatch_missing_task_sample_count = [0 for _ in range(task_num)]

            # total samples (remove dataloader-failed samples)
            total_samples = dataset_sizes[phase] - failed_sample_count
            # after an epoch, in train report loss for bag analysis
            if phase == "Train":
                epoch_average_sample_loss = accum_average_sample_loss / total_samples

                # todo use epoch_average_sample_loss to decide bag number? currently its wasted
                if LR_scheduler is not None:  # lr scheduler: update
                    LR_scheduler.step()
            # In val, we update best-epoch model index
            else:
                if type(loss) == float and loss == 0.0:
                    # in all running of val, the loss is not generated
                    print("In all running of val, the loss is not generated")
                    raise
                # compare the validation loss
                if loss <= temp_best_epoch_loss and (epoch + 1) >= intake_epochs:
                    best_epoch_idx = epoch + 1
                    temp_best_epoch_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                elif (epoch + 1) == intake_epochs:
                    # starting taking epoch weight
                    best_epoch_idx = epoch + 1
                    temp_best_epoch_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            epoch_results = []
            for task_idx in range(task_num):
                epoch_valid_num = total_samples - total_missing_task_sample_count[task_idx]
                assert epoch_valid_num != 0  # whole epoch should have at least 1 sample with 1 label
                # CLS
                if running_key_CLS[task_idx] != "not applicable":
                    epoch_acc = running_measurement[task_idx] / epoch_valid_num * 100
                    # calculate micro auc
                    epoch_preds[task_idx] = np.concatenate(epoch_preds[task_idx], axis=0)
                    epoch_labels[task_idx] = np.concatenate(epoch_labels[task_idx], axis=0)
                    try:
                        pred_one_hot = epoch_preds[task_idx]
                        auc_score = roc_auc_score(epoch_labels[task_idx], pred_one_hot, average='macro', multi_class='ovr')
                    except:
                        task_name = running_key_CLS[task_idx] if running_key_CLS[task_idx] != "not applicable" else \
                        running_key_REG[task_idx]
                        print(
                            f"Warning: task {task_name} has missing classes in {phase} set, thus cannot use confidence for AUC!")
                        auc_score = -1.0  # fixme: placeholder: here we use -1.0 to stands for missing value
                    # print results
                    results = (running_key_CLS[task_idx], epoch_cls_task_matrix_dict[running_key_CLS[task_idx]],
                               float(epoch_acc), auc_score)
                    epoch_results.append(results)
                # REG
                elif running_key_REG[task_idx] != "not applicable":
                    results = (running_key_REG[task_idx], float(running_measurement[task_idx]) / epoch_valid_num)
                    epoch_results.append(results)
                else:
                    print("record error in task_idx", task_idx)
                # loss
                phase_running_loss[task_idx] /= epoch_valid_num

            print(f"\nEpoch: {epoch + 1} {phase}     time used: {time.time() - epoch_time:.2f} seconds "
                  f"Average_sample_loss: {phase_running_loss}\n{epoch_results}\n\n")

            # attach the records to the tensorboard backend
            if writer is not None:
                # ...log the running loss
                for task_idx, task_name in enumerate(task_dict):
                    writer.add_scalar(phase + "_" + task_name + " loss", float(phase_running_loss[task_idx]), epoch + 1)
                    # use the last indicator as the measure indicator
                    writer.add_scalar(phase + "_" + task_name + " measure",
                                      float(epoch_results[task_idx][-1]), epoch + 1)

            if phase == "Train":
                # create the dict
                log_dict[epoch + 1] = {
                    phase: {"Average_sample_loss": phase_running_loss, "epoch_results": epoch_results}
                }
            else:
                log_dict[epoch + 1][phase] = {"Average_sample_loss": phase_running_loss, "epoch_results": epoch_results}

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("best_epoch_idx:", best_epoch_idx)
    for phase in log_dict[best_epoch_idx]:
        best_epoch_running_loss = log_dict[best_epoch_idx][phase]["Average_sample_loss"]
        best_epoch_results = log_dict[best_epoch_idx][phase]["epoch_results"]
        print("In:", phase, "    Average_sample_loss:", best_epoch_running_loss,
              "\nepoch_results:", best_epoch_results)
        # load best model weights as final model training result

    # attach the records to the tensorboard backend
    if writer is not None:
        writer.close()

    # save json_log  indent=2 for better view
    log_path = os.path.join(runs_path, time.strftime("%Y_%m_%d") + "_log.json")
    json.dump(log_dict, open(log_path, "w"), ensure_ascii=False, indent=2)

    model.load_state_dict(best_model_wts)
    return model, log_path


def main(args):
    run_name = "MTL_" + args.model_name
    run_name = run_name + "_" + str(args.tag) if args.tag is not None else run_name

    # PATH info
    runs_path = args.runs_path  # root path of saving the multiple experiments runs
    save_model_path = args.save_model_path or args.runs_path  # root path to saving models, if none, will go to draw root
    data_path = args.data_path  # path to a dataset
    # load label csv
    task_description_csv = args.task_description_csv or \
                           os.path.join(data_path, args.task_setting_folder_name, "task_description.csv")

    # filtered tasks
    task_idx_or_name_list = args.tasks_to_run.split("%") if args.tasks_to_run is not None else None

    # build task settings
    task_config_path = os.path.join(data_path, args.task_setting_folder_name, "task_configs.yaml")
    task_name_list, MTL_task_dict, MTL_heads_configs, criterions, loss_weight, class_num, task_describe = (
        task_filter_auto(Task_idx_or_name_list=task_idx_or_name_list, task_config_path=task_config_path))
    print("ROI task_dict", MTL_task_dict)

    # filtered tasks
    print("*********************************{}*************************************".format("settings"))
    for a in str(args).split(","):
        print(a)
    print("*********************************{}*************************************\n".format("setting"))

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

    if args.enable_tensorboard:
        writer = SummaryWriter(runs_path)
        # if u run locally
        # nohup tensorboard --logdir=/4tbB/WSIT/runs --host=0.0.0.0 --port=7777 &
        # tensorboard --logdir=/4tbB/WSIT/runs --host=0.0.0.0 --port=7777
        # python3 -m tensorboard.main --logdir=/Users/zhangtianyi/Desktop/ITH/results --host=172.31.209.166 --port=7777
    else:
        writer = None

    # Data Augmentation
    data_transforms = data_augmentation(args.data_augmentation_mode, edge_size=args.edge_size)

    # initiate the dataset
    Train_dataset = Bulk_ROI_Dataset(
        data_path,
        task_description_csv,
        task_setting_folder_name=args.task_setting_folder_name,
        task_name_list=task_name_list,
        split_target_key=args.split_target_key,
        split_name="Train",
        edge_size=args.edge_size,
        transform=data_transforms['Train'])
    Val_dataset = Bulk_ROI_Dataset(
        data_path,
        task_description_csv,
        task_setting_folder_name=args.task_setting_folder_name,
        task_name_list=task_name_list,
        split_target_key=args.split_target_key,
        split_name="Val",
        edge_size=args.edge_size,
        transform=data_transforms['Val'])

    # print(Train_dataset.get_embedded_sample_with_try(20))
    dataloaders = {
        "Train": torch.utils.data.DataLoader(
            Train_dataset,
            batch_size=args.batch_size,
            collate_fn=MTL_collate_fn,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        ),
        "Val": torch.utils.data.DataLoader(
            Val_dataset,
            batch_size=args.batch_size,
            collate_fn=MTL_collate_fn,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
        ),
    }
    dataset_sizes = {"Train": len(Train_dataset), "Val": len(Val_dataset)}

    # GPU idx start with 0. -1 to use multiple GPU
    if args.gpu_idx == -1:  # use all cards
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            gpu_use = args.gpu_idx
        else:
            print("we dont have more GPU idx here, try to use gpu_idx=0")
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # setting k for: only card idx k is sighted for this code
                gpu_use = 0
            except:
                print("GPU distributing ERROR occur use CPU instead")
                gpu_use = "cpu"

    else:
        # Decide which device we want to run on
        try:
            # setting k for: only card idx k is sighted for this code
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
            gpu_use = args.gpu_idx
        except:
            print("we dont have that GPU idx here, try to use gpu_idx=0")
            try:
                # setting 0 for: only card idx 0 is sighted for this code
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                gpu_use = 0
            except:
                print("GPU distributing ERROR occur use CPU instead")
                gpu_use = "cpu"

    # device environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = build_ROI_task_model(
        model_name=args.model_name, edge_size=args.edge_size,
        MTL_heads_configs=MTL_heads_configs,
        latent_feature_dim=args.latent_feature_dim,
        pretrained_backbone=True if args.model_weight_path is None else args.model_weight_path)

    model = model.to(device)
    # fixme this have bug for gigapath in train, but ok with val, possible issue with Triton
    # model = torch.compile(model)

    print("GPU:", gpu_use)
    if gpu_use == -1:
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    # cosine Scheduler by https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.num_epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    LR_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    trained_model, log_path = ROI_MTL_train(model, dataloaders, dataset_sizes,
                                            criterions, optimizer, LR_scheduler, loss_weight,
                                            MTL_task_dict, task_describe, num_epochs=args.num_epochs,
                                            accum_iter_train=args.accum_iter_train,
                                            check_minibatch=args.check_minibatch, intake_epochs=args.intake_epochs,
                                            runs_path=runs_path, writer=writer, device=device)

    if gpu_use == -1:
        state_dict = trained_model.module.state_dict()
    else:
        state_dict = trained_model.state_dict()

    torch.save(state_dict, save_model_path)
    print('model trained by GPU (idx:' + str(gpu_use) + ') has been saved at ', save_model_path)

    # print training summary
    check_json_with_plot(log_path, MTL_task_dict, save_path=runs_path)


def get_args_parser():
    parser = argparse.ArgumentParser(description="MTL Training")

    # Environment parameters
    parser.add_argument("--gpu_idx", default=0, type=int,
                        help="use a single GPU with its index, -1 to use multiple GPU")

    # Model settings
    parser.add_argument("--model_name", default="vit", type=str, help="MTL backbone model name")
    # Model tag (for example k-fold)
    parser.add_argument("--tag", default=None, type=str, help="Model tag (for example 5-fold)")
    # Module settings
    parser.add_argument('--edge_size', default=224, type=int, help='edge size of input image')
    parser.add_argument("--latent_feature_dim", default=128, type=int, help="MTL module dim")

    # PATH
    parser.add_argument("--data_path", default=None, type=str, help="MTL dataset root")
    parser.add_argument("--save_model_path", default=None, type=str, help="save model root")
    parser.add_argument("--runs_path", default=ROOT_PATH / "runs", type=str, help="save running results path")
    parser.add_argument('--model_weight_path', type=str, default=None,
                        help='path of the embedding model weight')
    # labels
    parser.add_argument("--task_description_csv", default=None, type=str, help="label csv file path")

    # Task settings
    parser.add_argument("--tasks_to_run", default=None, type=str,
                        help="tasks to run MTL, split with %, default is None with all tasks in task config to be run")
    # Task settings and configurations for dataloaders
    parser.add_argument("--task_setting_folder_name", default="task-settings-5folds",
                        type=str, help="task-settings folder name")
    parser.add_argument("--split_target_key", default="fold_information_5fold-1",
                        type=str, help="key identifying the split information")
    parser.add_argument("--num_workers", default=20, type=int, help="dataloader num_workers")

    # training settings
    parser.add_argument("--batch_size", default=200, type=int, help="batch_size , default 1")
    parser.add_argument("--num_epochs", default=10, type=int, help="total training epochs, default 200")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="warmup_epochs training epochs, default 50")
    parser.add_argument("--intake_epochs", default=5, type=int, help="only save model at epochs after intake_epochs")
    parser.add_argument("--accum_iter_train", default=1, type=int,
                        help="training accum_iter for loss accuming, default 2")
    parser.add_argument("--lr", default=1e-5, type=float, help="training learning rate, default 0.00001")
    parser.add_argument("--lrf", default=0.1, type=float, help="Cosine learning rate decay times, default 0.1")

    # Dataset specific augmentations in dataloader
    parser.add_argument('--data_augmentation_mode', default=-1, type=int, help='data_augmentation_mode')

    # helper
    parser.add_argument("--check_minibatch", default=100, type=int, help="check batch_size")
    parser.add_argument("--enable_notify", action="store_true", help="enable notify to send email")
    parser.add_argument("--enable_tensorboard", action="store_true", help="enable tensorboard to save status")

    return parser


if __name__ == "__main__":
    # setting up the random seed
    setup_seed(42)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

    '''
    If you have encounter EOF error 'open too many files'
    # Increase the maximum number of open files allowed by your system:
    ulimit -n 65536
    '''
