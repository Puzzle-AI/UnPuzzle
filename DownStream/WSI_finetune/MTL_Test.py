"""
MTL Test      Script ver: Feb 8th 01:00
flexible to multiple-tasks and missing labels
"""
import os
import sys
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

# For convenience, import all path to sys
# Go up 3 levels
ROOT_PATH = Path(__file__).resolve().parent.parent.parent

this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir))
sys.path.append(str(this_file_dir.parent))
sys.path.append(str(this_file_dir.parent.parent))
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from DataPipe.dataset_framework import SlideDataset, MTL_collate_fn
from DownStream.MTL.task_settings import task_filter_auto, task_idx_converter, result_recorder
from ModelBase.Get_WSI_model import build_WSI_task_model
from ModelBase.MTL_modules.modules import bin_MTL_tasks
from Utils.tools import setup_seed


def WSI_MTL_test(model, dataloader, dataset_size, criterions, loss_weight, task_dict, task_describe, idx_converter,
                 check_minibatch=20, runs_path="./", device=torch.device("cpu")):
    since = time.time()

    # log dict
    log_dict = {}

    task_num = len(task_dict)
    running_key_CLS = []
    running_key_REG = []
    running_task_name_dict = {}
    for key in task_dict:
        if task_dict[key] == list:
            running_task_name_dict[key] = task_describe[key]
            running_key_CLS.append(key)
            running_key_REG.append("not applicable")
        else:
            running_key_CLS.append("not applicable")
            running_key_REG.append(key)

    model.eval()  # Set model to evaluate mode

    epoch_time = time.time()
    model_time = time.time()
    index = 0

    epoch_running_loss = [0.0 for _ in range(task_num)]
    temp_running_loss = [0.0 for _ in range(task_num)]
    accum_running_loss = [0.0 for _ in range(task_num)]

    running_measurement = [0.0 for _ in range(task_num)]  # give every task a 0.0
    temp_running_measurement = [0.0 for _ in range(task_num)]
    accum_running_measurement = [0.0 for _ in range(task_num)]

    epoch_preds = [[] for _ in range(task_num)]     # pred in current epoch
    epoch_labels = [[] for _ in range(task_num)]     # label in current epoch

    # missing count
    minibatch_missing_task_sample_count = [0 for _ in range(task_num)]
    total_missing_task_sample_count = [0 for _ in range(task_num)]

    loss = 0.0  # for back propagation, assign as float
    failed_sample_count = 0  # the whole batch is available by dataloader

    test_recorder = result_recorder(
        task_dict=task_dict,
        task_describe=task_describe,
        batch_size=dataloader.batch_size,
        total_size=dataset_size,
        runs_path=runs_path,
    )

    # all matrix dict (initialize them with 0 values for each task)
    epoch_cls_task_matrix_dict = {}
    for cls_task_name in running_task_name_dict:
        class_names = [key for key in running_task_name_dict[cls_task_name]]
        # initiate the empty matrix_log_dict
        matrix_log_dict = {}
        for cls_idx in range(len(class_names)):
            # only float type is allowed in json, set to float inside
            matrix_log_dict[class_names[cls_idx]] = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
        epoch_cls_task_matrix_dict[cls_task_name] = matrix_log_dict

    # data loop
    for data_iter_step, sample in enumerate(dataloader):

        # move recorder to the correct sample batch
        test_recorder.add_step(data_iter_step)

        # jump the batch if it cannot correct by WSI_collate_fn in dataloader
        if sample is None:
            failed_sample_count += dataloader.batch_size
            continue

        else:
            # take data and task_description_list from sample
            # image_features is a tensor of [B,N,D],  coords_yx is tensor of [B,N,2]
            image_features = sample['image_features'].to(device)
            coords_yx = sample['coords_yx'].to(device)
            # task_description_list [task, batch_size] batch-stacked tensors, element of long-int or float
            task_description_list = sample['task_description_list']
            slide_id = sample['slide_id']

        # count failed samples in dataloader (should be 0, normally)
        # default B - B = 0, we don't have the last batch issue 'drop last batch in training code'
        failed_sample_count += dataloader.batch_size - len(task_description_list[0])

        # Tracking the loss for loss-drive bag settings update and also recordings, initialize with 0.0
        running_loss = 0.0  # all tasks loss over a batch

        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                y = model(image_features, coords_yx)  # y is the predication on all old tasks (of training)

            for task_idx in range(task_num):
                head_loss = 0.0  # head_loss for a task_idx with all samples in a same batch
                # for back propagation, assign as float, if called it will be tensor (Train)

                for batch_idx, label_value in enumerate(task_description_list[task_idx]):  # in batch
                    # y[task_idx][batch_idx] we have [bag_size]:batch inside model pred (y)
                    output = y[idx_converter(task_idx)][batch_idx].unsqueeze(0)  # [1,CLS] conf. or [1] reg.

                    if label_value >= 99999999:  # stop sign fixme: maybe it's a temporary stop sign
                        # Task track on missing or not
                        total_missing_task_sample_count[task_idx] += 1
                        minibatch_missing_task_sample_count[task_idx] += 1

                        # record predications
                        label = None
                        if running_key_CLS[task_idx] != "not applicable":  # cls task
                            _, preds = torch.max(output.cpu().data, 1)
                            preds = int(preds)
                        else:  # reg task
                            preds = float(output.cpu().data)
                        # record sample predication for classification task
                        test_recorder.add_data(batch_idx, task_idx, preds, label)

                        continue  # record jump to skip this missing task for correct calculation

                    # take corresponding task criterion for task_description_list and predict output
                    label = label_value.to(device).unsqueeze(0)  # [1] long-int value out of k or [1] reg.
                    head_loss += criterions[task_idx](output, label)  # calculate B[1] loss and aggregate
                    # calculate and note down the measurement
                    if running_key_CLS[task_idx] != "not applicable":
                        # for CLS task, record the measurement in ACC
                        task_name = running_key_CLS[task_idx]
                        class_names = [key for key in running_task_name_dict[task_name]]

                        _, preds = torch.max(output.cpu().data, 1)
                        long_labels = label.cpu().data
                        # check the tp for running_measurement
                        running_measurement[task_idx] += torch.sum(preds == long_labels)

                        # record pred and label for auc calc
                        pred_one_hot = torch.softmax(output.clone().detach(), dim=1).cpu().numpy()
                        label_one_hot = label.detach().cpu().numpy()
                        epoch_preds[task_idx].append(pred_one_hot)
                        epoch_labels[task_idx].append(label_one_hot)

                        # record sample predication for classification task
                        test_recorder.add_data(batch_idx, task_idx, int(preds), int(long_labels), pred_one_hot, label_one_hot)

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

                        # record sample predication for regression task
                        test_recorder.add_data(batch_idx, task_idx, float(output), float(label))

                if type(head_loss) == float:  # the loss is not generated: maintains as float
                    print(
                        f"they are all missing task_description_list for this task: {task_idx}\n"
                        f"over the whole batch of data_iter_step: {data_iter_step}")

                    pass  # they are all missing task_description_list for this task over the whole batch
                else:
                    # Track a task's loss (over a batch) into running_loss (all tasks loss / a batch)
                    # For running loss: balanced loss as its meaning is measuring the balanced process
                    running_loss += head_loss.item() * loss_weight[task_idx]
                    # For epoch recording, we measure the actual situation of each task
                    epoch_running_loss[task_idx] += head_loss.item()

            # accum the loss from all tasks (from a batches), if called it will be more than 0.0
            loss += float(running_loss)

            index += 1  # index start with 1

            if index % check_minibatch == 0:

                check_time = time.time() - model_time
                model_time = time.time()
                check_index = index // check_minibatch
                epoch_idx = "Test"

                print(f"In epoch: {epoch_idx} index of {check_minibatch} minibatch: "
                      f"{check_index}     time used: {check_time}")

                check_minibatch_results = []
                for task_idx in range(task_num):
                    # create value
                    temp_running_loss[task_idx] = epoch_running_loss[task_idx] - accum_running_loss[task_idx]
                    # update accum
                    accum_running_loss[task_idx] += temp_running_loss[task_idx]
                    # update average running
                    valid_num = (check_minibatch * len(task_description_list[0])
                                 - minibatch_missing_task_sample_count[task_idx])
                    # fixme in test maybe we can easy the limit (wrt. valid_num)?
                    # if we want to check something (or maybe assign an empty label in dataset without task_description_list)
                    # assert valid_num != 0  # whole check runs should have at least 1 sample with 1 label
                    if valid_num == 0:
                        task_name = running_key_CLS[task_idx] if running_key_CLS[task_idx] != "not applicable" else running_key_REG[task_idx]
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
    total_samples = dataset_size - failed_sample_count
    epoch_results = []
    for task_idx in range(task_num):
        epoch_valid_num = total_samples - total_missing_task_sample_count[task_idx]
        # fixme in test maybe we can easy the limit?
        #  if we want to check something (or maybe assign an empty label in dataset without task_description_list)
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
            except Exception as e:
                task_name = running_key_CLS[task_idx] if running_key_CLS[task_idx] != "not applicable" else running_key_REG[task_idx]
                print(f"Warning: task {task_name} has missing classes in test set, thus cannot use confidence for AUC!")
                auc_score = -1.0    # fixme: placeholder: here we use -1.0 to stands for missing value
            # print results
            results = (
                running_key_CLS[task_idx],
                epoch_cls_task_matrix_dict[running_key_CLS[task_idx]],
                auc_score,
                float(epoch_acc))
            epoch_results.append(results)
        # REG
        elif running_key_REG[task_idx] != "not applicable":
            results = (
                running_key_REG[task_idx], 
                float(running_measurement[task_idx]) / epoch_valid_num)
            epoch_results.append(results)
        else:
            print("record error in task_idx", task_idx)
        # loss
        epoch_running_loss[task_idx] /= epoch_valid_num

    print(f"\nEpoch: Test     time used: {time.time() - epoch_time:.2f} seconds "
          f"Average_sample_loss for each task: {epoch_running_loss}\n{epoch_results}\n\n")

    # create the dict
    log_dict["Test"] = {"Test": {"Average_sample_loss": epoch_running_loss, "epoch_results": epoch_results}}
    average_sum_loss = loss / dataset_size
    print("Average sample loss (weighted sum):", average_sum_loss)

    # save json_log  indent=2 for better view
    log_path = os.path.join(runs_path, time.strftime("%Y_%m_%d") + "_log.json")
    json.dump(log_dict, open(log_path, "w"), ensure_ascii=False, indent=2)

    test_recorder.finish_and_dump(tag="test_")

    time_elapsed = time.time() - since
    print("Testing complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


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
    old_task_idx_or_name_list = args.old_tasks_to_run.split("%") if args.old_tasks_to_run is not None else None

    # build task settings
    task_config_path = os.path.join(data_path, args.task_setting_folder_name, "task_configs.yaml")
    task_name_list, MTL_task_dict, MTL_heads_configs, criterions, loss_weight, class_num, task_describe = task_filter_auto(
        Task_idx_or_name_list=task_idx_or_name_list, task_config_path=task_config_path)
    print("WSI MTL_task_dict", MTL_task_dict)

    if args.old_task_config is not None and os.path.exists(args.old_task_config):
        # fixme in this case we actually need to put the previous tasks been run,
        #  but here we assume as all previous tasks has been run
        _, old_task_dict, MTL_heads_configs, _, _, _, _ = task_filter_auto(
            Task_idx_or_name_list=old_task_idx_or_name_list, task_config_path=args.old_task_config)
        print("old_task_dict", old_task_dict)
        idx_converter = task_idx_converter(old_task_dict, MTL_task_dict)
    else:
        idx_converter = task_idx_converter(MTL_task_dict, MTL_task_dict)
    # filtered tasks
    print("*********************************{}*************************************".format("settings"))
    for a in str(args).split(","):
        print(a)
    print("*********************************{}*************************************\n".format("setting"))

    # we use the run name as a warp folder for both Train and Test
    assert os.path.exists(os.path.join(runs_path, run_name))
    runs_path = os.path.join(runs_path, run_name, 'Test')
    os.makedirs(runs_path, exist_ok=True)
    save_model_path = os.path.join(save_model_path, run_name, run_name + '.pth')

    # Set padding for loading
    if args.batch_size > 1:
        assert len(task_idx_or_name_list) == 1, "only support single task learning when batch_size>1"
        if args.padding is not True:
            print(f'Warning: padding set to True automatically as batch_size > 1')
            args.padding = True

    # test dataloader
    dataset = SlideDataset(
        data_path,
        task_description_csv,
        task_setting_folder_name=args.task_setting_folder_name,
        split_name="Test",
        slide_id_key=args.slide_id_key,
        split_target_key=args.split_target_key,
        task_name_list=task_name_list,
        max_tiles=args.max_tiles,
        shuffle_tiles=args.shuffle_tiles,
        padding=args.padding
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             collate_fn=MTL_collate_fn,
                                             shuffle=False, num_workers=args.num_workers)
    # info
    dataset_size = len(dataset)

    # GPU idx start with0, -1 to use multiple GPU
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
                print("GPU distributing ERRO occur use CPU instead")
                gpu_use = "cpu"
    # device environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build binning config fixme temp design, future move into model building
    if args.Head_strategy == 'expression_bin':
        task_description_data_df = pd.read_csv(task_description_csv)
        bin_df = bin_MTL_tasks(task_description_data_df, task_name_list,
                               MTL_token_bins_num = args.num_of_bin, method='expression_bin')
        bin_df.to_csv(args.runs_path +'/'+"bin_df", index=False, header = True)
    else:
        bin_df = None

    # build model
    model = build_WSI_task_model(
        model_name=args.model_name,
        local_weight_path=False,
        ROI_feature_dim=args.ROI_feature_dim,
        MTL_heads_configs=MTL_heads_configs,
        latent_feature_dim=args.latent_feature_dim,
        Head_strategy=args.Head_strategy,
        bin_df=bin_df)
    
    loaded_dict = torch.load(save_model_path, weights_only=True)
    load_result = model.load_state_dict(loaded_dict)
    print(f'loading status: {load_result}')
    print(">> Missing keys:")
    for k in load_result.missing_keys:
        print("   ", k)
    print(">> Unexpected keys:")
    for k in load_result.unexpected_keys:
        print("   ", k)
    print(f"model loaded from {save_model_path}")

    # # load model
    # model.load_state_dict(torch.load(save_model_path, weights_only=True))
    # print("model loaded")

    # model = torch.compile(model)  # this cause some delay, Tianyi is fixing and exploring

    print("GPU:", gpu_use)
    if gpu_use == -1:
        model = nn.DataParallel(model)

    model.to(device)

    WSI_MTL_test(model, dataloader, dataset_size, criterions, loss_weight, MTL_task_dict, task_describe,
                 idx_converter, check_minibatch=20, runs_path=runs_path, device=device)


def get_args_parser():
    parser = argparse.ArgumentParser(description="MTL Testing")
    # Environment parameters
    parser.add_argument("--gpu_idx", default=0, type=int,
                        help="use a single GPU with its index, -1 to use multiple GPU")
    # Name
    parser.add_argument("--model_name", default="gigapath", type=str, help="slide level model name")
    # Model tag (for example k-fold)
    parser.add_argument("--tag", default=None, type=str, help="Model tag (for example 5-fold)")

    # PATH
    parser.add_argument("--data_path", default=None, type=str, help="MTL dataset root")
    parser.add_argument("--save_model_path", default=None, type=str, help="save model root")
    parser.add_argument("--runs_path", default=ROOT_PATH / "runs", type=str, help="save running results path")
    # labels
    parser.add_argument("--task_description_csv", default=None, type=str, help="label csv file path")
    # old task config (this allows testing old model on new set of tasks)
    parser.add_argument("--old_task_config", default=None, type=str, help="path to old training config file")

    # Task settings
    parser.add_argument("--tasks_to_run",default=None,type=str,
                        help="tasks to run MTL, split with %, default is None with all tasks to be run")
    parser.add_argument("--old_tasks_to_run",default=None,type=str,
                        help="tasks to run MTL, split with %, default is None with all tasks to be run")

    # Task settings and configurations for dataloaders
    parser.add_argument("--task_setting_folder_name", default="task-settings", type=str,
                        help="task-settings folder name")
    parser.add_argument("--slide_id_key", default="patient_id", type=str, help="key for mapping the label")
    parser.add_argument("--split_target_key", default="fold_information", type=str,
                        help="key identifying the split information")
    parser.add_argument("--num_workers", default=2, type=int, help="dataloader num_workers")
    parser.add_argument("--max_tiles", default=None, type=int,
                        help="max tile for loading, default will load config or be 10000")
    parser.add_argument("--shuffle_tiles", nargs='?', const=True,
                        type=lambda x: x.lower() in ['true', 'True'] if x else True,
                        default=None, help="set empty or True as value to shuffle the tiles in loading. "
                                           "Defaults to None if not specified.")
    # module settings
    parser.add_argument("--padding", action='store_true', help="padding the tiles to enable batch fetching")
    parser.add_argument("--latent_feature_dim", default=128, type=int, help="MTL module dim")
    parser.add_argument("--slide_embed_dim", default=768, type=int, help="feature slide_embed_dim , default 768")
    parser.add_argument("--ROI_feature_dim", default=1536, type=int, help="feature slide_embed_dim , default 768")

    # Model settings
    parser.add_argument("--Head_strategy", default=None, type=str, help="expression_bin for binning method")
    parser.add_argument("--num_of_bin", default=None, type=int,
                        help="total number of bins if binning method is expression_bin ")

    # training settings
    parser.add_argument("--batch_size", default=1, type=int, help="batch_size , default 1")

    # help
    parser.add_argument("--check_minibatch", default=25, type=int, help="check batch_size")
    parser.add_argument("--enable_notify", action="store_true", help="enable notify to send email")

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
