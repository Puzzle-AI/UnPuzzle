"""
MTL Train     Script  ver: Dec 8th 15:00

flexible to multiple-tasks and missing labels

we have enable multiple samples training by controlling the gradient in different task labels
we break the process of controlling when calculating the gradient, and
we use loss-aggregate technique to combine each sample for back-propagation

todo:
* [ ] When some tasks have label and some doesnt, DDP cannot work
* [ ] Add checkpoint

"""

import os
import sys
from pathlib import Path

# Go up 3 levels
ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_PATH))

import json
import copy
import time
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from itertools import product
from torch.distributed import all_reduce, ReduceOp
from torch.utils.data.distributed import DistributedSampler
from DataPipe.dataset_framework import SlideDataset, MTL_collate_fn
from DownStream.MTL.task_settings import task_filter_auto
from ModelBase.Get_WSI_model import build_WSI_task_model
from Utils.MTL_plot_json import check_json_with_plot
from Utils.tools import setup_seed, create_logger, setup_devices_ddp, variable_sum_ddp


def setup_tasks(task_dict, task_describe):
    """Initialize tasks and categorize them into classification (CLS) and regression (REG)."""
    task_names = {'CLS': [], 'REG': []}  # Separate classification (CLS) and regression (REG) tasks
    task_description = {}
    for key, task_type in task_dict.items():
        if task_type == list:  # Classification task
            task_description[key] = task_describe[key]
            task_names['CLS'].append(key)
        else:  # Regression task
            task_names['REG'].append(key)
    return task_names, task_description

def initialize_epoch_metrics(task_descriptions):
    """Initialize the dictionary to track metrics for classification tasks.
    For classification tasks, the metric will store metric;
    For regression tasks, the metric will store loss.
    """
    metrics = {}
    for task, classes in task_descriptions.items():
        metrics[task] = {cls_name: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for cls_name in classes}
    return metrics

def calculate_metrics(output, label, metrics_dict, task_name):
    """Calculate TP, TN, FP, FN for each class in a classification task."""
    _, preds = torch.max(output.cpu(), 1)  # Get predicted class
    label = label.cpu()  # Move label to CPU for calculation
    # print(f'pred: {preds}, label: {label}')
    for cls_idx, cls_name in enumerate(metrics_dict[task_name]):
        tp = np.dot((label == cls_idx).numpy(), (preds == cls_idx).numpy())
        tn = np.dot((label != cls_idx).numpy(), (preds != cls_idx).numpy())
        fp = preds.eq(cls_idx).sum().item() - tp
        fn = label.eq(cls_idx).sum().item() - tp
        # Update metrics dictionary
        metrics_dict[task_name][cls_name]['tp'] += int(tp)
        metrics_dict[task_name][cls_name]['tn'] += int(tn)
        metrics_dict[task_name][cls_name]['fp'] += int(fp)
        metrics_dict[task_name][cls_name]['fn'] += int(fn)

def calculate_accuracy(preds, labels):
    """Calculate the accuracy for a batch of predictions and labels."""
    correct_predictions = torch.sum(preds == labels).item()
    total_predictions = labels.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy

def make_epoch_results(epoch_task_loss, epoch_task_acc, epoch_metrics, task_names):
    # generate epoch result for plot
    epoch_results = {}
    for task_name in task_names['CLS']:
        epoch_results[task_name] = {
            'epoch_task_loss': epoch_task_loss,
            'epoch_metrics': epoch_metrics,
            'epoch_task_acc': epoch_task_acc
        }
    for task_name in task_names['REG']:
        epoch_results[task_name] = {
            'epoch_task_loss': epoch_task_loss
        }
    return epoch_results

def process_batch(model, data, criterions, device, task_names, loss_weight, epoch_metrics,
                  epoch_task_acc={}, epoch_task_loss={}, epoch_sample_cnt={}):
    """Process a single batch and calculate losses for each task."""
    image_features, coords_yx, task_descriptions, _ = data
    image_features, coords_yx = image_features.to(device), coords_yx.to(device)
    outputs = model(image_features, coords_yx)  # Forward pass through model
    
    total_loss = torch.tensor(0.0, device=device)
    
    for task_idx, task_name in enumerate(task_names['CLS'] + task_names['REG']):
        task_loss = torch.tensor(0.0, device=device)  # Initialize task-specific loss
        task_labels = task_descriptions[task_idx].to(device)
        
        for batch_idx, label in enumerate(task_labels):
            if label.item() < 99999999:  # If valid label (not a stop sign)
                output = outputs[task_idx][batch_idx].unsqueeze(0)  # [1,bag_size] conf. or [1] reg.
                task_loss += criterions[task_idx](output, label.unsqueeze(0))   # calculate B[1] loss and aggregate
                epoch_sample_cnt[task_name] = epoch_sample_cnt.get(task_name, 0) + 1
                if task_name in task_names['CLS']:
                    # For classification tasks, calculate accuracy
                    _, preds = torch.max(output, 1)
                    accuracy = calculate_accuracy(preds, label.unsqueeze(0))
                    epoch_task_acc[task_name] = epoch_task_acc.get(task_name, 0) + accuracy
                    calculate_metrics(output, label, epoch_metrics, task_name)

        # Apply weighting to task loss and add to total loss
        if task_loss != 0.0:
            weighted_loss = task_loss * loss_weight[task_idx]
            total_loss += weighted_loss
            epoch_task_loss[task_name] = epoch_task_loss.get(task_name, 0) + weighted_loss.detach()

    return total_loss, epoch_metrics, epoch_task_acc, epoch_task_loss, epoch_sample_cnt

def train_epoch(
        model, scaler, dataloader, optimizer, phase, criterions, device, task_names, task_descriptions, loss_weight, mix_precision, 
        accum_iter, check_minibatch, logger, writer=None, epoch=0, local_rank=0):
    """Run training or validation for one epoch, with accuracy tracking for classification tasks."""
    
    # Set model to training or evaluation mode
    model.train() if phase == 'Train' else model.eval()

    epoch_loss = torch.tensor(0.0, device=device)    # Track loss for the epoch, just for logging
    accum_loss = torch.tensor(0.0, device=device)    # Track loss for accumulated batches for back propagte
    epoch_metrics = initialize_epoch_metrics(task_descriptions) # Track epoch classification result metrics
    epoch_task_acc = {}   # Initialize accuracy tracking for classification tasks
    epoch_task_loss = {}    # Initialize task loss for all tasks
    epoch_sample_cnt = {}   # Count samples amount in current batch for each task

    model_time = time.time()
    step_minibatch = 0
    minibatch_index = 0

    for step, data in enumerate(dataloader):
        if data is None:  # Skip invalid batches
            continue

        if step % accum_iter == 0:
            optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=mix_precision):  # Enable mixed precision casting
            batch_loss, epoch_metrics, epoch_task_acc, epoch_task_loss, epoch_sample_cnt = process_batch(
                model, data, criterions, device, task_names, loss_weight, 
                epoch_metrics, epoch_task_acc, epoch_task_loss, epoch_sample_cnt)
        
        # Accumulate loss
        # logger.info(f'getting accum_loss...')
        # batch_loss = variable_sum_ddp(batch_loss, device, avg=True)
        accum_loss += (batch_loss / accum_iter)

        # In a multi-GPU setup, some GPUs may not receive values, causing an imbalance where some GPUs 
        # have updated gradients while others do not. This results in a synchronization error, halting 
        # the process as it waits for all GPUs to align.
        # In this case, we process all loss and the loss will be ignored if it is 0.
        if step % accum_iter == 0:  

            if phase == 'Train':    # Only backpropagate and optimize in training phase
                # logger.info(f'updating params based on accum_loss: {accum_loss}')
                # print(f'updating params based on accum_loss: {accum_loss}')
                if not accum_loss.requires_grad:  # Check if accum_loss has no grad. No grad means no valid sample in current batch.
                    scaler.update()
                else:
                    scaler.scale(accum_loss).backward()     # Scale and backpropagate accum_loss
                    scaler.step(optimizer)                  # Update model parameters
                    scaler.update()                         # Update the scaler for next iteration

                # flush loss (accum_iter) only for train, but for val it will be compared later
                accum_loss = torch.tensor(0.0, device=device)
            step_minibatch += 1

        torch.cuda.synchronize()  # wait till all device's task finished
        
        # Update cumulative epoch loss
        # logger.info(f'updating epoch_loss...')
        epoch_loss += batch_loss

        # check for each minibatch
        # logger.info(f'checking for minibatch...')
        if step_minibatch % check_minibatch == 0 and step_minibatch != 0:
            # print('checking outputs')
            check_time = time.time() - model_time
            model_time = time.time()
            minibatch_index += 1
            step_minibatch = 0

            # # get global result
            # logger.info(f'defining visualize_metrics...')
            # visualize_metrics = copy.deepcopy(epoch_metrics)
            # logger.info(f'defining visualize_task_loss...')
            visualize_task_loss = copy.deepcopy(epoch_task_loss)
            # logger.info(f'sum visualize_metrics...')
            # for task_name in visualize_metrics:
            #     for cls_name in visualize_metrics[task_name]:
            #         for stat in visualize_metrics[task_name][cls_name]:
            #             print(f'[rank{device}] single task visualize_metrics[{task_name}][{cls_name}][{stat}]: {visualize_metrics[task_name][cls_name][stat]}, {type(visualize_metrics[task_name][cls_name][stat])}')
            #             visualize_metrics[task_name][cls_name][stat] = variable_sum_ddp(visualize_metrics[task_name][cls_name][stat], device)
            #             if visualize_metrics[task_name][cls_name][stat] >= 1000:
            #                 raise ValueError(f'invalid visualize_metrics[{task_name}][{cls_name}][{stat}]: {visualize_metrics[task_name][cls_name][stat]}')
            #             logger.info(f'visualize_metrics[{task_name}][{cls_name}][{stat}]: {visualize_metrics[task_name][cls_name][stat]}')
            # logger.info(f'sum visualize_task_loss...')
            for task_name in visualize_task_loss:
                # print(f'single task visualize_task_loss {task_name}: {visualize_task_loss[task_name]}')
                visualize_task_loss[task_name] = variable_sum_ddp(visualize_task_loss[task_name], device) / epoch_sample_cnt[task_name]
                # logger.info(f'visualize_task_loss {task_name}: {visualize_task_loss[task_name]}')

            # logger.info(f'step: {step}/{len(dataloader)} time used: {check_time:.2f} seconds')
            logger.info(f"epoch {epoch + 1} | {phase} | minibatch index {minibatch_index}/{len(dataloader) // (check_minibatch * accum_iter)} | task loss: {visualize_task_loss} | time used: {check_time:.2f} seconds")
            # logger.info(f'performance metrics: {visualize_metrics}')
    
    logger.info(f'epoch {epoch + 1} finished!')
    # abort if no valid sample label found for each task or the entire dataset
    assert epoch_loss != 0.0, ValueError(f'Not a single valid label founded in current dataset, aborted!')
    for task_name in task_names['REG'] + task_names['CLS']:
        assert task_name in epoch_task_loss, ValueError(f'Not a single valid label founded for task {task_name}, aborted!')

    # all reduce to get global epoch metrics, epoch loss, and epoch accuracy
    epoch_loss = variable_sum_ddp(epoch_loss, device) / len(dataloader)
    for task_name in epoch_sample_cnt:  # get global epoch sample count
        epoch_sample_cnt[task_name] = variable_sum_ddp(epoch_sample_cnt[task_name], device)
    for task_name in epoch_metrics:     # calculate epoch accuracy
        epoch_task_acc[task_name] = variable_sum_ddp(epoch_task_acc[task_name], device) / epoch_sample_cnt[task_name]
    for task_name in epoch_task_loss:   # calculate epoch loss
        epoch_task_loss[task_name] = variable_sum_ddp(epoch_task_loss[task_name], device) / epoch_sample_cnt[task_name]
    for task_name in epoch_metrics:     # sum all classification metrics
        for cls_name in epoch_metrics[task_name]:
            for stat in epoch_metrics[task_name][cls_name]:
                epoch_metrics[task_name][cls_name][stat] = variable_sum_ddp(epoch_metrics[task_name][cls_name][stat], device)

    # Log metrics to TensorBoard if writer is provided
    if writer is not None and local_rank == 0:
        # CLS: accuracy
        for task_name in task_names['CLS']:
            writer.add_scalar(f'{phase}_{task_name}_Accuracy', epoch_task_acc[task_name], epoch + 1)
        # CLS and REG: loss
        for task_name in task_names['REG'] + task_names['CLS']:
            writer.add_scalar(f'{phase}_{task_name}_Loss', epoch_task_loss[task_name], epoch + 1)
        # epoch overall loss
        writer.add_scalar(f'{phase}_Loss', epoch_loss, epoch + 1)

    # Return epoch loss, metrics, and accuracies
    return epoch_loss, epoch_metrics, epoch_task_loss, epoch_task_acc

def train(model, dataloaders, dataset_sizes, criterions, optimizer, LR_scheduler, loss_weight, task_dict, task_describe, logger,
          local_rank=0, num_epochs=25, accum_iter_train=1, check_minibatch=5, intake_epochs=1,
          runs_path='./', writer=None, device=torch.device("cpu"), mix_precision=True):
    """Main training loop over multiple epochs with accuracy logging for classification tasks."""
    since = time.time()

    task_names, task_descriptions = setup_tasks(task_dict, task_describe)  # Set up task names and descriptions
    best_model_wts = copy.deepcopy(model.state_dict())  # Track the best model weights
    best_epoch_loss, best_epoch_idx = float('inf'), 0  # Initialize best epoch metrics
    log_dict = {}  # Dictionary to hold logging information

    # Enable mixed precision scaling if specified
    scaler = torch.amp.GradScaler(enabled=mix_precision)
    
    for epoch in range(num_epochs):
        for phase in ['Train', 'Val']:  # Run training and validation phases

            # Set sampler to make shuffle work in DDP
            if phase == 'Train' and device != torch.device("cpu"):
                dataloaders[phase].sampler.set_epoch(epoch)

            accum_iter = accum_iter_train if phase == 'Train' else 1
            epoch_loss, epoch_metrics, epoch_task_loss, epoch_task_acc = train_epoch(
                model, scaler, dataloaders[phase], optimizer, phase, criterions, device, 
                task_names, task_descriptions, loss_weight, mix_precision, accum_iter, check_minibatch, 
                logger, writer, epoch, local_rank
            )
            
            # Update scheduler after each training epoch
            if LR_scheduler and phase == 'Train':
                LR_scheduler.step()
            
            # Calculate average loss per sample
            total_samples = dataset_sizes[phase]
            avg_loss = epoch_loss / total_samples
            logger.info(f"Epoch {epoch+1}/{num_epochs} {phase} Avg. Loss: {avg_loss:.4f}")
            
            # Log accuracy for each classification task
            for task_name, acc in epoch_task_acc.items():
                logger.info(f"Epoch {epoch+1}/{num_epochs} {phase} Accuracy for {task_name}: {acc * 100:.2f}%")

            # Store log_dict
            # legacy: epoch_task_loss = Average_sample_loss, epoch_metrics = epoch_results
            epoch_results = make_epoch_results(epoch_task_loss, epoch_task_acc, epoch_metrics, task_names)
            if epoch + 1 not in log_dict:
                log_dict[epoch + 1] = {}
            log_dict[epoch + 1][phase] = {'epoch_task_loss': epoch_task_loss, 'epoch_results': epoch_results}
            
            # Update best model if validation loss improves
            if phase == 'Val' and avg_loss < best_epoch_loss and epoch + 1 >= intake_epochs:
                best_epoch_loss = avg_loss
                best_epoch_idx = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

    # Make summary
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info(f'best_epoch_idx:{best_epoch_idx}')
    for phase in log_dict[best_epoch_idx]:
        best_epoch_task_loss = log_dict[best_epoch_idx][phase]['epoch_task_loss']
        best_epoch_result = log_dict[best_epoch_idx][phase]['epoch_results']
        logger.info(
            f'In: {phase}\n'
            f'best epoch_task_loss: {best_epoch_task_loss}\n'
            f'best epoch_result: {best_epoch_result}'
        )
    
    # Save logs if running on main thread
    log_path = os.path.join(runs_path, time.strftime('%Y_%m_%d') + '_log.json')
    if local_rank == 0:
        with open(log_path, 'w') as f:
            json.dump(log_dict, f, indent=2)
        # attach the records to the tensorboard backend
        if writer:
            writer.close()

    # Load the best model weights at the end of training
    model.load_state_dict(best_model_wts)
    
    return model, log_path

def main(args):
    
    if not os.path.exists(args.save_model_path):          
        os.mkdir(args.save_model_path)
    if not os.path.exists(args.runs_path):
        os.mkdir(args.runs_path)

    # define run name
    run_name = 'MTL_' + args.model_name
    run_name = run_name + '_' + str(args.tag) if args.tag is not None else run_name

    # define output path
    save_model_path = os.path.join(args.save_model_path, run_name + '.pth')
    draw_path = os.path.join(args.runs_path, run_name)
    if not os.path.exists(draw_path):
        os.mkdir(draw_path)

    # create logger
    logger = create_logger(output_dir=draw_path, name=f"{run_name}")

    # specify device and init ddp environment
    assert args.gpu_idx == None, "gpu_idx is deprecated, please use --device."
    device, device_to_use, local_rank, logger = setup_devices_ddp(logger, device_str=args.device)

    if args.enable_tensorboard:
        writer = SummaryWriter(draw_path)
        # if u run locally
        # nohup tensorboard --logdir=/4tbB/WSIT/runs --host=0.0.0.0 --port=7777 &
        # tensorboard --logdir=/4tbB/WSIT/runs --host=0.0.0.0 --port=7777
        # python3 -m tensorboard.main --logdir=/Users/zhangtianyi/Desktop/ITH/results --host=172.31.209.166 --port=7777
    else:
        writer = None

    # filtered tasks
    task_idx_or_name_list = args.tasks_to_run.split('%') if args.tasks_to_run is not None else None

    # build task settings
    task_config_path = os.path.join(args.root_path, args.task_setting_folder_name, 'task_configs.yaml')
    # WSI_task_dict, MTL_heads_configs, WSI_criterions, loss_weight, class_num, WSI_task_describe = \
    #     task_filter_auto(Task_idx_or_name_list=task_idx_or_name_list,
    #                      task_config_path=task_config_path)
    task_name_list, WSI_task_dict, MTL_heads_configs, WSI_criterions, loss_weight, class_num, WSI_task_describe = (
        task_filter_auto(Task_idx_or_name_list=task_idx_or_name_list, task_config_path=task_config_path))
    logger.info(f'WSI_task_dict: {WSI_task_dict}')
    logger.info(f'WSI_task_describe: {WSI_task_describe}')

    # filtered tasks
    logger.info("*********************************{}*************************************".format('settings'))
    for a in str(args).split(','):
        logger.info(a)
    logger.info("*********************************{}*************************************\n".format('setting'))

    # instantiate the dataset
    Train_dataset = SlideDataset(args.root_path, args.task_description_csv,
                                 task_setting_folder_name=args.task_setting_folder_name,
                                 split_name='Train', slide_id_key=args.slide_id_key,
                                 split_target_key=args.split_target_key,
                                 max_tiles=args.max_tiles, padding=args.padding)
    Val_dataset = SlideDataset(args.root_path, args.task_description_csv,
                               task_setting_folder_name=args.task_setting_folder_name,
                               split_name='Val', slide_id_key=args.slide_id_key,
                               split_target_key=args.split_target_key,
                               max_tiles=args.max_tiles, padding=args.padding)

    # define data sampler
    if device_to_use == 'gpu':
        args.batch_size = args.batch_size // torch.cuda.device_count()
        assert args.batch_size == 1 or args.padding, "batch_size cannot > 1 unless padding=True"
        # create samplers for distributed training
        train_sampler = DistributedSampler(Train_dataset, rank=local_rank, shuffle=True)
        val_sampler = DistributedSampler(Val_dataset, rank=local_rank, shuffle=False)
        # create dataloaders using the samplers
        dataloaders = {
            'Train': torch.utils.data.DataLoader(Train_dataset, batch_size=args.batch_size,
                                                 collate_fn=MTL_collate_fn,
                                                 num_workers=args.num_workers, drop_last=True,
                                                 sampler=train_sampler),
            'Val': torch.utils.data.DataLoader(Val_dataset, batch_size=args.batch_size,
                                               collate_fn=MTL_collate_fn,
                                               num_workers=args.num_workers, drop_last=True,
                                               sampler=val_sampler)}
        # define dataset size as data num in current gpu
        dataset_sizes = {'Train': len(train_sampler), 'Val': len(val_sampler)}
    else:
        dataloaders = {
            'Train': torch.utils.data.DataLoader(Train_dataset, batch_size=args.batch_size,
                                                 collate_fn=MTL_collate_fn,
                                                 shuffle=True, num_workers=args.num_workers, drop_last=True),
            'Val': torch.utils.data.DataLoader(Val_dataset, batch_size=args.batch_size,
                                               collate_fn=MTL_collate_fn,
                                               shuffle=False, num_workers=args.num_workers, drop_last=True)}
        dataset_sizes = {'Train': len(Train_dataset), 'Val': len(Val_dataset)}

    # build model
    model = build_WSI_task_model(model_name=args.model_name, local_weight_path=args.local_weight_path,
                                 ROI_feature_dim=args.ROI_feature_dim,
                                 MTL_heads_configs=MTL_heads_configs, latent_feature_dim=args.latent_feature_dim)
    model.bfloat16().to(device) if args.turn_off_mix_precision else model.to(device)  # fixme csc check
    # fixme this have bug for gigapath in train, but ok with val, possible issue with Triton
    # model = torch.compile(model)

    if device_to_use == 'gpu':
        # model = nn.DataParallel(model)
        logger.info(f"Start running DDP on rank {local_rank}.")

        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=local_rank, broadcast_buffers=False)

    # check the number of parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)

    # cosine Scheduler by https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.num_epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    LR_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    trained_model, log_path = train(model, dataloaders, dataset_sizes, WSI_criterions, optimizer, LR_scheduler,
                                    loss_weight, WSI_task_dict, WSI_task_describe, logger, local_rank=local_rank,
                                    num_epochs=args.num_epochs, accum_iter_train=args.accum_iter_train, check_minibatch=args.check_minibatch,
                                    intake_epochs=args.intake_epochs, runs_path=draw_path, writer=writer,
                                    device=device, mix_precision=not args.turn_off_mix_precision)

    if local_rank == 0: # only work in main thread
        # save model
        torch.save(trained_model.module.state_dict() if device_to_use == 'gpu' else trained_model.state_dict(), save_model_path)
        # # print training summary
        # fixme: format different from previous, need to alter check_json_with_plot.
        # check_json_with_plot(log_path, WSI_task_dict, save_path=draw_path)


def get_args_parser():
    parser = argparse.ArgumentParser(description='MTL Training')

    # Environment parameters
    parser.add_argument('--device', default='gpu', type=str,
        help=(
            "Select the device(s) to use. Options: "
            "'cpu' to use the CPU, "
            "'gpu' to use all available GPUs, or "
            "a comma-separated list of GPU indices to use specific GPUs. "
            "Examples: '--device cpu', '--device gpu', '--device 0', '--device 0,1'."
        )
    )

    # Model tag (for example k-fold)
    parser.add_argument('--tag', default=None, type=str,
                        help='Model tag (for example 5-fold)')

    # PATH
    parser.add_argument('--root_path', default=None, type=str,
                        help='MTL dataset root')
    parser.add_argument('--local_weight_path', default=None, type=str,
                        help='local weight path')
    parser.add_argument('--save_model_path', default=ROOT_PATH/'saved_models', type=str,
                        help='save model root')
    parser.add_argument('--runs_path', default=ROOT_PATH/'runs', type=str, help='save runing results path')

    # labels
    parser.add_argument('--task_description_csv', default=None, type=str,
                        help='label csv file path')

    # Task settings
    parser.add_argument('--tasks_to_run', default=None, type=str,
                        help='tasks to run MTL, split with %, default is None with all tasks to be run')

    # Task settings and configurations for dataloaders
    parser.add_argument('--task_setting_folder_name', default='task-settings', type=str,
                        help='task-settings folder name')
    parser.add_argument('--slide_id_key', default='patient_id', type=str,
                        help='key for mapping the label')
    parser.add_argument('--split_target_key', default='fold_information', type=str,
                        help='key identifying the split information')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='dataloader num_workers')
    parser.add_argument('--max_tiles', default=10000, type=int,
                        help='max tile for loading')

    # module settings
    parser.add_argument('--latent_feature_dim', default=128, type=int,
                        help='MTL module dim')
    parser.add_argument('--slide_embed_dim', default=768, type=int,
                        help='feature slide_embed_dim , default 768')
    parser.add_argument('--ROI_feature_dim', default=1536, type=int,
                        help='feature slide_embed_dim , default 768')

    # Model settings
    parser.add_argument('--model_name', default='gigapath', type=str,
                        help='slide level model name')

    # training settings
    parser.add_argument('--padding', action='store_true',
                        help='padding all images into max_tiles, allow batch_size > 1 in single GPU')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size , default 1')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='total training epochs, default 200')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='warmup_epochs training epochs, default 50')
    parser.add_argument('--intake_epochs', default=50, type=int,
                        help='only save model at epochs after intake_epochs')
    parser.add_argument('--accum_iter_train', default=2, type=int,
                        help='training accum_iter for loss accuming, default 2')
    parser.add_argument('--lr', default=0.000001, type=float,
                        help='training learning rate, default 0.00001')
    parser.add_argument('--lrf', default=0.1, type=float,
                        help='Cosine learning rate decay times, default 0.1')
    # turn_off_mix_precision
    parser.add_argument('--turn_off_mix_precision', action='store_true',  # fixme csc check
                        help='turn_off_mix_precision for certain models like RWKV for debugging')

    # ddp related
    parser.add_argument('--local_rank', type=int, default=0, help='Current process sequence')

    # helper
    parser.add_argument('--check_minibatch', default=25, type=int,
                        help='check batch_size')
    parser.add_argument('--enable_notify', action='store_true',
                        help='enable notify to send email')
    parser.add_argument('--enable_tensorboard', action='store_true',
                        help='enable tensorboard to save status')

    # legacy
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='deprecated')

    return parser


if __name__ == '__main__':
    # setting up the random seed
    setup_seed(42)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)