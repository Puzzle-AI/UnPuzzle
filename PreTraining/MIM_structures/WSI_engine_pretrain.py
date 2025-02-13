"""
Training Engine   Script  ver： Nov 11th 14:00

Based on MAE code.
https://github.com/facebookresearch/mae

"""
import os
import sys
from pathlib import Path
import PreTraining.MIM_structures.misc as misc
# For convenience, import all path to sys
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir))
sys.path.append(str(this_file_dir.parent))
sys.path.append(str(this_file_dir.parent.parent))
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

import math
from typing import Iterable

import torch
import PreTraining.MIM_structures.misc
import Utils.schedulers as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, fix_position_ratio_scheduler=None,
                    print_freq=20, log_writer=None, args=None):
    # puzzle_patch_size_scheduler=None,
    model.train(True)

    # update logger
    metric_logger = misc.MetricLogger(delimiter="  ")
    # 初始化学习率记录
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:  # Tensorboard PATH
        print('log_dir: {}'.format(args.log_dir))

    # Iteration
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # jump the batch if it cannot correct by WSI_collate_fn in dataloader
        if samples is None:
            # failed_sample_count += dataloader.batch_size
            continue
        else:
            # take data and task_description_list from sample
            image_features, coords, task_description_list, slide_ids = samples
            # image_features is a tensor of [B,N,D],  coords_yx is tensor of [B,N,2]
            image_features = image_features.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)
            # task_description_list [task, batch_size] batch-stacked tensors, element of long-int or float

        # per iteration lr scheduler基于中间epoch位置
        # 来实现更精确的调节学习率：data_iter_step / len(data_loader) + epoch
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with (torch.cuda.amp.autocast()):  # 使用自动混合精度加速训练

            fix_position_ratio = fix_position_ratio_scheduler(
                epoch) if fix_position_ratio_scheduler is not None else None
            '''
            puzzle_patch_size = puzzle_patch_size_scheduler(
                epoch) if puzzle_patch_size_scheduler is not None else None
            '''

            if args.SSL_name.split('_')[0] == 'MAE':
                mask_ratio = 1 - fix_position_ratio if fix_position_ratio is not None else args.mask_ratio
                loss, pred, mask_patch_indicators = model(image_features, coords, mask_ratio=mask_ratio)
                # fixme mae curriculum maybe not good enough for future

            else:
                raise NotImplementedError
            '''
            elif args.SSL_name.split('_')[0] == 'SAE':
                loss, pred, imgs_puzzled_patches = model(image_features, coords,
                                                         fix_position_ratio=fix_position_ratio,
                                                         puzzle_patch_size=puzzle_patch_size)  # SAE
            '''

        if args.DDP_distributed:
            loss_value = loss.item()
        else:
            loss_value = float(loss.cpu().detach().numpy()) \
                if torch.cuda.device_count() == 1 else sum(loss.cpu().detach().numpy())

        if not math.isfinite(loss_value):  # 检查确保没有loss爆炸
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter  # 计算的是每个minibatch的loss，如果有梯度累加则需要减少占比，loss在loss_scaler里面会进行叠加

        # loss backward 核心（不要怕，其实就是功能上集成了loss.backward+opt.step，然后引入了梯度裁剪)
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()  # 等待当前设备上所有流中的所有核心完成

        # 更新记录
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # 计算平均在单卡上的loss
        loss_value_reduce = misc.all_reduce_mean(loss_value)

    if log_writer is not None:
        log_writer.add_scalar('train_loss', loss_value_reduce, epoch)
        log_writer.add_scalar('lr', lr, epoch)

        if fix_position_ratio is not None:
            log_writer.add_scalar('fix_position_ratio', fix_position_ratio, epoch)
        '''
        if puzzle_patch_size is not None:
            log_writer.add_scalar('puzzle_patch_size', puzzle_patch_size, epoch)
        '''

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if fix_position_ratio is not None:
        print('fix_position_ratio:', fix_position_ratio)
    '''
    if puzzle_patch_size is not None:
        print('puzzle_patch_size:', puzzle_patch_size)
    '''

    # 返回记录，其他的已经在对象内迭代
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
