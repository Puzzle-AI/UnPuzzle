"""
Slide MAE Training   Script  ver： Dec 5th 17:00

# References:
Based on PuzzleTuning code base
https://github.com/sagizty/PuzzleTuning

Based on MAE code.
https://github.com/facebookresearch/mae

"""
import os
import sys
from pathlib import Path

# For convenience, import all path to sys
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir))
sys.path.append(str(this_file_dir.parent))
sys.path.append(str(this_file_dir.parent.parent))
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

import argparse
import datetime
import json
import numpy as np

import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import PreTraining.MIM_structures.misc as misc
from PreTraining.MIM_structures.misc import NativeScalerWithGradNormCount as NativeScaler
from Utils.schedulers import ratio_scheduler

from DataPipe.dataset_framework import SlideDataset,MTL_collate_fn
from PreTraining.MIM_structures.WSI_MIM_model import build_WSI_MIM_model
from PreTraining.MIM_structures.WSI_engine_pretrain import train_one_epoch


def add_weight_decay(model, weight_decay, skip_list=("bias", "LayerNorm.weight", "norm.weight")):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Ignore frozen parameters
        if any(skip in name for skip in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ]

def main(args):
    # choose encoder for timm
    slide_model_name = args.model_name

    # choose decoder version
    args.SSL_name = args.SSL_name + '_decoder' if args.dec_idx is not None else args.SSL_name
    
    # set args.model_idx to trace the experiment in misc out output dir
    args.model_idx = args.model_idx + args.model_name+ '_' + args.SSL_name + '_' + args.dec_idx \
        if args.dec_idx is not None else args.model_idx + args.model_name+ '_' + args.SSL_name

    # fix the seed for reproducibility
    if args.DDP_distributed:
        misc.init_distributed_mode(args)
        seed = args.seed + misc.get_rank()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set GPUs
    cudnn.benchmark = True
    device = torch.device(args.device)  # cuda

    # instantiate the dataset
    dataset_train = SlideDataset(args.tile_embedding_path, task_type='embedding', max_tiles=args.max_tiles)
    print('dataset_train:', dataset_train)  # Train data

    if args.DDP_distributed:  # args.DDP_distributed is True we use distributed data parallel(DDP)
        num_tasks = misc.get_world_size()  # use misc to set up DDP
        global_rank = misc.get_rank()  # get the rank of the current running

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))
        enable_DistributedSampler = True
        # fixme for slide we can only use 1 not args.batch_size * torch.cuda.device_count()
        batch_size_for_Dataloader = 1

    else:  # Data parallel(DP) instead of distributed data parallel(DDP)
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        enable_DistributedSampler = False
        # fixme for slide we can only use 1 not args.batch_size * torch.cuda.device_count()
        batch_size_for_Dataloader = 1

    # set log on the main process
    if global_rank == 0 and args.log_dir is not None:
        '''
        if not args.disable_notify:
            import notifyemail as notify
            notify.Reboost(mail_host='smtp.163.com', mail_user='tum9598@163.com', mail_pass='xxxxx',
                           default_reciving_list=['tum9598@163.com'],  # change here if u want to use notify
                           log_root_path='log', max_log_cnt=5)
            notify.add_text('Slide VisionSSL log:\n' + args.model_idx)
            notify.add_text("{}".format(args).replace(', ', ',\n'))
            notify.send_log()
        '''

        # setting up logging
        args.log_dir = os.path.join(args.log_dir, args.model_idx)
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)  # Tensorboard

        print('Task: ' + args.model_idx)
        print("Use", torch.cuda.device_count(), "GPUs!")
        print('job AImageFolderDir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(', ', ',\n'))
    else:
        log_writer = None

    # output_dir
    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, args.model_idx)
        os.makedirs(args.output_dir, exist_ok=True)
        print('Training output files will be at', args.output_dir)
    else:
        print('no out put path specified!')
        raise

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,  # the shuffle=True is already set in the sampler
        collate_fn=MTL_collate_fn,
        batch_size=batch_size_for_Dataloader,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)

    # define the curriculum learning schedulers
    if args.strategy is not None:
        # strategy=None for fixed else reduce ratio gradually
        fix_position_ratio_scheduler = ratio_scheduler(total_epoches=args.epochs,
                                                       warmup_epochs=args.warmup_epochs,
                                                       basic_ratio=0.25,  # start ratio
                                                       fix_position_ratio=args.fix_position_ratio,  # None
                                                       strategy=args.strategy)  # default 'loop'
        # todo curriculum designs for better convergence
    else:
        fix_position_ratio_scheduler = None

    # the effective batch size for setting up lr
    if args.DDP_distributed:
        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    else:
        eff_batch_size = args.batch_size * torch.cuda.device_count() * args.accum_iter
    print('eff_batch_size:', eff_batch_size)

    if args.lr is None:  # when only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # take the model parameters for optimizer update
    model = build_WSI_MIM_model(SSL_name=args.SSL_name, model_name=slide_model_name, dec_idx=args.dec_idx,
                                MTL_token_num=args.MTL_token_num,
                                local_weight_path=args.local_weight_path,
                                ROI_feature_dim=args.ROI_feature_dim,
                                norm_pix_loss=args.norm_pix_loss)
    model_without_ddp = model
    
    if args.DDP_distributed:
        model.cuda()  # args.gpu is obtained by misc.py
        # find_unused_parameters=True for the DDP to correctly synchronize layers in back propagation
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.to(device)

    print("Model = %s" % str(model_without_ddp))

    # following timm: set wd as 0 for bias and norm layers
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # loss scaler with gradient clipping
    loss_scaler = NativeScaler(GPU_count=torch.cuda.device_count(), DDP_distributed=args.DDP_distributed)

    # if we have --resume，we will load the checkpoint and continue training, if not, we start a new training
    # the checkpoint should include model, optimizer, loss_scaler information
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Training by epochs
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # use args.start_epoch to jump to resume checkpoint

        if enable_DistributedSampler:  # DistributedSampler need to .set_epoch(epoch) at each epoch
            data_loader_train.sampler.set_epoch(epoch)

        # training iterations
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler,
                                      fix_position_ratio_scheduler=fix_position_ratio_scheduler,
                                      print_freq=args.print_freq, log_writer=log_writer, args=args)

        if args.output_dir and (epoch % args.check_point_gap == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, model_idx=args.model_idx)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        # Write log
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # time stamp
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_args_parser():
    parser = argparse.ArgumentParser('SAE pre-training', add_help=False)

    # disable_notify
    parser.add_argument('--disable_notify', action='store_true', help='do not send email of tracking')

    # Model Name or index
    parser.add_argument('--model_idx', default='Slide_VisionSSL_Tuning_', 
                        type=str, help='experiment index')
    parser.add_argument('--model_name', default='SlideViT',
                        type=str, help='Slide backbone model name')
    # Model config
    parser.add_argument("--local_weight_path", default=None, type=str, help="local weight path")
    parser.add_argument("--ROI_feature_dim", default=1536, type=int,
                        help="the feature dim of the embedded tiles")

    # MIM Model parameters  sae_vit_base_patch16  mae_vit_base_patch16
    parser.add_argument('--SSL_name', default='MAE', type=str,
                        help='Name of SSL framework to train')  # ori mae_vit_large_patch16
    parser.add_argument('--dec_idx', default=None, type=str, metavar='segmentation decoder',
                        help='Name of segmentation decoder')

    # fixme for slide we can only use batch = 1 to encounter slides with different size (for now)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations '
                             '(for increasing the effective batch size under memory constraints)')

    # if we have --resume，we will load the checkpoint and continue training, if not, we start a new training
    # the checkpoint should include model, optimizer, loss_scaler information
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch of checkpoint')

    # MAE mask_ratio
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches)')
    parser.add_argument('--MTL_token_num', default=0, type=int,
                        help='MTL_token_num')

    # Curriculum learning Tuning settings
    parser.add_argument('--strategy', default=None, type=str,
                        help='use linear or other curriculum scheduler')  # fixme here only 'loop' for now
    parser.add_argument('--fix_position_ratio', default=None, type=float,
                        help='ablation fix_position_ratio (percentage of position token patches)')

    # loss settings
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-slide_feature) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer settings
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr), default=None')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * effective batch size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # PATH settings
    parser.add_argument('--tile_embedding_path',
                        default='/data/hdd_1/BigModel/TCGA-LUAD-LUSC/Tile_embeddings/gigapath',
                        type=str, help='the dataset path of the embedded tiles')
    parser.add_argument('--output_dir', default='/data/hdd_1/BigModel/runs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/data/hdd_1/BigModel/runs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    # dataloader setting
    parser.add_argument('--max_tiles', default=None, type=str,
                        help='set max_tiles for loading, default None will try to load task config or 10000')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # print_freq and checkpoint
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--check_point_gap', default=50, type=int)
    parser.add_argument('--check_samples', default=1, type=int, help='check how many images in a checking batch')

    # DDP_distributed training parameters for DDP
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of DDP_distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up DDP_distributed training')
    parser.add_argument('--DDP_distributed', action='store_true', help='Use DDP in training. '
                                                                       'without calling, DP with be applied')

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
