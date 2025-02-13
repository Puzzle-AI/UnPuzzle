'''
MTL training panel       Script  ver： Nov 06th 00:00

demo from gigapath, cannot run!
'''
import os
import torch
import pandas as pd
import numpy as np

from training_tools import train
from params import get_finetune_params
from DownStream.MTL.task_configs import load_task_config
from DataPipe.Slide_dataset_tools import SlideDataset, load_task_settings
from utils import seed_torch, get_exp_code, get_loader


if __name__ == '__main__':
    args = get_finetune_params()
    print(args)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # set the random seed
    seed_torch(device, args.seed)

    # load the task configuration
    print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config = load_task_config(args.task_cfg_path)
    print(args.task_config)
    args.task = args.task_config.get('name', 'task')

    # set the experiment save directory
    args.save_dir = os.path.join(args.save_dir, args.task, args.exp_name)
    args.model_code, args.task_code, args.exp_code = get_exp_code(args) # get the experiment code
    args.save_dir = os.path.join(args.save_dir, args.exp_code)
    os.makedirs(args.save_dir, exist_ok=True)
    print('Experiment code: {}'.format(args.exp_code))
    print('Setting save directory: {}'.format(args.save_dir))

    # set the learning rate
    eff_batch_size = args.batch_size * args.gc
    if args.lr is None or args.lr < 0:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.gc)
    print("effective batch size: %d" % eff_batch_size)

    # set the split key
    if args.pat_strat:
        args.split_target_key = 'pat_id'
    else:
        args.split_target_key = 'slide_id'

    # set up the dataset
    args.split_dir = os.path.join(args.split_dir, args.task_code) if not args.pre_split_dir else args.pre_split_dir
    os.makedirs(args.split_dir, exist_ok=True)
    print('Setting split directory: {}'.format(args.split_dir))
    # set up the results dictionary
    results = {}

    # start cross validation
    for fold in range(args.folds):
        # get the splits
        split_target_key = 'split_{}fold-{}'.format(args.folds,fold+1) if args.folds > 1 else 'split'
        # 'split_nfold-k', n is the total fold number and k is the fold index
        # set up the fold directory
        save_dir = os.path.join(args.save_dir, f'fold_{fold}') if args.fold > 1 else args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        # instantiate the dataset
        train_data, val_data, test_data = (SlideDataset(args.root_path, task_setting_folder_name='task_settings',
                                                split_name='train', split_target_key=split_target_key),
                                           SlideDataset(args.root_path, task_setting_folder_name='task_settings',
                                              split_name='val', split_target_key=split_target_key),
                                           SlideDataset(args.root_path, task_setting_folder_name='task_settings',
                                               split_name='test', split_target_key=split_target_key))
        # todo config MTL
        task_name_list = train_data.task_name_list
        # build heads? fixme
        # get the dataloader
        train_loader, val_loader, test_loader = get_loader(train_data, val_data, test_data, **vars(args))
        # start training
        val_records, test_records = train((train_loader, val_loader, test_loader), fold, args)

        # update the results
        records = {'val': val_records, 'test': test_records}
        for record_ in records:
            for key in records[record_]:
                if 'prob' in key or 'label' in key:
                    continue
                key_ = record_ + '_' + key
                if key_ not in results:
                    results[key_] = []
                results[key_].append(records[record_][key])

    # save the results into a csv file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.save_dir, 'summary.csv'), index=False)

    # print the results, mean and std
    for key in results_df.columns:
        print('{}: {:.4f} +- {:.4f}'.format(key, np.mean(results_df[key]), np.std(results_df[key])))
    print('Results saved in: {}'.format(os.path.join(args.save_dir, 'summary.csv')))
    print('Done!')
