"""
Tools   Script  ver： Nov 11th 10:00
"""
import os
import sys
import shutil
import torch
import logging
import functools
import numpy as np
import torch.distributed as dist
from termcolor import colored
from collections import OrderedDict

# Tools
def del_file(filepath):
    """
    clear all items within a folder
    :param filepath: folder path
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def to_2tuple(input):
    if type(input) is tuple:
        if len(input) == 2:
            return input
        else:
            if len(input) > 2:
                output = (input[0], input[1])
                return output
            elif len(input) == 1:
                output = (input[0], input[0])
                return output
            else:
                print('cannot handle none tuple')
    else:
        if type(input) is list:
            if len(input) == 2:
                output = (input[0], input[1])
                return output
            else:
                if len(input) > 2:
                    output = (input[0], input[1])
                    return output
                elif len(input) == 1:
                    output = (input[0], input[0])
                    return output
                else:
                    print('cannot handle none list')
        elif type(input) is int:
            output = (input, input)
            return output
        else:
            print('cannot handle ', type(input))
            raise ('cannot handle ', type(input))


def find_all_files(root, suffix=None):
    """
    Return a list of file paths ended with specific suffix
    """
    res = []
    if type(suffix) is tuple or type(suffix) is list:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None:
                    status = 0
                    for i in suffix:
                        if not f.endswith(i):
                            pass
                        else:
                            status = 1
                            break
                    if status == 0:
                        continue
                res.append(os.path.join(root, f))
        return res

    elif type(suffix) is str or suffix is None:
        for root, _, files in os.walk(root):
            for f in files:
                if suffix is not None and not f.endswith(suffix):
                    continue
                res.append(os.path.join(root, f))
        return res

    else:
        print('type of suffix is not legal :', type(suffix))
        return -1


# Transfer state_dict by removing misalignment
def FixStateDict(state_dict, remove_key_head=None):
    """
    Obtain a fixed state_dict by removing misalignment

    :param state_dict: model state_dict of OrderedDict()
    :param remove_key_head: the str or list of strings need to be remove by startswith
    """

    if remove_key_head is None:
        return state_dict

    elif type(remove_key_head) == str:
        keys = []
        for k, v in state_dict.items():
            if k.startswith(remove_key_head):  # 将‘arc’开头的key过滤掉，这里是要去除的层的key
                continue
            keys.append(k)

    elif type(remove_key_head) == list:
        keys = []
        for k, v in state_dict.items():
            jump = False
            for a_remove_key_head in remove_key_head:
                if k.startswith(a_remove_key_head):  # 将‘arc’开头的key过滤掉，这里是要去除的层的key
                    jump = True
                    break
            if jump:
                continue
            else:
                keys.append(k)
    else:
        print('erro in defining remove_key_head !')
        return -1

    new_state_dict = OrderedDict()
    for k in keys:
        new_state_dict[k] = state_dict[k]
    return new_state_dict


def setup_seed(seed):  # setting up the random seed
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


### GPU related funcs ###

@functools.lru_cache()
def create_logger(output_dir, dist_rank=-1, name=''):
    # --------------------------------------------------------
    # Initialize logger, only print output from rank=0 (main thread)
    # References:
    # SimMIM: https://github.com/microsoft/SimMIM
    # --------------------------------------------------------

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def setup_devices_ddp(logger, device_str='gpu'):

    # find device based on input str
    if device_str.lower() == 'cpu':
        logger.info("Using CPU.")
        device_to_use = 'cpu'
    elif device_str.lower() == 'gpu':
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Using all available GPUs ({gpu_count} GPUs).")
            device_to_use = 'gpu'
        else:
            logger.warning("GPU requested but no GPUs are available. Falling back to CPU.")
            device_to_use = 'cpu'
    else:
        try:
            # Try to parse comma-separated GPU indices
            gpu_indices = [int(idx.strip()) for idx in device_str.split(',')]
            if all(idx >= 0 and idx < torch.cuda.device_count() for idx in gpu_indices):
                logger.info(f"Using specified GPU indices: {gpu_indices}.")
                logger.warn(f"Warning: speficy GPU in python script may not work properly! \
                            Please consider specify in shell script.")
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))
                device_to_use = 'gpu'
            else:
                raise ValueError("Invalid GPU indices provided.")
        except Exception as e:
            logger.error(f"Invalid device specification '{device_str}': {e}")
            logger.info("Falling back to CPU.")
            device_to_use = 'cpu'

    # specify cpu or gpu device
    if device_to_use == 'cpu':
        device = torch.device('cpu')
        local_rank = 0
    elif device_to_use == 'gpu':
        # setup ddp
        torch.distributed.init_process_group("nccl")
        local_rank = torch.distributed.get_rank()
        device = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device)

    # create console handlers for master process
    if local_rank == 0:
        color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    logger.info(f'using device: {device_to_use}')

    return device, device_to_use, local_rank, logger


def variable_sum_ddp(target_var, device='cuda', avg=False):
    # print(f'[rank{device}] target_var: {target_var}, type: {type(target_var)}')
    target_var_tensor = torch.tensor(target_var, device=device)
    # print(f'[rank{device}] target_var_tensor: {target_var_tensor}, type: {type(target_var_tensor)}')
    dist.barrier()  # make sure all processes are synced, otherwise all_reduce may get wrong result
    dist.all_reduce(target_var_tensor, op=dist.ReduceOp.SUM)
    if avg:
        target_var_sum = target_var_tensor.item() / torch.distributed.get_world_size()
    else:
        target_var_sum = target_var_tensor.item()
    # print(f'[rank{device}] all_reduced target_var_sum: {target_var_sum}, type: {type(target_var_sum)}')
    return target_var_sum