"""
MTL Task settings   Script  ver： Feb 2nd 17:00

flexible to multiple-tasks and missing labels
"""

import os
import json
import copy
import numpy as np
import torch.nn as nn
import yaml  # Ensure pyyaml is installed: pip install pyyaml


def build_all_tasks(task_config_path=None):
    try:
        # todo: consider merging task_to_run with all_task_dict
        with open(task_config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.Loader)
            all_task_name_list = config.get('tasks_to_run')  # config.get('all_task_dict').keys()
            all_task_dict = config.get('all_task_dict')
            all_task_one_hot_describe = config.get('one_hot_table')
            # load manual loss design for all models, default to None for no config
            all_criterions_name_dic = config.get('all_criterions', None)
            # a name list of ['L1Loss','CrossEntropyLoss', 'MSELoss', ...]

    except:
        raise  # no task-settings folder for the dataset
    else:
        # Generate task_idx_to_name, task_name_to_idx, all_class_num, all_loss_weight, all_criterions
        idx = 0
        task_name_to_idx = {}  # task name to idx list
        task_idx_to_name = []  # task idx to name list

        all_class_num = []  # class number list
        all_loss_weight = []  # todo: need to allow manually config in the future, maybe config in yaml file?
        all_criterions_name_list = []  # task loss func name

        all_MTL_heads_configs = []  # MTL_heads_configs: a list with task output dimensions for each task

        for task in all_task_dict:
            # if task is regression task
            if all_task_dict[task] == 'float':
                all_task_dict[task] = float
                all_class_num.append(0)
                if all_criterions_name_dic is not None:
                    all_criterions_name_list.append(all_criterions_name_dic[task])
                else:
                    all_criterions_name_list.append('L1Loss')
                all_MTL_heads_configs.append(1)
            # if task is cls task
            elif all_task_dict[task] == 'list':
                all_task_dict[task] = list
                all_class_num.append(len(all_task_one_hot_describe[task]))
                if all_criterions_name_dic is not None:
                    all_criterions_name_list.append(all_criterions_name_dic[task])
                else:
                    all_criterions_name_list.append('CrossEntropyLoss')
                # pred (type: float): [Batch, cls], GT (type: long int): [Batch]
                # each label long-int is a scaler, after stacking, it becomes [Batch]
                all_MTL_heads_configs.append(int(all_class_num[idx]))
            else:
                raise ValueError('Not valid data type!')

            task_name_to_idx[task] = idx
            task_idx_to_name.append(task)
            all_loss_weight.append(1.0)
            idx += 1
        # build loss for tasks
        # ensure the loss func name list is correct
        assert len(all_criterions_name_list) == len(all_task_dict)
        all_criterions = []
        for criterions_name in all_criterions_name_list:
            if criterions_name == 'CrossEntropyLoss':
                all_criterions.append(nn.CrossEntropyLoss())
            elif criterions_name == 'L1Loss':
                all_criterions.append(nn.L1Loss(size_average=None, reduce=None))
            elif criterions_name == 'MSELoss':
                all_criterions.append(nn.MSELoss(size_average=None, reduce=None))
            else:
                raise NotImplementedError

    return (all_task_name_list, task_name_to_idx, task_idx_to_name, all_task_dict, all_MTL_heads_configs,
            all_criterions, all_loss_weight, all_class_num, all_task_one_hot_describe)


def task_filter_auto(Task_idx_or_name_list=None, task_config_path=None):
    """
    Auto task filter defined by json files

    Args:
        task_config_path (str): task_settings_path/task_configs.yaml
        latent_feature_dim (int, optional): _description_. Defaults to 768.

    Raises:
        ValueError: _description_

    Returns:
        task_name_list: the name of the tasks defined by input index/names, this is the order of the tasks
        task_dict: task settings dictionary for tracking its CLS (list) or regression (float) task
        MTL_heads_configs: a list with task output dimensions for each task, a list
        criterions,: loss function for each tasks, a list
        loss_weight: penalty settings for each task (currently the settings is all 1)
        class_num: CLS number for each CLS task, a list
        task_describe: one-hot label settings for each CLS task
    """
    assert task_config_path is not None  # the task_config should not be none

    (all_task_name_list, task_name_to_idx, task_idx_to_name, all_task_dict, all_MTL_heads_configs, all_criterions,
     all_loss_weight, all_class_num, all_task_one_hot_describe) = build_all_tasks(task_config_path)

    # building running config according to WSI_task_idx_or_name_list(support both index and text)
    if Task_idx_or_name_list is None:
        Task_idx_or_name_list = all_task_name_list

    if type(Task_idx_or_name_list[0]) == int:
        # the specification is tasks index
        task_idx_list = Task_idx_or_name_list
    else:
        # the specification is tasks name
        for task in Task_idx_or_name_list:
            assert task in all_task_name_list, f"task {task} not in yaml config"
        task_idx_list = [task_name_to_idx[task] for task in Task_idx_or_name_list]

    # build tasks
    task_dict = {}
    MTL_heads_configs = []
    criterions = []
    loss_weight = []
    class_num = []
    task_describe = {}
    task_name_list = []  # this is the ordering of tasks

    for idx in task_idx_list:
        task_name_list.append(task_idx_to_name[idx])
        task_dict[task_idx_to_name[idx]] = all_task_dict[task_idx_to_name[idx]]
        MTL_heads_configs.append(all_MTL_heads_configs[idx])
        criterions.append(all_criterions[idx])
        loss_weight.append(all_loss_weight[idx])
        class_num.append([all_class_num[idx]])
        if task_idx_to_name[idx] in all_task_one_hot_describe:  # if its cls task
            task_describe[task_idx_to_name[idx]] = all_task_one_hot_describe[task_idx_to_name[idx]]

    return task_name_list, task_dict, MTL_heads_configs, criterions, loss_weight, class_num, task_describe


# base on the task dict to convert the task idx of model output
class task_idx_converter:
    def __init__(self, old_task_dict, new_task_dict):
        idx_dict = {}
        idx_project_dict = {}
        for old_idx, key in enumerate(old_task_dict):
            idx_dict[key] = old_idx
        for new_idx, key in enumerate(new_task_dict):
            idx_project_dict[new_idx] = idx_dict[key]

        self.idx_project_dict = idx_project_dict

    def __call__(self, new_idx):
        back_to_old_idx = self.idx_project_dict[new_idx]
        return back_to_old_idx


def listed_onehot_dic_to_longint_name_dic(name_onehot_list_dic):
    """
    converting name_onehot_list_dic to longint_name_dic
    Example
    name_onehot_list_dic = {'3RD': [0, 0, 0, 0, 1],
                            '4TH': [0, 0, 0, 1, 0],
                            '5TH': [0, 0, 1, 0, 0],
                            '6TH': [0, 1, 0, 0, 0],
                            '7TH': [1, 0, 0, 0, 0]}

    out
    longint_name_dic = {4: '3RD', 3: '4TH', 2: '5TH', 1: '6TH', 0: '7TH'}
    """

    longint_name_dic = {}

    for cls_name in name_onehot_list_dic:
        listed_onehot = name_onehot_list_dic[cls_name]
        long_int = np.array(listed_onehot).argmax()
        longint_name_dic[long_int] = cls_name

    return longint_name_dic


class result_recorder:
    def __init__(self, task_dict, task_describe, batch_size=1, total_size=10, runs_path=None):
        assert runs_path is not None
        self.runs_path = runs_path

        self.task_dict = task_dict
        self.batch_size = batch_size
        self.total_size = total_size

        # set up the indicators
        self.longint_to_name_dic_for_all_task = {}
        self.task_template = {}
        self.task_idx_to_key = {}
        task_idx = 0

        for key in task_dict:
            self.task_template[key] = {"pred": None, "label": None}

            self.task_idx_to_key[task_idx] = key
            task_idx += 1

            if task_dict[key] == list:
                self.longint_to_name_dic_for_all_task[key] = listed_onehot_dic_to_longint_name_dic(task_describe[key])

        # set up the sample list
        self.record_samples = {}
        self.data_iter_step = 0

    def add_step(self, data_iter_step):
        self.data_iter_step = data_iter_step
        for i in range(self.batch_size):
            record_idx = data_iter_step * self.batch_size + i
            if record_idx == self.total_size:
                break  # now at the next of last sample
            else:
                self.record_samples[record_idx] = copy.deepcopy(self.task_template)

    def add_data(self, batch_idx, task_idx, pred, label=None, pred_one_hot=None, label_one_hot=None):
        record_idx = self.data_iter_step * self.batch_size + batch_idx

        if pred_one_hot is not None and label_one_hot is not None:
            pred_one_hot = pred_one_hot.tolist()
            label_one_hot = label_one_hot.tolist()

        key = self.task_idx_to_key[task_idx]
        # rewrite template
        if self.task_dict[key] == list:
            self.record_samples[record_idx][key]["pred"] = self.longint_to_name_dic_for_all_task[key][pred]
            self.record_samples[record_idx][key]["label"] = \
                self.longint_to_name_dic_for_all_task[key][label] if label is not None else None
            self.record_samples[record_idx][key]["pred_one_hot"] = pred_one_hot
            self.record_samples[record_idx][key]["label_one_hot"] = label_one_hot
        else:  # reg
            self.record_samples[record_idx][key]["pred"] = pred
            self.record_samples[record_idx][key]["label"] = label if label is not None else None

    def finish_and_dump(self, tag='Test_'):
        # save json_log  indent=2 for better view
        log_path = os.path.join(self.runs_path, str(tag) + 'predication.json')
        json.dump(self.record_samples, open(log_path, 'w'), ensure_ascii=False, indent=2)
