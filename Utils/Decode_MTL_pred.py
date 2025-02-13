"""
correlation decoding    Script  ver: Feb 8th 01:00

This support wsi and cell level tasks
and it support both cls and reg tasks

the output will be a csv file in the run folder
"""
import json
import os
import sys
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

# For convenience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels
import numpy as np
from collections import OrderedDict, defaultdict
import pandas as pd
import argparse
import yaml


def calculate_cls_auc(pred_one_hot, label_one_hot, average='micro'):
    """
    Calculate micro AUC (Area Under the Curve) for multi-class classification.

    Args:
        pred_one_hot (list of ndarray): A list of predicted probabilities in one-hot format, 
                                        where each element is an array of shape (batch_size, n_classes).
        label_one_hot (list of ndarray): A list of true labels in one-hot format,
                                         where each element is an array of shape (batch_size, 1).
        average (string): 'micro' or 'macro'.
            If 'micro' then calculate globally and ignore confidence. 
            If 'macro' then calculate with confidence, but at least one sample for each class must exist in test set.
    Returns:
        float: The AUC score.
    """
    # Calculate micro auc
    pred_one_hot = np.concatenate(pred_one_hot, axis=0)
    label_one_hot = np.concatenate(label_one_hot, axis=0)

    # # Do softmax if shape not match
    # if pred_one_hot.shape[0] != label_one_hot.shape:
    #     pred_one_hot = np.argmax(pred_one_hot, axis=1)

    if average == 'micro':
        # Convert to one hot, shape: (n_samples, n_classes)
        label_one_hot = label_one_hot.reshape(-1, 1)
        label_one_hot = OneHotEncoder(
            sparse_output=False, 
            categories=[np.arange(pred_one_hot.shape[1])]
        ).fit_transform(label_one_hot)
    
    # Calculate the AUC score using roc_auc_score.
    # `average='micro'` computes metrics globally by considering each element of the label matrix.
    # `multi_class='ovr'` indicates "one-vs-rest" strategy for multi-class AUC calculation.
    auc = roc_auc_score(label_one_hot, pred_one_hot, average=average, multi_class='ovr')
    return auc


def calculate_cls_acc(labels, predictions, one_hot_encoding_dict):
    """

    :param labels: a list of str label for all samples, some can be None or null when they are missing
    :param predictions:  a list of str label for all samples, this is the predicated output
    :param one_hot_encoding_dict: its all_task_one_hot_describe[task]: dic recording the one-hot rules for all cls task
    :return:
    """
    # Count non-missing labels (excluding None and "null")
    total_samples = len([label for label in labels if label is not None and label != "null"])
    task_metrics = defaultdict(lambda: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0})

    # Iterate over each task in all_task_one_hot_describe
    # Collect all classes for this task
    classes = list(one_hot_encoding_dict.keys())

    for true_label, predicted_label in zip(labels, predictions):
        if true_label is None or true_label == "null":
            continue  # Skip missing labels

        for class_label in classes:
            if true_label == class_label and predicted_label == class_label:
                task_metrics[class_label]['TP'] += 1
            elif true_label == class_label and predicted_label != class_label:
                task_metrics[class_label]['FN'] += 1
            elif true_label != class_label and predicted_label == class_label:
                task_metrics[class_label]['FP'] += 1
            elif true_label != class_label and predicted_label != class_label:
                task_metrics[class_label]['TN'] += 1

    # Calculate overall accuracy
    correct_predictions = sum(metrics['TP'] for metrics in task_metrics.values())
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    # Reporting per-class metrics
    for class_label, metrics in task_metrics.items():
        print(
            f"Class {class_label} - TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")

    return accuracy


def main(args):
    run_name = 'MTL_' + args.model_name
    run_name = run_name + '_' + str(args.tag) if args.tag is not None else run_name

    # we use the run name as a warp folder for both Train and Test
    assert os.path.exists(os.path.join(args.runs_path, run_name))
    experiment_run_path = os.path.join(args.runs_path, run_name, 'Test')

    # WSI level task output
    WSI_task_output = os.path.join(experiment_run_path, 'test_predication.json')
    # Cell level task output
    Cell_task_output = None  # todo future add the cell level task (need to do coding in test.py)

    # sample JSON log data of test_predication.json
    json_log = '''
    {
      "0": {  # the number is the sample index
      # there are n keys, indicating each task of a sample
        "C1QA": {  # the regression task
          "pred": 72.21339416503906,
          "label": 133.22122192382812
        },
        "EPCAM": {  # the regression task
          "pred": 72.21339416503906,
          "label": 133.22122192382812
        },
        "iCMS": {  # the classification task
          "pred": "iCMS2",
          "label": "iCMS2"
        },
        "CMS": {  # the classification task
          "pred": "CMS2",
          "label": "CMS4"
        },
        ...
      },
      "1": {
        "C1QA": {
          "pred": 36.20330810546875,
          "label": 31.104581832885742
        },...
      },...
    }
    '''

    check_path_list = {}
    check_task_list = {}

    if args.WSI_tasks:  # true(taking all tasks in config) or not none (manual tasks)
        if os.path.exists(os.path.join(args.data_path, args.task_setting_folder_name)):
            # build task settings
            task_config_path = os.path.join(args.data_path, args.task_setting_folder_name, 'task_configs.yaml')

            with open(task_config_path, 'r') as file:
                config = yaml.load(file, Loader=yaml.Loader)
                task_idx_or_name_list = config.get('tasks_to_run')
                WSI_all_task_dict = config.get('all_task_dict')
                all_task_one_hot_describe = config.get('one_hot_table')
            print('WSI_task_dict', WSI_all_task_dict)
        else:
            print('the current WSI_task_settings_path is not there:',
                  os.path.join(args.data_path, args.task_setting_folder_name))
            raise  # cannot load WSI_task_settings_path

        # confirm the tasks
        if type(args.WSI_tasks) is not bool:
            WSI_task_list = args.WSI_tasks.split('%')
            WSI_task_dict = OrderedDict((task, WSI_all_task_dict[task]) for task in WSI_task_list)
        else:
            WSI_task_dict = WSI_all_task_dict

        print('WSI_task_dict', WSI_task_dict)

        check_path_list['WSI_task'] = WSI_task_output
        check_task_list['WSI_task'] = WSI_task_dict

    if args.Cell_tasks is True:
        Cell_task_list = None  # todo
        Cell_task_dict = OrderedDict((task, 'float') for task in Cell_task_list)
        print('Cell_task_dict', Cell_task_dict)
        check_path_list['Cell_task'] = Cell_task_output
        check_task_list['Cell_task'] = Cell_task_dict

    for task_key in check_path_list:  # WSI or Cell

        test_file_path = check_path_list[task_key]
        if test_file_path is None:
            continue

        df = pd.DataFrame(index=check_task_list[task_key], columns=['corr', 'acc', 'auc'])

        with open(test_file_path) as f:
            task_name = os.path.split(os.path.split(test_file_path)[0])[1]
            print('In :', task_name)
            load_dict = json.load(f)

            for key, task_type in check_task_list[task_key].items():
                # Extract labels and predictions for each sample
                labels = []
                predictions = []
                preds_one_hot = []
                labels_one_hot = []
                try:
                    for sample in load_dict.values():
                        if sample[key]['label'] is not None:
                            labels.append(sample[key]['label'])
                            predictions.append(sample[key]['pred'])

                            # collect auc results for cls
                            if task_type == 'list':
                                preds_one_hot.append(sample[key]['pred_one_hot'])
                                labels_one_hot.append(sample[key]['label_one_hot'])
                except:
                    # cannot load sample's result for certain key in the task config
                    # (because we didn't cover this task in recording or didnt run this task)
                    print('task key is not recorded:', key)
                    print()
                    continue
                else:
                    print('in task:', key)

                if task_type == 'float':  # reg task, key is a name of gene
                    # Calculate correlation for regression task
                    correlation = np.corrcoef(labels, predictions)[0, 1]
                    print("Correlation on " + key + ":", "{:.3f}".format(correlation))
                    df.at[key, 'corr'] = "{:.3f}".format(correlation)

                elif task_type == 'list':  # CLS task  task_type == 'list':
                    accuracy = calculate_cls_acc(labels, predictions, all_task_one_hot_describe[key])
                    try:    # try calculate auc confidence first
                        auc = calculate_cls_auc(preds_one_hot, labels_one_hot, average='macro')
                        print("AUC on " + key + ":", "{:.3f}".format(auc))
                        df.at[key, 'auc'] = "{:.3f}".format(auc)
                    except Exception as e:
                        print(e)
                        print(f"Warning: task {key} has missing classes in test set, thus cannot use confidence for AUC!")
                        # auc = calculate_cls_auc(preds_one_hot, labels_one_hot, average='micro')
                    print("Accuracy on " + key + ":", "{:.3f}".format(accuracy))
                    df.at[key, 'acc'] = "{:.3f}".format(accuracy)
                else:
                    raise  # task_type is not recorded as 'list' ot 'float'
                print()

        csv_path = os.path.join(args.runs_path, run_name, f"{task_key}_results.csv")

        df.to_csv(csv_path, index=True)
        print(f'saved csv file at {csv_path}')


def get_args_parser():
    parser = argparse.ArgumentParser(description='Decoding the test results')
    # Model tag (for example k-fold)
    parser.add_argument('--tag', default=None, type=str, help='Model tag (for example 5-fold)')

    # PATH
    parser.add_argument('--data_path', default='/data/BigModel/embedded_datasets/', type=str,
                        help='MTL dataset root')
    parser.add_argument('--runs_path', default='/home/zhangty/Desktop/BigModel/runs',
                        type=str, help='save running results path')

    # Task settings
    parser.add_argument('--WSI_tasks', default=True, type=bool,
                        help='True for need decoding WSI-level tasks (retriving from config), '
                             'string split with % for manual tasks, None or False for no WSI level tasks')
    parser.add_argument('--Cell_tasks', default=False, type=bool,
                        help='need decoding Cell-level tasks')

    parser.add_argument('--task_setting_folder_name', default='task-settings', type=str,
                        help='task-settings folder name')

    # Model settings
    parser.add_argument('--model_name', default='gigapath', type=str, help='slide_feature level model name')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
