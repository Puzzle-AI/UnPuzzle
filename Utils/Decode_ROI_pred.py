"""
Summary ROI CLS test result to csv    Script  ver： Dec 19th
"""
import argparse
import json
import os

import pandas as pd


def init_csv(save_path, task_name):
    columns = ["Model", "Spec", "lr=1e-06", "lr=1e-05", "lr=1e-04"]
    # 文件名
    csv_file_name = os.path.join(save_path, task_name + ".csv")
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_file_name, index=False)


def main(args):
    run_root = args.run_root
    run_paths = os.listdir(run_root)
    # get csv file
    csv_file_name = os.path.join(args.save_path, args.task_name + ".csv")
    result_csv = pd.read_csv(csv_file_name)
    for run_path in run_paths:
        model_name = run_path.split('_lr_')[0]
        lr = run_path.split('_lr_')[-1]
        if lr == '0.0001':
            lr = '1e-04'
        # get raw of model
        row_model = result_csv[result_csv['Model'] == model_name].copy()
        if row_model.empty:
            row_model = {
                'Model': model_name,
                'Spec': 'Acc',
                "lr=1e-06": 0.,
                "lr=1e-05": 0.,
                "lr=1e-04": 0.
            }
            index_loc = len(result_csv)
        else:
            index_loc = result_csv.index[result_csv['Model'] == model_name]
        # get accaucy
        json_name = f'CLS_{model_name}_test_log.json'
        json_path = os.path.join(run_root, run_path, 'test', 'test', json_name)
        if not os.path.exists(json_path):
            json_path = os.path.join(run_root, run_path, 'test', json_name)
        # read accuracy from log json
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            acc = data['test']['test']['Acc']
            # modify
            column_name = f'lr={lr}'
            row_model[column_name] = round(acc, 2)
            result_csv.loc[index_loc] = row_model
        except Exception as e:
            print(f'Occupied a error: {e}')

        result_csv.to_csv(csv_file_name, index=False)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Summary results to a csv file.')

    parser.add_argument('--run_root', default='/home/BigModel/runs/CLS_NCT-CRC', type=str)
    parser.add_argument('--task_name', default='NCT-CRC', type=str)
    parser.add_argument('--save_path', default='/home/BigModel/Experiments', type=str)

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.save_path, args.task_name + ".csv")):
        init_csv(args.save_path, args.task_name)
    main(args)
