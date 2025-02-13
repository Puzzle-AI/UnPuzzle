import argparse
import os.path

import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Fill NaN.')
    parser.add_argument('--dataset_folder', default='./results_lung', type=str, help='Dataset folder path')
    parser.add_argument('--dataset_name', default='TCGA-lung', type=str, help='Dataset name')
    parser.add_argument('--data_type', default='WSI', type=str, help='Dataset type')
    parser.add_argument('--output_csv', default='TCGA-overall.csv', type=str, help='Output csv path')
    parser.add_argument('--append', action="store_true", help="append result to target output csv")
    return parser


def convert_task_name(task_name):
    """
    Convert task code to task name for TCGA datasets
    """
    task_code_dict = {
        'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced': 'AJCC Tumor Stage',
        'OS_MONTHS': 'Overall Survival (Months)',
        'lung-cancer-subtyping': 'NSCLC Subtyping',
        'FEV1_PERCENT_REF_POSTBRONCHOLIATOR': 'FEV1',
        'HISTOLOGICAL_DIAGNOSIS': 'Histological Diagnosis'
    }

    for task_code in task_code_dict:
        if task_name == task_code:
            task_name = task_code_dict[task_code]

    return task_name


def get_model_list(df):
    """
    Return all the model names from the CSV.
    We skip those first 2 rows to get the actual list of models.
    """
    model_names = df.iloc[2:, 0].dropna().unique().tolist()
    return model_names


def get_best_result(df, model_name):
    """
    Return the best (max) metric value for the given model row.
    We look across all columns except the first (which is the 'model name'),
    convert them to float, and take the maximum.
    """
    # Find the row for the model name in the first column
    row_mask = (df.iloc[:, 0] == model_name)
    if not row_mask.any():
        # If the model is not found, return None or np.nan
        return np.nan
    
    # Extract that row, skip the first column (the model name),
    # and convert everything else to float to get the max
    row_values = df.loc[row_mask].iloc[0, 1:]
    row_values = pd.to_numeric(row_values, errors='coerce')  # convert to float if possible
    return row_values.max(skipna=True)


def main(args):

    select_df = pd.DataFrame(columns = ['data_type', 'dataset', 'task_type', 'task', 'model', 'result'])

    for task_csv in os.listdir(args.dataset_folder):
        # get task name
        task_name = convert_task_name(task_csv[:-12])

        # load source data
        task_csv_path = os.path.join(args.dataset_folder, task_csv)
        df = pd.read_csv(task_csv_path)

        # define task type
        task_type = None
        if 'Acc' in df.columns:
            task_type = 'cls'
            # remove AUC columns
            df = df.drop(columns=[col for col in df.columns if col.startswith("AUC")], errors="ignore")
        elif 'Corr' in df.columns:
            task_type = 'reg'

        # get all model name list from csv
        model_list = get_model_list(df)
        
        for model_name in model_list:
            # get the best result for current model
            best_result = get_best_result(df, model_name)

            # for classification tasks, their value ranged 0-100
            if task_type == 'cls':
                best_result *= 100

            # Build a dictionary for the new row
            new_row = {
                'data_type': args.data_type, 
                'dataset': args.dataset_name,
                'task_type': task_type,
                'task': task_name,
                'model': model_name, 
                'result': best_result
            }

            # Insert the new row using .loc, with index = current length of the DataFrame
            select_df.loc[len(select_df)] = new_row

    # sort by task type
    select_df = select_df.sort_values(by=["task_type", "model"])

    # save result
    if args.append and os.path.exists(args.output_csv):
        select_df.to_csv(args.output_csv, index=False, mode='a', header=False)
    else:
        select_df.to_csv(args.output_csv, index=False)
    print(f'saved excel at {args.output_csv}')


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
