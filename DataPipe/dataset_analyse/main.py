"""
Version: Jan 9 2025 10:30
"""

import os
import shutil
import pandas as pd
import numpy as np
from tools.tools import * 
from tools.analyse_datasets import analyse_dataset


def analyse_single_dataset(dataset_name, dataset_path, result_save_path, 
                           selected_cols=None, train_val_test=False, dpi=300):
    """
    Analyse single dataset
    
    Args:
        dataset_name (str): Dataset name to determine the processing method and save file name.
        dataset_path (str): Dataset path, can be csv file or train/val/test set file path.
        result_save_path (str): Result saving path.
        selected_cols (list): Select specified columns to process, default to None (process all columns).
        train_val_test (bool): Whether to process train/val/test set, default to False.
        dpi (int): Image resolution.
    """
    print(f"Starting analysis: {dataset_name}") 

    # -------------------------------
    # Data loading
    # -------------------------------
    # Check if the input is csv or file
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)  # Load CSV
    else:
        df = retrieve_df_from_dataset(dataset_path, train_val_test=train_val_test)  # Load split data
    print(f"Dataset {dataset_name} has been loaded.")

    # -------------------------------
    # Data cleaning
    # -------------------------------
    if selected_cols:
        df = df[selected_cols].dropna()  # If specified, keep selected column and remove NA values
        print(f"Selected task: {selected_cols}. Cleaning missing values.")
    else:
        print("No columns specified. Processing all columns.")

    # -------------------------------
    # Divide dataset
    # -------------------------------
    split_column = 'fold_information_5fold-1'  # The split column

    # Check if split column exists
    if split_column not in df.columns:
        raise KeyError(f"The split column '{split_column}' does not exist in the dataset. Available columns: {df.columns.tolist()}")

    # Divide dataset
    df_train = df[df[split_column] == 'Train']
    df_val = df[df[split_column] == 'Val']
    df_test = df[df[split_column] == 'Test']

    print(f"Training set sample count: {len(df_train)}")
    print(f"Validation set sample count: {len(df_val)}")
    print(f"Testing set sample count: {len(df_test)}")

    # -------------------------------
    # Prepare dataset info
    # -------------------------------
    datasets = [
        ('overall', df),
        ('train', df_train),
        ('val', df_val),
        ('test', df_test)
    ]
    print(f"Processing subsets: {[name for name, _ in datasets]}")

    # -------------------------------
    # Manage output directories
    # -------------------------------
    # If exist already, reset
    if os.path.exists(result_save_path):
        shutil.rmtree(result_save_path)
        print(f"Resetting path: {result_save_path}")
    os.makedirs(result_save_path)
    print(f"Creating path: {result_save_path}")

    # -------------------------------
    # Recursively check each subset
    # -------------------------------
    for subset_name, subset_df in datasets:
        subset_save_path = os.path.join(result_save_path, subset_name)
        if os.path.exists(subset_save_path):
            shutil.rmtree(subset_save_path)
            print(f"Resetting path: {subset_save_path}")
        os.makedirs(subset_save_path)
        print(f"Creating path: {subset_save_path}")

        # Check if the current subset is empty
        if subset_df.empty:
            print(f"Subset {subset_name} is empty. Skipping this subset.")
            continue

        # -------------------------------
        # Call analyse func
        # -------------------------------
        analyse_dataset(dataset_name, subset_name, subset_df, subset_save_path, dpi)


def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Dataset Analysis Tool")

    # Path
    parser.add_argument('--csv_dataset_path', '--csv', type=str, 
                        default="/data/ssd_1/WSI/TCGA-lung/tiles-embeddings/task-settings-5folds/task_description_tcga-lung_20241206.csv",
                        help="Dataset csv label path, can be .csv file.")
    parser.add_argument('--wsi_dataset_path', '--wsi', type=str,
                        default=None,
                        help="Dataset WSI file path.")
    parser.add_argument('--result_save_path', '--rp', type=str, default="results",
                        help="Result saving path")
    parser.add_argument('--dataset_name', '--dn', type=str, default="TCGA-lung",
                        help="Dataset name")

    # Other config
    parser.add_argument('--dpi', type=int, default=400, help="Image resolution in DPI")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    return parser

def main(args):
    dataset_path = args.csv_dataset_path if args.csv_dataset_path else args.wsi_dataset_path
    result_save_path = os.path.join(args.result_save_path, args.dataset_name)
    dataset_name = args.dataset_name
    dpi = args.dpi

    # Check path
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"invalid path: {dataset_path}")
    os.makedirs(result_save_path, exist_ok=True)
    print(f"result path: {result_save_path}")

    # Set random seed
    if args.seed is not None:
        setup_seed(args.seed)

    # Analyse single dataset
    analyse_single_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,  # 使用统一参数名
        result_save_path=result_save_path,
        selected_cols=None,
        train_val_test=False,
        dpi=dpi
    )

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)

    """
    # TCGA-BRCA
    python main.py \
        --dataset_name TCGA-BRCA \
        --csv_dataset_path /data/ssd_1/WSI/TCGA-BRCA/tiles-embeddings/task-settings-5folds/task_description_tcga-brca_20241206.csv

    # TCGA-lung
    python main.py \
        --dataset_name TCGA-lung \
        --csv_dataset_path /data/ssd_1/WSI/TCGA-lung/tiles-embeddings/task-settings-5folds/task_description_tcga-lung_20241206.csv

    # TCGA-MESO
    python main.py \
        --dataset_name TCGA-MESO \
        --csv_dataset_path /data/ssd_1/WSI/TCGA-MESO/tiles-embeddings/task-settings-5folds/task_description_tcga-meso_20241217.csv

    # TCGA-UCEC
    python main.py \
        --dataset_name TCGA-UCEC \
        --csv_dataset_path /data/ssd_1/WSI/TCGA-UCEC/tiles-embeddings/task-settings-5folds/task_description_tcga-ucec_20241218.csv

    # TCGA-USC
    python main.py \
        --dataset_name TCGA-UCS \
        --csv_dataset_path /data/ssd_1/WSI/TCGA-UCS/tiles-embeddings/task-settings-5folds/task_description_tcga-ucs_20241219.csv

    # TCGA-UVM
    python main.py \
        --dataset_name TCGA-UVM \
        --csv_dataset_path /data/ssd_1/WSI/TCGA-UVM/tiles-embeddings/task-settings-5folds/task_description_tcga-uvm_20241219.csv
        
    # TCGA-BLCA
    python main.py \
        --dataset_name TCGA-BLCA \
        --csv_dataset_path /data/ssd_1/WSI/TCGA-BLCA/tiles-embeddings/task-settings-5folds/task_description_tcga-blca_20241218.csv

    # TCGA-CESC
    python main.py \
        --dataset_name TCGA-CESC \
        --csv_dataset_path /data/ssd_1/WSI/TCGA-CESC/tiles-embeddings/task-settings-5folds/task_description_tcga-cesc_20241220.csv

    # PANDA
    python main.py \
        --dataset_name PANDA \
        --csv_dataset_path /data/ssd_1/WSI/PANDA/tiles-embeddings/task-settings-5folds/task_description_20241230.csv

    # CAMELYON16
    python main.py \
        --dataset_name CAMELYON16 \
        --csv_dataset_path /data/ssd_1/WSI/CAMELYON16/tiles-embeddings/task-settings-5folds/task_description.csv

    """
