"""
Script ver: Jan 9th 2025 10:00
Get csv results for a dataset
"""

import csv
import os
import pandas as pd
import argparse

def read_from_csv(csv_path):
    """Read content from a CSV file."""
    content = pd.read_csv(csv_path)
    return content

def aggregate_result(runs_path, output_path, task_name_complete):
    """
    Aggregate results from CSV files under specified runs_path and save them to a new CSV.
    
    Searches for folders named in the format:
    'MTL_{MODEL_NAME}_no_slidePT_tiles_{DATASET_NAME}_{TASK_NAME}_{LR}',
    and finds CSV files under 'FOLDER_NAME/Test/WSI_task_results.csv'.
    
    Each CSV has one row, with one column containing a value. This function identifies
    the column with the value and prints the task, model, learning rate, and the value.
    """
    # Define the order of output csv
    lr_order = ["1e-06", "1e-05", "1e-04"]
    model_name_list = ["SlideAve", "ABMIL", "DSMIL", "CLAM", "LongNet", "TransMIL", "SlideViT", "SlideVPT", "SlideMax", "gigapath"]

    os.makedirs(output_path, exist_ok=True)
    
    # Initialize a dictionary to store results in the format required
    task_results = {
        task: {
            "Model": [],
            "LR": [],
            "Corr": [],
            "Acc": [],
            "AUC": []
        } for task in task_name_complete}
    
    # Walk through all directories in the runs_path
    for root, dirs, files in os.walk(runs_path):
        for file in files:
            if file == "WSI_task_results.csv":
                csv_path = os.path.join(root, file)
                # folder_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
                folder_name = os.path.basename(os.path.dirname(csv_path))
                
                # Parse the folder name for task, model, and learning rate
                parts = folder_name.split("_")
                if len(parts) >= 6 and parts[0] == "MTL":
                    # Fetch info from CSV name
                    model_name = parts[1]
                    lr = parts[-1]
                    task_name = None
                    for task in task_name_complete:
                        if task in folder_name:
                            task_name = task
                    assert task_name != None

                    # Read CSV content
                    content = read_from_csv(csv_path)
                    row = content[content['Unnamed: 0'] == task_name]
                    corr = row['corr'].values[0]
                    acc = row['acc'].values[0]
                    auc = row['auc'].values[0]

                    # Append results to the dictionary
                    task_results[task_name]["Model"].append(model_name)
                    task_results[task_name]["LR"].append(lr)
                    task_results[task_name]["Corr"].append(corr)
                    task_results[task_name]["Acc"].append(acc)
                    task_results[task_name]["AUC"].append(auc)

    for task, data in task_results.items():
        if data["Model"]:  # Only save the CSV if there are results for the task
            df = pd.DataFrame(data)

            # Ensure that the 'LR' column is of the correct categorical order
            df["LR"] = pd.Categorical(df["LR"], categories=lr_order, ordered=True)

            # Ensure that the 'Model' column is sorted according to the predefined order
            df["Model"] = pd.Categorical(df["Model"], categories=model_name_list, ordered=True)

            # Pivot the DataFrame to achieve the desired format
            pivot_df = df.pivot_table(index=["Model"], columns="LR", values=["Corr", "Acc", "AUC"], aggfunc="first")
            
            # Save the pivoted DataFrame to a CSV file
            save_csv_name = os.path.join(output_path, f"{task}_results.csv")
            pivot_df.to_csv(save_csv_name)

            print(f"Results have been saved to {save_csv_name}")


if __name__ == "__main__":
    # take input from argument
    parser = argparse.ArgumentParser(description="Summrize benchmark experiment results")
    
    # Define command-line arguments
    parser.add_argument('--runs_path', type=str, required=True, 
                        help="Path to the runs directory.")
    parser.add_argument('--output_path', type=str, required=True, 
                        help="Path to the output directory.")
    parser.add_argument('--task_name_complete', type=str, nargs='+', required=True,
                        help="List of task names (space-separated).")
    
    # Parse the arguments
    args = parser.parse_args()

    # Pass arguments to the function
    aggregate_result(args.runs_path, args.output_path, args.task_name_complete)

    """
    # TCGA-lung
    python retrieve_csv.py \
        --runs_path /data/private/BigModel/runs_TCGA-lung \
        --output_path results_lung \
        --task_name_complete FEV1_PERCENT_REF_POSTBRONCHOLIATOR OS_MONTHS

    # TCGA-BLCA
    python retrieve_csv.py \
        --runs_path /data/private/BigModel/runs_TCGA-BLCA \
        --output_path results_blca \
        --task_name_complete AJCC_PATHOLOGIC_TUMOR_STAGE GRADE OS_MONTHS

    # TCGA-BRCA
    python retrieve_csv.py \
        --runs_path /data/private/BigModel/runs_TCGA-BRCA \
        --output_path results_brca \
        --task_name_complete AJCC_PATHOLOGIC_TUMOR_STAGE_reduced IHC_HER2 HISTOLOGICAL_DIAGNOSIS OS_MONTHS

    # TCGA-UCEC
    python retrieve_csv.py \
        --runs_path /data/private/BigModel/runs_TCGA-UCEC \
        --output_path results_ucec \
        --task_name_complete GRADE OS_MONTHS HISTOLOGICAL_DIAGNOSIS TUMOR_INVASION_PERCENT

    # TCGA-UCS
    python retrieve_csv.py \
        --runs_path /data/private/BigModel/runs_TCGA-UCS \
        --output_path results_ucs \
        --task_name_complete OS_MONTHS HISTOLOGICAL_DIAGNOSIS TUMOR_INVASION_PERCENT OS_STATUS

    # TCGA-UVM
    python retrieve_csv.py \
        --runs_path /data/private/BigModel/runs_TCGA-UVM \
        --output_path results_uvm \
        --task_name_complete OS_MONTHS HISTOLOGICAL_DIAGNOSIS AJCC_PATHOLOGIC_TUMOR_STAGE_reduced OS_STATUS TUMOR_THICKNESS

    # TCGA-CESC
    python retrieve_csv.py \
        --runs_path /data/private/BigModel/runs_TCGA-CESC \
        --output_path results_cesc \
        --task_name_complete LYMPHOVASCULAR_INVOLVEMENT LYMPH_NODES_EXAMINED GRADE OS_MONTHS

    # TCGA-MESO
    python retrieve_csv.py \
        --runs_path /data/private/BigModel/runs_TCGA-MESO \
        --output_path results_meso \
        --task_name_complete AJCC_PATHOLOGIC_TUMOR_STAGE_reduced HISTOLOGICAL_DIAGNOSIS

    # PANDA
    python retrieve_csv.py \
        --runs_path /data/private/BigModel/runs_PANDA \
        --output_path results_panda \
        --task_name_complete isup_grade gleason_score

    # CAMELYON16
    python retrieve_csv.py \
        --runs_path /data/private/BigModel/runs_CAMELYON16 \
        --output_path results_camelyon16 \
        --task_name_complete BREAST_METASTASIS
    """
