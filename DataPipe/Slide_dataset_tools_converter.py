"""
tools for slide_feature level dataset      Script  ver: Nov Nov 27th 16:30

load previous patient level task config, convert to slide level task config


------------
For WSI slides, a patient may have multiple WSI samples, where some samples are positive, some are negative.

In the mean time, typical WSI clinical datasets like cBioPortal and GDC only have patient level labels.

In this case, the positive samples of the dataset should be marked as the patient label, 
while the negative samples should be marked as a default set of value (like 'tumor free').

------
How to distinguish TCGA normal (negative) sample by barcode:

A typical TCGA barcode: TCGA-AA-BBBB-CCD, where 'CC' stands for the sample type:
 - Tumor types range from 01 - 09
 - normal types from 10 - 19

Reference: https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/

"""
import os
import re
import random
import pandas as pd
import numpy as np
import shutil
import yaml  # Ensure pyyaml is installed: pip install pyyaml
from sklearn.model_selection import KFold

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def read_df_from_file(file_path: str):
    """Read file into a dataframe

    Args:
        file_path (str): Read file path.

    Returns:
        df: dataframe object
    """
    file_type = file_path.split('.')[-1]

    if file_type == 'tsv':
        df = pd.read_csv(file_path, sep='\t')
    elif file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'txt':
        df = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f'{file_type}: File type not supported.')

    # Convert to numeric if possible
    df.replace(to_replace="'--", value=None, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')

    return df


# csv data split tools:
def write_csv_data(task_description_csv, id_key, id_data, key='split', val='Train'):
    """
    legacy, will be removed!

    Edit the CSV file by adding (if not there, otherwise edit) a column name of key (such as 'split')

    Parameters:
    - task_description_csv: Path to the CSV file.
    - id_key: The name of the column that contains the IDs to match with id_data.
    - id_data: A list of values corresponding to the id_key column, for which the key column should be updated.
    - key: The name of the column to add or update. Defaults to 'split'.
    - val: The value to set in the key column for the matching rows. Defaults to 'train'.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(task_description_csv)

    # If the key column does not exist, create it and fill with empty strings, else will rewrite
    if key not in df.columns:
        df[key] = ""

    # Update the rows where the id_key matches any of the values in id_data
    df.loc[df[id_key].isin(id_data), key] = val

    # Write the updated DataFrame back to the CSV
    df.to_csv(task_description_csv, index=False)


###########################################
###          CSV Checking Functions      ###
###########################################


def csv_check_default(patient_label, id_key, id_column, key):
    """Check and clean csv data for normal sample in default dataset.

    Args:
        patient_label (pd.DataFrame): A DataFrame containing patient data to be cleaned.
        id_key (str): The key corresponding to the patient identifier column.
        id_column (str): The name of the column used for indexing patient data.
        key (str): An additional key for data processing and validation.

    Returns:
        pd.DataFrame: The modified `patient_label` DataFrame with cleaned data.
    """

    # Define a list of columns to exempt from clearing
    exempt_list = [id_key, id_column, key]

    # Clear all non-exempt columns by setting them to None
    patient_label.loc[~patient_label.index.isin(exempt_list)] = None

    return patient_label


def csv_check_tcga(patient_label, id_key, id_column, key):
    """Check and clean csv data for normal sample in TCGA dataset.

    This function modifies the `patient_label` DataFrame to clean and standardize
    its contents for the TCGA dataset. It clears unwanted data while retaining
    specific columns, and assigns default values to predefined fields.

    Args:
        patient_label (pd.DataFrame): A DataFrame containing patient data to be cleaned.
        id_key (str): The key corresponding to the patient identifier column.
        id_column (str): The name of the column used for indexing patient data.
        key (str): An additional key for data processing and validation.

    Returns:
        pd.DataFrame: The modified `patient_label` DataFrame with cleaned data.
    """

    # Define a list of columns to exempt from clearing
    exempt_list = [id_key, id_column, key, 'SEX', 'AGE']

    # Clear all non-exempt columns by setting them to None
    patient_label.loc[~patient_label.index.isin(exempt_list)] = None

    # Add default values to specific columns
    if 'TUMOR_STATUS' in patient_label.index:
        patient_label['TUMOR_STATUS'] = 'TUMOR FREE'
    elif 'lung-cancer-subtyping' in patient_label.index:
        patient_label['lung-cancer-subtyping'] = 'normal'
    elif 'cancer_type' in patient_label.index:  # for TCGA-COAD-READ
        patient_label['cancer_type'] = 'normal'
    return patient_label


###########################################
###      Split By Patient Functions      ###
###########################################


def build_patient_sample_dict(all_wsi_names, all_wsi_folders, mode='TCGA'):
    """Build a dictionary mapping patients to their WSI samples.

    This function processes WSI data to create mappings between patients and their 
    associated WSIs for the TCGA dataset. It also identifies missing, redundant, 
    positive (tumor), and negative (normal) samples based on the TCGA barcode format.

    Args:
        all_wsi_names (list): A list of WSI names extracted from the CSV file.
        all_wsi_folders (list): A list of WSI folder paths present in the dataset.
        mode (str, optional): Dataset mode; defaults to 'TCGA'. If not 'TCGA', 
            assumes all samples within the same WSI are regarded as having the same label.

    Returns:
        dict: `patient_wsis`, mapping each patient ID to a list of WSIs for training, 
            validation, and testing.
        dict: `missing_patient_wsis`, mapping each patient ID to WSIs that are in 
            the CSV file but missing in the folder.
        dict: `redundant_patient_wsis`, mapping each patient ID to WSIs that are in 
            the folder but not listed in the CSV file.
        list: `positive_samples`, containing WSIs classified as tumor-positive.
        list: `negative_samples`, containing WSIs classified as tumor-negative.
    """
    # Initialize dictionaries and lists
    patient_wsis = {}   # Patient-WSI mapping for valid samples
    missing_patient_wsis = {}   # Samples in CSV but missing in folder
    redundant_patient_wsis = {} # Samples in folder but missing in CSV
    positive_samples = []       # Tumor-positive WSIs
    negative_samples = []       # Tumor-negative WSIs
    
    if mode == 'TCGA':
        # Map WSIs to patients and identify missing samples
        for wsi_name in all_wsi_names:
            # Extract patient name (first 12 characters in TCGA naming convention)
            patient_name = wsi_name[:12]

            # Initialize patient entry in `patient_wsis` if not already present
            patient_wsis.setdefault(patient_name, [])

            # Find WSIs belonging to the patient
            for sample in all_wsi_folders:
                if patient_name in sample:
                    patient_wsis[patient_name].append(sample)

            # If no samples are found for the patient, record it in `missing_patient_wsis`
            if len(patient_wsis[patient_name]) == 0:
                del patient_wsis[patient_name]
                missing_patient_wsis.setdefault(patient_name, [])
                missing_patient_wsis[patient_name].append(sample)

        # Identify samples in the folder but missing in CSV
        for sample in all_wsi_folders:
            patient_name = sample[:12]
            if patient_name not in patient_wsis:
                redundant_patient_wsis.setdefault(patient_name, [])
                redundant_patient_wsis[patient_name].append(sample)

        # Classify WSIs as positive or negative based on TCGA barcode
        # TCGA format: TCGA-XX-YYYY-ZZD-WWW, where ZZ determines tumor status
        pattern = r"^TCGA-\w{2}-\w{4}-\d{2}[A-Z]"
        for wsi_folder in all_wsi_folders:
            match = re.match(pattern, wsi_folder)
            assert match, f"Invalid barcode format for folder name: {wsi_folder}"
            sample_type_code = int(match.group()[13:15])

            # Classify tumor status based on sample type code
            if 1 <= sample_type_code <= 9:
                positive_samples.append(wsi_folder)
            elif 10 <= sample_type_code <= 19:
                negative_samples.append(wsi_folder)
            else:
                raise ValueError(f"Sample {wsi_folder} has invalid sample type code: {sample_type_code}")
    else:
        # Default mode: Treat all WSIs as belonging to the same patient
        # As we don't know which sample is normal sample, we regard every sample to be positive sample
        print("Warning: Non-TCGA mode. All WSIs within the same folder will be treated as having the same label.")
        for wsi_name in all_wsi_names:
            patient_name = wsi_name[:]  # Treat WSI name as patient name
            patient_wsis.setdefault(patient_name, [])
            for sample in all_wsi_folders:
                if patient_name in sample:
                    patient_wsis[patient_name].append(sample)

    return patient_wsis, missing_patient_wsis, redundant_patient_wsis, positive_samples, negative_samples


def modify_csv_data(patient_wsis, task_description_csv, negative_samples, slide_id_key, slide_id, key, mode='TCGA'):
    """Modify CSV data with split and sample information for a dataset.

    This function processes a task description CSV file by adding split and sample information
    based on given fold information and patient-wise WSI data. It also handles negative samples
    (e.g., tumor-free WSIs) differently depending on the dataset mode.

    Args:
        patient_wsis (dict): A dictionary mapping each patient ID to their associated WSI samples.
        task_description_csv (str): Path to the CSV file containing task description data.
        negative_samples (list): A list of samples labeled as tumor-free.
        slide_id_key (str): Column name in the CSV that identifies patient/sample IDs.
        slide_id (str): Column name to be used or created for storing sample identifiers.
        key (str): Column name to be used or created for storing split labels ('Train', 'Val', 'Test').
        mode (str, optional): Dataset mode; defaults to 'TCGA'. If not 'TCGA', negative samples are ignored.

    Returns:
        pd.DataFrame: A new DataFrame with updated split and sample information.
    """
    # Read the CSV file and load it into a DataFrame
    df = pd.read_csv(task_description_csv)

    # Ensure key columns exist in the DataFrame
    k = -1
    if key in df.columns:
        k = 1
    else:
        while True:
            key_fold = f'{key}_{k}fold-{k}'
            if key_fold in df.columns:
                k += 1
            else:
                break
            if k >= 20:  # raise error if there is no split column
                raise ValueError(f'No valid split info found in csv: {task_description_csv}')

    # Ensure slide_id column exist in the DataFrame
    if slide_id not in df.columns:
        df[slide_id] = ""

    # Initialize a new DataFrame for storing the updated data
    df_new = pd.DataFrame(columns=df.columns)

    # Process negative sample information
    for patient in patient_wsis:
        for sample in patient_wsis[patient]:
            # Extract rows related to the current sample
            patient_label = df[df[slide_id_key].apply(lambda x: str(x) in sample)].copy()

            # # Match sample directly if multiple rows are found
            # if len(patient_label) > 1:
            #     patient_label = df[df[slide_id] == sample].copy()

            # Update columns and reset index
            patient_label[slide_id] = sample  # Add sample identifier
            patient_label = patient_label.squeeze() # Convert from DataFrame to Series

            # Handle negative samples
            if sample in negative_samples:
                if mode == 'TCGA':
                    patient_label = csv_check_tcga(patient_label, slide_id_key, slide_id, key)
                else:
                    patient_label = csv_check_default(patient_label, slide_id_key, slide_id, key)

            # Append the processed rows to the new DataFrame
            df_new.loc[len(df_new)] = patient_label

    return df_new


def build_data_split_for_csv(task_description_csv, data_path, slide_id_key='patient_id', slide_id='slide_id', 
                             test_ratio=0.2, k=1, mode='TCGA', key='split'):
    """
    Modify a CSV file to include data split information for k-fold cross-validation.

    This function processes a CSV file containing task descriptions, and appends 
    columns to indicate train/val/test split information for k-fold cross-validation 
    based on the provided dataset structure and mode.

    Args:
        task_description_csv (str): Path to the task description CSV file.
        data_path (str): Root directory containing WSI sample folders.
        slide_id_key (str, optional): Column name in the CSV that identifies patient/sample IDs. 
            Defaults to 'patient_id'.
        slide_id (str, optional): Column name to store slide/sample IDs. Defaults to 'slide_id'.
        test_ratio (float, optional): Proportion of data allocated to the test set. Defaults to 0.2.
        k (int, optional): Number of folds for k-fold cross-validation. Defaults to 1.
        mode (str, optional): Dataset mode; defaults to 'TCGA'. Determines how negative samples 
            are handled and patient-sample mapping is constructed.
        key (str, optional): Column name for storing split labels ('split'). Defaults to 'split'.

    Returns:
        None: Modifies the input CSV file by appending new columns for data splits.

    Process:
        1. Determine the total number of folds:
           - If `k > 1`, use `k` folds for cross-validation.
           - If `k == 1`, default to using a single fold from a 5-fold split.
        2. Retrieve all WSI samples:
           - `all_wsi_names`: Extracted from the CSV file (patient-level labels assumed).
           - `all_wsi_folders`: Extracted from the dataset directory.
        3. Map WSIs to patients:
           - Using `build_patient_sample_dict`, match patient IDs to WSI folders.
           - Identify missing, redundant, and valid WSIs.
        4. Split the data into train/val/test:
           - Use `make_split_by_patient` to generate split information at the patient level.
        5. Modify the CSV file:
           - Use `modify_csv_data` to update the CSV with split labels and handle negative samples.
        6. Save the updated CSV file back to the original path.

    """

    print('Warning: You are running slide_dataset_tools_converter, test_ratio and k will be ignored!')

    # Get a list of all WSI samples from the CSV file
    all_wsi_names = list(pd.read_csv(task_description_csv)[slide_id_key])

    # Get a list of all WSI sample folders from the dataset directory
    all_wsi_folders = [
        wsi_folder for wsi_folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, wsi_folder)) and not wsi_folder.startswith('task-settings')
    ]
    print(f'Found {len(all_wsi_folders)} WSI folders under {data_path}')

    # Map WSIs to patients and identify missing, redundant, and valid WSIs
    patient_wsis, missing_patient_wsis, redundant_patient_wsis, positive_samples, negative_samples = \
        build_patient_sample_dict(all_wsi_names, all_wsi_folders, mode)
    print(f'Valid patients: {len(patient_wsis)} | '
          f'Missing: {len(missing_patient_wsis)} | '
          f'Redundant: {len(redundant_patient_wsis)} | '
          f'Negative samples: {len(negative_samples)}')

    # Modify CSV data to include negative sample labels
    df_csv = modify_csv_data(patient_wsis, task_description_csv, negative_samples, slide_id_key, slide_id, key, mode)

    # Write the updated DataFrame back to the CSV file
    df_csv.to_csv(task_description_csv, index=False)

    return k


###########################################
###           Support Functions           ###
###########################################


def load_pickle_data_split_for_csv(task_description_csv, slide_id_key='slide_id', key='split', input_pkl_rootpath=None,
                                   mode='TCGA', k=1):
    """
    fixme: there is no write_csv_data func now.
    This write previous pkl split into csv
    Args:
        task_description_csv:
        slide_id_key:
        key:
        input_pkl_rootpath:
        mode:
        k:

    Returns:

    Demo
    # load 5 fold pkl to csv
    load_pickle_data_split_for_csv(
        task_description_csv='/data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv',
        slide_id_key='patient_id', key='fold_information', input_pkl_rootpath='/data/ai4dd/TCGA_5-folds',
        mode='TCGA', k=5)
    """
    import pickle

    def load_data(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    if k == 1:
        Train_list = load_data(os.path.join(input_pkl_rootpath, 'Train.pkl'))
        Val_list = load_data(os.path.join(input_pkl_rootpath, 'Val.pkl'))
        Test_list = load_data(os.path.join(input_pkl_rootpath, 'Test.pkl'))

        # in TCGA the csv slide_id_key is patient name so we need to write the patient split into it
        if mode == 'TCGA':
            Train_list = [sample[:12] for sample in Train_list]
            Val_list = [sample[:12] for sample in Val_list]
            Test_list = [sample[:12] for sample in Test_list]

        write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Train_list, key=key, val='Train')
        write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Val_list, key=key, val='Val')
        write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Test_list, key=key, val='Test')
    else:
        for fold in range(1,k+1):
            fold_pkl_rootpath = os.path.join(input_pkl_rootpath, 'task-settings-' + str(k) + 'folds_fold-' + str(fold))

            Train_list = load_data(os.path.join(fold_pkl_rootpath, 'Train.pkl'))
            Val_list = load_data(os.path.join(fold_pkl_rootpath, 'Val.pkl'))
            Test_list = load_data(os.path.join(fold_pkl_rootpath, 'Test.pkl'))

            # in TCGA the csv slide_id_key is patient name so we need to write the patient split into it
            if mode == 'TCGA':
                Train_list = [sample[:12] for sample in Train_list]
                Val_list = [sample[:12] for sample in Val_list]
                Test_list = [sample[:12] for sample in Test_list]

            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Train_list,
                           key=key + '_{}fold-{}'.format(k, fold), val='Train')
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Val_list,
                           key=key + '_{}fold-{}'.format(k, fold), val='Val')
            write_csv_data(task_description_csv, id_key=slide_id_key, id_data=Test_list,
                           key=key + '_{}fold-{}'.format(k, fold), val='Test')

    print('done')


# task config tools:
def build_task_config_settings(df, new_labels, one_hot_table={}, all_task_dict={}, max_possible_values=100):
    assert all(label in df.columns for label in new_labels)

    selected_new_labels = []
    invalid_labels = [] # store invalid labels and return

    for label in new_labels:
        # new label should not be in existing config
        if label in one_hot_table or label in all_task_dict:
            raise ValueError(f'Duplicate label: {label}')

        # get the list of all possible values under the current column
        content_list = list(df[label].value_counts().keys())  # this also removes the duplicates
        # change all value type to string
        valid_content_list = [str(i) for i in content_list if i != 'missing in csv']
        # fixme this is to handel bug outside

        try:
            # ensure all can be converted to float
            for content in valid_content_list:
                tmp = float(content)
        except:
            # consider as classification task if any data cannot be transformed into float.
            str_flag = True
        else:
            str_flag = False

        if not str_flag:
            all_task_dict[label] = 'float'
            print(f'Regression task added to task settings: {label}')
        else:  # maybe it's a cls task
            # skip if too many possible values
            if len(valid_content_list) > max_possible_values:
                invalid_labels.append(label)
                continue  # jump this label
            # skip if the value is constant
            elif len(valid_content_list) == 1:
                continue  # jump this label
            # sort valid_content_list
            valid_content_list.sort()
            # confirm its a valid cls task
            all_task_dict[label] = 'list'
            # generate task settings
            value_list = np.eye(len(valid_content_list), dtype=int)
            value_list = value_list.tolist()
            idx = 0
            one_hot_table[label] = {}
            for content in valid_content_list:
                one_hot_table[label][content] = value_list[idx]
                idx += 1
            print(f'Classification task added to task settings: {label}')

        selected_new_labels.append(label)

    return one_hot_table, all_task_dict, selected_new_labels, invalid_labels


def build_yaml_config_from_csv(task_description_csv, task_settings_path, dataset_name='lung-mix',
                               tasks_to_run=None, max_tiles=1000000, mode='TCGA', shuffle_tiles=True,
                               excluding_list=('WSI_name', 'slide_id', 'split','Folder', 'File_name','Slide','Tissue',
                                               'Type','Disposition','mpp','Sub folder','Status','Date',
                                               'Time','Patient ID', 'uniquePatientKey', 'FORM_COMPLETION_DATE', 
                                               'OTHER_PATIENT_ID', 'studyId'), yaml_config_name='task_configs.yaml'):
    """
    Build a YAML configuration file from a CSV file containing task descriptions.

    Parameters:
    task_description_csv (str): Path to the task_description .csv file.
    task_settings_path (str): Output directory for the YAML file. (task-settings path)

    dataset_name (str): Name of the dataset. Default is 'lung-mix'.
    tasks_to_run (str): Setting type (e.g., 'MTL'). Default is 'MTL'.
    max_tiles (int): Maximum number of tiles. Default is 1000000.
    shuffle_tiles (bool): Whether to shuffle tiles or not. Default is True.
    excluding_list (tuple): List of columns to exclude. Default is ('WSI_name', ...).
                            the attribute starts with 'split' will be ignored as they are designed for control split
                            EG: 'split_nfold-k', n is the total fold number and k is the fold index
    """

    try:
        task_description = read_df_from_file(task_description_csv)
    except:  # no valid label selected
        raise ValueError('Invalid input!', task_description_csv)

    one_hot_table, all_task_dict = {}, {}
    excluding_list = list(excluding_list)

    # select columns in csv to be used as the labels.
    # By default, all columns except slide_id_key will be used as label.
    tentative_task_labels = [col for col in task_description.columns if col not in excluding_list]

    if tasks_to_run is not None:
        for task in tasks_to_run:
            assert task in tentative_task_labels
    else:
        # take all tasks as valid tasks
        tasks_to_run = tentative_task_labels

    one_hot_table, all_task_dict, selected_new_labels, invalid_labels = \
        build_task_config_settings(task_description, tentative_task_labels, one_hot_table, all_task_dict)

    # remove invalid labels from tasks_to_run
    tasks_to_run = [task for task in tasks_to_run if task not in invalid_labels and task in all_task_dict]

    print(f'#' * 30)
    print(f'Add labels to config: {selected_new_labels}')
    print(f'#' * 30)

    config = {
        'name': dataset_name,
        'tasks_to_run': tasks_to_run,
        'all_task_dict': all_task_dict,
        'one_hot_table': one_hot_table,
        'max_tiles': max_tiles,
        'shuffle_tiles': shuffle_tiles,
        'mode': mode
    }

    if not os.path.exists(task_settings_path):
        os.makedirs(task_settings_path)

    yaml_output_path = os.path.join(task_settings_path, yaml_config_name)
    if os.path.exists(yaml_output_path):
        os.remove(yaml_output_path)

    with open(yaml_output_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    return all_task_dict, one_hot_table


'''
def load_yaml_config(yaml_path):
    """Load the YAML configuration file."""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
    The error you're encountering suggests that the YAML file contains a Python tuple, which yaml.safe_load 
    doesn't know how to handle by default. By default, yaml.safe_load does not allow the loading of arbitrary 
    Python objects, including tuples, for security reasons.

To solve this issue, you can use yaml.Loader instead of SafeLoader, which supports loading Python objects like tuples. 
'''

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    return config


def build_split_and_task_configs(data_path, task_description_csv, dataset_name,
                                 tasks_to_run, slide_id_key, slide_id='slide_id', split_target_key='fold_information',
                                 task_setting_folder_name='task-settings',
                                 mode='TCGA', test_ratio=0.2, k=1, yaml_config_name='task_configs.yaml',
                                 excluding_list=('WSI_name', 'slide_id', 'split','Folder', 'File_name','Slide','Tissue',
                                               'Type','Disposition','mpp','Sub folder','Status','Date',
                                               'Time','Patient ID', 'uniquePatientKey', 'FORM_COMPLETION_DATE', 
                                               'OTHER_PATIENT_ID', 'studyId')):

    assert os.path.exists(data_path), 'data_path does not exist: {}'.format(data_path)
    output_dir = os.path.join(data_path, task_setting_folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if task_description_csv is None:
        task_description_csv = os.path.join(data_path, task_setting_folder_name, 'task_description.csv')
    else:
        assert os.path.exists(task_description_csv), 'task_description_csv does not exist!'
        shutil.copy(task_description_csv, os.path.join(data_path, task_setting_folder_name, 'task_description.csv'))
        task_description_csv = os.path.join(data_path, task_setting_folder_name, 'task_description.csv')

    k = build_data_split_for_csv(task_description_csv, data_path, slide_id_key=slide_id_key, slide_id=slide_id, test_ratio=test_ratio, k=k,
                             mode=mode, key=split_target_key)

    excluding_list = list(excluding_list)
    excluding_list.extend([slide_id_key, split_target_key])

    if k > 1:
        excluding_list.extend([split_target_key + '_{}fold-{}'.format(k, fold) for fold in range(1,k+1)])

    build_yaml_config_from_csv(task_description_csv, output_dir, dataset_name=dataset_name,
                               tasks_to_run=tasks_to_run,
                               max_tiles=1000000, shuffle_tiles=True,mode=mode,
                               excluding_list=excluding_list,
                               yaml_config_name=yaml_config_name)
    # check
    load_yaml_config(os.path.join(data_path, task_setting_folder_name, yaml_config_name))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Build split and task configs.')

    parser.add_argument('--data_path', type=str, default='/data/BigModel/embedded_datasets/',
                        help='Root path for the datasets')
    parser.add_argument('--task_description_csv',
                        default=None,
                        type=str, help='label csv file path, if given'
                                       ' it will be copied to the task_setting_folder as task_description.csv'
                                       'default none will go to find task_description.csv in task_setting_folder')

    parser.add_argument('--slide_id_key', type=str, default='patient_id',
                        help='Slide ID key in the dataset')
    parser.add_argument('--slide_id', type=str, default='slide_id',
                        help='Slide ID to write in csv')
    parser.add_argument('--split_target_key', type=str, default='fold_information',
                        help='Key to split the dataset')

    parser.add_argument('--task_setting_folder_name', type=str, default='task-settings-5folds',
                        help='Folder name for task settings')
    parser.add_argument('--mode', type=str, default='TCGA',
                        help='Mode (e.g., TCGA)')
    parser.add_argument('--k', type=int, default=5,
                        help='k-fold k num')

    parser.add_argument('--dataset_name', type=str, default='coad-read',
                        help='Name of the dataset')
    parser.add_argument('--tasks_to_run', type=str,
                        default=None,
                        help='Tasks to run, separated by "%", default None will take all tasks in csv')

    args = parser.parse_args()

    # Convert tasks_to_run to a list
    tasks_to_run = args.tasks_to_run.split('%') if args.tasks_to_run else None

    build_split_and_task_configs(
        data_path=args.data_path,
        task_description_csv=args.task_description_csv,
        dataset_name=args.dataset_name,
        tasks_to_run=tasks_to_run,
        slide_id_key=args.slide_id_key,
        slide_id=args.slide_id,
        split_target_key=args.split_target_key,
        task_setting_folder_name=args.task_setting_folder_name,
        mode=args.mode,
        k=args.k
    )
    # demo with TCGA
    '''
    python Slide_dataset_tools_converter.py --data_path /data/BigModel/embedded_datasets/ \
    --task_description_csv /home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
    --slide_id_key patient_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings-5folds \
    --mode TCGA \
    --dataset_name coad-read \
    --tasks_to_run iCMS%CMS%MSI.status%EPCAM%COL3A1%CD3E%PLVAP%C1QA%IL1B%MS4A1%CD79A \
    --k 5

    python Slide_dataset_tools_converter.py --data_path /data/BigModel/embedded_lung/ \
    --task_description_csv /data/BigModel/embedded_lung/task-settings/TCGA-LUNG-task_description.csv \
    --slide_id_key WSI_name \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings \
    --mode None \
    --dataset_name lung-mix \
    --tasks_to_run purity%FRACTION_GENOME_ALTERED%AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%lung-cancer-subtyping \
    --k 1
    '''

    # demo with dev
    '''
    python -u Slide_dataset_tools_converter.py \
        --data_path /data/ssd_1/WSI/TCGA-COAD-READ/tiles-embeddings/ \
        --task_description_csv /data/ssd_1/WSI/TCGA-COAD-READ/tiles-embeddings/task-settings-5folds/20240827_TCGA_log_marker10.csv \
        --slide_id_key patient_id \
        --slide_id slide_id \
        --split_target_key fold_information \
        --task_setting_folder_name task-settings-5folds \
        --mode TCGA \
        --dataset_name coad-read \
        --k 5

    python -u Slide_dataset_tools_converter.py \
        --data_path /data/ssd_1/WSI/TCGA-lung/tiles-embeddings/ \
        --task_description_csv /data/ssd_1/WSI/TCGA-lung/tiles-embeddings/task-settings-5folds/task_description_tcga-lung_20241121.csv \
        --slide_id_key patientId \
        --slide_id slide_id \
        --split_target_key fold_information \
        --task_setting_folder_name task-settings-5folds \
        --mode TCGA \
        --dataset_name lung \
        --k 5
    '''

    '''
    # load 5 fold pkl to csv (if we need to load 5 fold from previous split)
    load_pickle_data_split_for_csv(
        task_description_csv='/data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv',
        slide_id_key='patient_id', key='fold_information', input_pkl_rootpath='/data/ai4dd/TCGA_5-folds',
        mode='TCGA', k=5)
    '''

    # demo with SGH
    '''
    python Slide_dataset_tools_converter.py --data_path '/data/hdd_1/BigModel/qupath_embedded_datasets' \
    --task_description_csv /home/zhangty/Desktop/BigModel/prov-gigapath/PuzzleAI/Archive/dataset_csv/SGH_Transcriptome_Log_Combined.csv \
    --slide_id_key slide_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings-5folds \
    --mode SGH \
    --dataset_name SGH_aidd \
    --tasks_to_run None \
    --k 5
    '''
