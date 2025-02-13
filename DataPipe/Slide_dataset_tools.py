"""
tools for slide_feature level dataset      Script  ver: Jan 20th 16:00

build and load task config

------------
Split by subject

TCGA mode: we find the patient name as 'subject name' and split by them, then we took the corresponding wsi folders

Other mode: we assume the subject name will appear within the folder name of each sample,
            we split by the specified subject name (can be wsi name, can be patient name, etc)
            we go through the folders, take the folders of the subjects (with subject_name appear in folder_name)

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
import warnings
import yaml  # Ensure pyyaml is installed: pip install pyyaml
from sklearn.model_selection import KFold
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


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


###########################################
###          CSV Fixing for TCGA Functions      ###
###########################################

def csv_get_new_columns(mode):
    """Return a list of new columns based on mode
    """
    new_list = []
    # Add a list of new columns
    if mode == 'TCGA':
        new_list = ['TISSUE_TYPE', 'SLIDE_TYPE', 'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced']
    return new_list


def csv_check_normal_tcga(sample_label, sample, subject_key, sample_key, normal=True):
    """Check and clean csv data for normal sample in TCGA dataset.

    This function modifies the `sample_label` DataFrame to clean and standardize
    its contents for the TCGA dataset. It clears unwanted data while retaining
    specific columns, and assigns default values to predefined fields.

    Args:
        sample_label (pd.DataFrame): A DataFrame containing sample data to be cleaned.
        sample (str): Sample WSI file directory.
        subject_key (str): The key corresponding to the patient identifier column.
        sample_key (str): The name of the column used for indexing slide data.

        normal (bool, optional):
    Returns:
        pd.DataFrame: The modified `patient_label` DataFrame with cleaned data.
    """

    # Define a list of columns to exempt from clearing
    exempt_list = [subject_key, sample_key, 'SEX', 'AGE']

    # Clear all non-exempt columns by setting them to None
    if normal:
        sample_label.loc[~sample_label.index.isin(exempt_list)] = None

    # Add default values to specific columns
    if 'TISSUE_TYPE' in sample_label.index:
        sample_label['TISSUE_TYPE'] = 'TUMOR FREE' if normal else 'WITH TUMOR'

    # Add slide type based on barcode
    if 'SLIDE_TYPE' in sample_label.index:
        pattern = r"^TCGA-\w{2}-\w{4}-\d{2}[A-Z]-\w{2}-\w{3}"
        match = re.match(pattern, sample)
        assert match, f"Invalid barcode format for folder name: {sample}"
        tcga_slide_type = match.group()[20:23]
        sample_label['SLIDE_TYPE'] = tcga_slide_type

    # Get value for new labels (only if is not normal)
    if 'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced' in sample_label.index:
        try:
            tumor_stage_full = sample_label['AJCC_PATHOLOGIC_TUMOR_STAGE']
            if type(tumor_stage_full) == str:  # check if it is empty
                for tumor_stage in ['Stage I', 'Stage II', 'Stage III', 'Stage IV']:
                    if tumor_stage in tumor_stage_full:
                        sample_label['AJCC_PATHOLOGIC_TUMOR_STAGE_reduced'] = tumor_stage
        except:
            pass

    return sample_label


###########################################
###      Split By subject Functions      ###
###########################################


def build_WSI_subject_sample_dict(data_path=None,
                                  task_description_csv=None, subject_key=None, sample_key=None,
                                  mode='TCGA'):
    """Build a dictionary mapping subjects(patients) to their WSI samples.

    This function processes WSI data to create mappings between subjects(patients) and their
    associated WSIs for the TCGA dataset. It also identifies missing, redundant,
    positive (tumor), and negative (normal) samples based on the TCGA barcode format.

    Args:
        data_path (str): Root directory containing WSI sample folders.

        task_description_csv (str): Path to the task description CSV file.
        subject_key (str): Column name in the CSV that identifies patient/sample IDs.
            Defaults to 'patient_id'. we build the split based on this

        sample_key (str): Column name to store slide/sample IDs. Defaults to 'slide_id'.
                    we based on subject_name to find the sample_name of each sample belonging to certain subject,
                    this sample_key collumn should be the name of each sample

        mode (str, optional): Dataset mode; defaults to 'TCGA'. If not 'TCGA',
            assumes all samples within the same WSI are regarded as having the same label.

    Returns:
        dict: `subject_sample_dict`, mapping each subject(patient) ID to a list of WSIs for training,
            validation, and testing.
        dict: `missing_sample_wsis`, mapping each subject(patient) ID to WSIs that are in
            the CSV file but missing in the folder.
        dict: `redundant_sample_wsis`, mapping each subject(patient) ID to WSIs that are in
            the folder but not listed in the CSV file.
        list: `valid_samples`, containing all valid WSIs.
        list: `positive_samples`, containing WSIs classified as tumor-positive.
        list: `negative_samples`, containing WSIs classified as tumor-negative.
    """
    assert task_description_csv is not None and subject_key is not None and sample_key is not None
    # Initialize dictionaries and lists
    subject_sample_dict = {}  # Subject(Patient or etc)-WSI mapping for valid samples
    missing_sample_wsis = {}  # Samples in CSV but missing in folder
    redundant_sample_wsis = {}  # Samples in folder but missing in CSV
    positive_samples = []  # Tumor-positive WSIs
    negative_samples = []  # Tumor-negative WSIs

    if mode == 'TCGA':
        assert data_path is not None
        # Get a list of all subject samples from the CSV file
        all_subject_names = list(pd.read_csv(task_description_csv)[subject_key])

        # Get a list of all WSI sample folders from the dataset directory
        all_wsi_folders = [
            wsi_folder for wsi_folder in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, wsi_folder)) and not wsi_folder.startswith('task-settings')
        ]
        print(f'Found {len(all_wsi_folders)} WSI folders under {data_path}')

        # Map WSIs to subjects (patients) and identify missing samples
        for wsi_name in all_subject_names:
            # Extract subject name (first 12 characters in TCGA naming convention)
            subject_name = wsi_name[:12]

            # Initialize subject entry in `subject_sample_dict` if not already present
            subject_sample_dict.setdefault(subject_name, [])

            # Find WSIs belonging to the subject
            for sample in all_wsi_folders:
                if subject_name in sample:
                    subject_sample_dict[subject_name].append(sample)

            # If no samples are found for the subject, record it in `missing_sample_wsis`
            if len(subject_sample_dict[subject_name]) == 0:
                del subject_sample_dict[subject_name]
                missing_sample_wsis.setdefault(subject_name, [])
                missing_sample_wsis[subject_name].append(sample)

        # Identify samples in the folder but missing in CSV
        for sample in all_wsi_folders:
            subject_name = sample[:12]
            if subject_name not in subject_sample_dict:
                redundant_sample_wsis.setdefault(subject_name, [])
                redundant_sample_wsis[subject_name].append(sample)

        # Classify WSIs as positive or negative based on TCGA barcode
        # Also, only keep FFPE samples.
        # Ref: https://andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/
        # TCGA format: TCGA-XX-YYYY-ZZD-WWW, where ZZ determines tumor status
        pattern = r"^TCGA-\w{2}-\w{4}-\d{2}[A-Z]-\w{2}-\w{3}"
        valid_samples = []
        for sample in all_wsi_folders:
            match = re.match(pattern, sample)
            assert match, f"Invalid barcode format for folder name: {sample}"
            tcga_slide_type = match.group()[20:23]
            tcga_tumor_type = int(match.group()[13:15])
            subject_name = match.group()[:12]

            # Skip if not a selected subject
            if subject_name not in subject_sample_dict:
                continue

            # Filter frozen slides and other type of invalid slides
            if tcga_slide_type[:2] != 'DX':  # FFPE
                subject_sample_dict[subject_name].remove(sample)
                continue

            # Classify tumor status based on sample type code
            if 1 <= tcga_tumor_type <= 9:
                positive_samples.append(sample)
            elif 10 <= tcga_tumor_type <= 19:
                negative_samples.append(sample)
            else:
                raise ValueError(f"Sample {sample} has invalid sample type code: {tcga_tumor_type}")

            valid_samples.append(sample)

    elif mode == 'Mapping':
        print("Warning: Mapping mode. The sample and subject mapping will be built by the keys.")
        # Read CSV and keep only the relevant columns
        task_description_df = pd.read_csv(task_description_csv)[[subject_key, sample_key]]
        # Get the unique subject names
        unique_subject_names = task_description_df[subject_key].unique()
        # Create a list to hold valid samples
        valid_samples = []

        for subject_name in unique_subject_names:
            # Slicing the DataFrame rows where subject_key == subject_name, and extracting the sample_key column
            subject_samples = task_description_df.loc[
                task_description_df[subject_key] == subject_name, sample_key].tolist()
            subject_sample_dict[subject_name] = subject_samples

            # Extend the valid_samples list
            valid_samples.extend(subject_samples)

    else:
        assert data_path is not None
        # Get a list of all subject samples from the CSV file
        all_subject_names = list(pd.read_csv(task_description_csv)[subject_key])

        # Get a list of all WSI sample folders from the dataset directory
        all_wsi_folders = [
            wsi_folder for wsi_folder in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, wsi_folder)) and not wsi_folder.startswith('task-settings')
        ]
        print(f'Found {len(all_wsi_folders)} WSI folders under {data_path}')

        # Default mode: Treat all WSIs as belonging to the same subject,
        print("Warning: Non-TCGA mode. All WSIs within the same folder will be treated as having the same label.")
        # here we assume the subject name will appear in the folder name of the sample
        valid_samples = []
        for subject_name in all_subject_names:
            # Set a dic for each subject_name
            subject_sample_dict.setdefault(subject_name, [])
            # Go through the folders, take the folders with subject_name inside
            for sample in all_wsi_folders:
                if subject_name in sample:
                    subject_sample_dict[subject_name].append(sample)
                    valid_samples.append(sample)

    return subject_sample_dict, missing_sample_wsis, redundant_sample_wsis, valid_samples, \
        positive_samples, negative_samples


def make_split_by_subject(subject_sample_dict, k=1, test_ratio=0.2):
    """Generate train, validation, and test splits by subject.

    This function creates splits for a dataset grouped by subjects, ensuring that
    the data is split at the subject level, not at the sample level. The test set
    is separated first, and then the remaining subjects are split into training
    and validation sets using K-Fold cross-validation.

    Args:
        subject_sample_dict (dict): A dictionary where keys are subject IDs and values are
            lists of sample identifiers (e.g., WSI file paths) belonging to each subject.
        k (int): Number of splits for K-Fold cross-validation (for training and validation).
        test_ratio (float, optional): The proportion of subjects to allocate to the test set.
            Defaults to 0.2.

    Returns:
        dict: A dictionary containing split information for each fold. Each key is a fold number,
        and the corresponding value is a dictionary with keys 'Train', 'Val', and 'Test',
        where each contains a list of sample identifiers for the respective split.
    """

    # Shuffle the subject names
    shuffled_subjects = list(subject_sample_dict.keys())
    random.shuffle(shuffled_subjects)

    # Initialize K-Fold cross-validation
    # Due to KFold only accept k>1, if k==1, we make n_splits=5
    kfold = KFold(n_splits=k if k > 1 else 5, shuffle=True, random_state=42)

    # Split the dataset into test and train/validation sets
    train_val_subjects = np.array(shuffled_subjects[int(len(shuffled_subjects) * test_ratio):])
    test_subjects = np.array(shuffled_subjects[:int(len(shuffled_subjects) * test_ratio)])

    # Store fold information
    fold_info = {}
    print('Total subjects num:', len(shuffled_subjects))

    # Perform K-Fold splitting on the train/validation set
    for f, (train_idx, val_idx) in enumerate(kfold.split(train_val_subjects)):
        fold = f + 1
        fold_info.setdefault(fold, {})

        # Get samples for train, validation, and test splits
        train_data = []
        val_data = []
        test_data = []
        for subject in train_val_subjects[train_idx]:
            train_data.extend(subject_sample_dict[subject])
        for subject in train_val_subjects[val_idx]:
            val_data.extend(subject_sample_dict[subject])
        for subject in test_subjects:
            test_data.extend(subject_sample_dict[subject])

        # Store split data in fold_info
        fold_info[fold] = {
            'Train': train_data,
            'Val': val_data,
            'Test': test_data
        }

        # break if k == 1
        if k == 1:
            break

    return fold_info


def split_mapping(fold_info, task_description_df, k, sample_key, split_target_key):
    """
    This modifies the task_description_df to put in the split information
    Args:
        fold_info (dict): A dictionary containing fold split information with keys 'Train', 'Val',
            and 'Test', each mapping to lists of samples for training, validation, and testing.
        task_description_df (pandas df): containing task description data.
        subject_sample_dict (dict): A dictionary where keys are subject IDs and values are
            lists of sample identifiers (e.g., WSI file paths) belonging to each subject.
        k (int): Number of splits for K-Fold cross-validation (for training and validation).
        sample_key (str): Column name to be used or created for storing sample identifiers.
        split_target_key (str): Column name (to be used or created) for storing split labels ('Train', 'Val', 'Test').
    """
    # Create a mapping dictionary for sample-to-split assignment
    split_mapping = {}

    for fold in fold_info:
        train_data = fold_info[fold]['Train']
        val_data = fold_info[fold]['Val']
        test_data = fold_info[fold]['Test']

        # Print split sizes for the current fold
        print(f'Fold {fold}: Train/Val/Test = {len(train_data)}/{len(val_data)}/{len(test_data)}')

        for phase, data in zip(['Train', 'Val', 'Test'], [train_data, val_data, test_data]):
            for sample in data:
                if k == 1:
                    split_mapping[sample] = phase
                else:
                    key_fold = f'{split_target_key}_{k}fold-{fold}'
                    if sample not in split_mapping:
                        split_mapping[sample] = {}
                    split_mapping[sample][key_fold] = phase

    # Apply the mapping to the DataFrame
    print('Apply the mapping to the DataFrame')
    if k == 1:
        task_description_df[split_target_key] = task_description_df[sample_key].map(split_mapping)
    else:
        for fold in tqdm(fold_info, total=len(fold_info), unit="fold",
                         desc=f'Mapping csv for folds '):
            key_fold = f'{split_target_key}_{k}fold-{fold}'
            task_description_df[key_fold] = task_description_df[sample_key].map(
                lambda x: split_mapping.get(x, {}).get(key_fold, None))

    return task_description_df


def modify_csv_data(fold_info, subject_sample_dict, task_description_csv, subject_key,
                    sample_key, split_target_key, mode='TCGA', negative_samples=None):
    """Modify CSV data with split and sample information for a dataset.

    This function processes a task description CSV file by adding split and sample information
    based on given fold information and patient-wise WSI data. It also handles negative samples
    (e.g., tumor-free WSIs) differently depending on the dataset mode.

    Args:
        fold_info (dict): A dictionary containing fold split information with keys 'Train', 'Val',
            and 'Test', each mapping to lists of samples for training, validation, and testing.
        subject_sample_dict (dict): A dictionary mapping each sample ID to their associated subject ID (as dict key)

        task_description_csv (str): Path to the CSV file containing task description data.

        subject_key (str): Column name in the CSV that identifies patient/sample IDs.
        sample_key (str): Column name to be used or created for storing sample identifiers.
        split_target_key (str): Column name (to be used or created) for storing split labels ('Train', 'Val', 'Test').

        mode (str, optional): Dataset mode; defaults to 'TCGA'. If not 'TCGA', negative samples are ignored.
        negative_samples (list): A list of samples labeled as tumor-free.
    Returns:
        pd.DataFrame: A new DataFrame with updated split and sample information.
    """
    # Read the CSV file and load it into a DataFrame
    df = pd.read_csv(task_description_csv)
    k = len(fold_info.keys())  # a dic of several fold

    # Ensure split_target_key columns exist in the DataFrame
    # THis also flush the split information if it's there in the previous csv
    if k == 1:
        df[split_target_key] = ""
    else:
        for fold in fold_info:
            split_target_key_fold = f'{split_target_key}_{k}fold-{fold}'
            df[split_target_key_fold] = ""

    # Ensure slide_id column exist in the DataFrame
    if sample_key not in df.columns:
        df[sample_key] = ""

    # Add a list of new columns, due to different mode
    new_columns_list = csv_get_new_columns(mode)
    for new_col in new_columns_list:
        if new_col not in df.columns:
            df[new_col] = ""

    # Process subject/ sample information
    if mode == 'TCGA':
        # Initialize a new DataFrame for storing the updated data
        df_new = pd.DataFrame(columns=df.columns)  # [0 rows x N columns]

        for subject in tqdm(subject_sample_dict, total=len(subject_sample_dict), unit="subject",
                            desc=f'Writing csv for subjects ', disable=False):
            for sample in tqdm(subject_sample_dict[subject], total=len(subject_sample_dict[subject]),
                               unit="sample", desc=f'Writing csv for its samples ', disable=True):
                # Extract rows related to the current sample (as string subset to be generalized)
                sample_label = df[df[subject_key].apply(lambda x: str(x) in sample)].copy()
                # if sample_label is empty, use sample_key
                if sample_label.empty:
                    sample_label = df[df[sample_key].apply(lambda x: str(x) in sample)].copy()

                # Ensure slide_id column exist in the DataFrame, rename to sample
                sample_label[sample_key] = sample  # Add sample identifier, (may be as string subset to be generalized)
                sample_label = sample_label.squeeze()  # Convert from DataFrame to Series

                # Handle negative samples
                is_normal = True if sample in negative_samples else False
                sample_label = csv_check_normal_tcga(sample_label, sample, subject_key, sample_key, normal=is_normal)

                # Append the processed rows (as series) to the new DataFrame
                df_new.loc[len(df_new)] = sample_label
    else:
        df_new = df

    # Add fold information to the DataFrame
    df_new = split_mapping(fold_info, df_new, k, sample_key, split_target_key)

    return df_new


def build_WSI_data_split_for_csv(task_description_csv, data_path=None, subject_key='patient_id', sample_key='slide_id',
                                 test_ratio=0.2, k=1, mode='TCGA', split_target_key='split'):
    """
    Modify a CSV file to include data split information for k-fold cross-validation.

    This function processes a CSV file containing task descriptions, and appends
    columns to indicate train/val/test split information for k-fold cross-validation
    based on the provided dataset structure and mode.

    Args:
        task_description_csv (str): Path to the task description CSV file.
        data_path (str): Root directory containing WSI sample folders.
        subject_key (str, optional): Column name in the CSV that identifies patient/sample IDs.
            Defaults to 'patient_id'. we build the split based on this

        sample_key (str, optional): Column name to store slide/sample IDs. Defaults to 'slide_id'.
                    we based on subject_name to find the sample_name of each sample belonging to certain subject,
                    this sample_key collumn should be the name of each sample

        test_ratio (float, optional): Proportion of data allocated to the test set. Defaults to 0.2.
        k (int, optional): Number of folds for k-fold cross-validation. Defaults to 1.
        mode (str, optional): Dataset mode; defaults to 'TCGA'. Determines how negative samples
            are handled and patient-sample mapping is constructed.
        split_target_key (str, optional): Column name for storing split labels ('split'). Defaults to 'split'.

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
    # Map WSIs to patients and identify missing, redundant, and valid WSIs
    subject_sample_dict, missing_sample_wsis, redundant_sample_wsis, valid_samples, positive_samples, \
        negative_samples = build_WSI_subject_sample_dict(data_path, task_description_csv,
                                                         subject_key, sample_key, mode='TCGA')

    print(f'Valid subjects: {len(subject_sample_dict)} | '
          f'Missing subjects: {len(missing_sample_wsis)} | '
          f'Redundant subjects: {len(redundant_sample_wsis)} | '
          f'Valid samples: {len(valid_samples)} | '
          f'Negative samples: {len(negative_samples)}')

    # Create train/val/test splits by subject
    fold_info = make_split_by_subject(subject_sample_dict, k, test_ratio)

    # Modify CSV data to include negative sample labels and k-fold split labels
    final_task_description_df = modify_csv_data(
        fold_info, subject_sample_dict, task_description_csv, subject_key, sample_key,
        split_target_key, mode, negative_samples)

    # Write the updated DataFrame back to the CSV file
    final_task_description_df.to_csv(task_description_csv, index=False)


def build_ROI_data_split_for_csv(task_description_csv, subject_key='patient_id', sample_key='slide_id',
                                 test_ratio=0.2, k=1, mode='ROI', split_target_key='split'):
    """
    Modify a CSV file to include data split information for k-fold cross-validation.

    This function processes a CSV file containing task descriptions, and appends
    columns to indicate train/val/test split information for k-fold cross-validation
    based on the provided dataset structure and mode.

    Args:
        task_description_csv (str): Path to the task description CSV file.

        subject_key (str, optional): Column name in the CSV that identifies patient/sample IDs.
            Defaults to 'patient_id'.
        sample_key (str, optional): Column name to store slide/sample IDs. Defaults to 'slide_id'.
        test_ratio (float, optional): Proportion of data allocated to the test set. Defaults to 0.2.

        k (int, optional): Number of folds for k-fold cross-validation. Defaults to 1.
        mode (str, optional): Dataset mode; defaults to 'ROI'.

        split_target_key (str, optional): Column name for storing split labels ('split'). Defaults to 'split'.

    Returns:
        None: Modifies the input CSV file by appending new columns for data splits.

    Process:
        1. Determine the total number of folds:
           - If `k > 1`, use `k` folds for cross-validation.
           - If `k == 1`, default to using a single fold from a 5-fold split.
        2. Map ROIs to be split:
           - Using `build_patient_sample_dict`, match patient IDs to WSI folders.
           - Identify missing, redundant, and valid WSIs.
        3. Split the data into train/val/test:
           - Use `make_split_by_patient` to generate split information at the patient level.
        4. Modify the CSV file:
           - Use `modify_csv_data` to update the CSV with split labels and handle negative samples.
        5. Save the updated CSV file back to the original path.

    """
    # Get a list of all subject samples from the CSV file
    # (sample is unique while many sample may have the same subject name)
    task_description_df = pd.read_csv(task_description_csv)
    all_subject_names = set(task_description_df[subject_key])
    all_sample_names = list(task_description_df[sample_key])
    print(f'Found {len(all_subject_names)} subjects and {len(all_sample_names)} samples')

    # Map samples to subjects
    subject_sample_dict = {}
    for subject_name in all_subject_names:
        sample_names = list(task_description_df[task_description_df[subject_key] == subject_name][sample_key])
        subject_sample_dict[subject_name] = []
        for sample_name in sample_names:
            subject_sample_dict[subject_name].append(sample_name)

    # Create train/val/test splits by subject
    fold_info = make_split_by_subject(subject_sample_dict, k, test_ratio)

    # Modify CSV data to include negative sample labels and k-fold split labels
    final_task_description_df = modify_csv_data(
        fold_info, subject_sample_dict, task_description_csv, subject_key, sample_key, split_target_key, mode)

    # Write the updated DataFrame back to the CSV file
    final_task_description_df.to_csv(task_description_csv, index=False)


###########################################
###       Support/compilable Functions      ###
###########################################


def load_pickle_data_split_for_csv(task_description_csv, subject_id='slide_id',
                                   split_target_key='split', input_pkl_rootpath=None,
                                   mode='TCGA', k=1):
    """
    This load the previous pkl split list and write the split information into csv

    Args:
        task_description_csv:
        subject_id:
        split_target_key:
        input_pkl_rootpath:
        mode:
        k:

    Returns:

    Demo
    # load 5 fold pkl to csv
    load_pickle_data_split_for_csv(
        task_description_csv='/data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv',
        subject_id='patient_id', key='fold_information', input_pkl_rootpath='/data/ai4dd/TCGA_5-folds',
        mode='TCGA', k=5)
    """
    import pickle

    def write_csv_data(task_description_csv, subject_id, id_data, split_target_key='split', split_val='Train'):
        """
        legacy, will be removed!

        Edit the CSV file by adding (if not there, otherwise edit) a column name of key (such as 'split')

        Parameters:
        - task_description_csv: Path to the CSV file.
        - subject_id: The name of the column that contains the IDs to match with id_data.
        - id_data: A list of values corresponding to the subject_id column, for which the key column should be updated.
        - key: The name of the column to add or update. Defaults to 'split'.
        - val: The value to set in the key column for the matching rows. Defaults to 'train'.
        """
        # Load the CSV into a DataFrame
        df = pd.read_csv(task_description_csv)

        # If the key column does not exist, create it and fill with empty strings, else will rewrite
        if split_target_key not in df.columns:
            df[split_target_key] = ""

        # Update the rows where the id_key matches any of the values in id_data
        df.loc[df[subject_id].isin(id_data), split_target_key] = split_val

        # Write the updated DataFrame back to the CSV
        df.to_csv(task_description_csv, index=False)

    def load_data(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    if k == 1:
        Train_list = load_data(os.path.join(input_pkl_rootpath, 'Train.pkl'))
        Val_list = load_data(os.path.join(input_pkl_rootpath, 'Val.pkl'))
        Test_list = load_data(os.path.join(input_pkl_rootpath, 'Test.pkl'))

        # in TCGA the csv subject_id is patient name so we need to write the patient split into it
        if mode == 'TCGA':
            Train_list = [sample[:12] for sample in Train_list]
            Val_list = [sample[:12] for sample in Val_list]
            Test_list = [sample[:12] for sample in Test_list]

        write_csv_data(task_description_csv, subject_id=subject_id, id_data=Train_list,
                       split_target_key=split_target_key, split_val='Train')
        write_csv_data(task_description_csv, subject_id=subject_id, id_data=Val_list,
                       split_target_key=split_target_key, split_val='Val')
        write_csv_data(task_description_csv, subject_id=subject_id, id_data=Test_list,
                       split_target_key=split_target_key, split_val='Test')
    else:
        for fold in range(1, k + 1):
            fold_pkl_rootpath = os.path.join(input_pkl_rootpath, 'task-settings-' + str(k) + 'folds_fold-' + str(fold))

            Train_list = load_data(os.path.join(fold_pkl_rootpath, 'Train.pkl'))
            Val_list = load_data(os.path.join(fold_pkl_rootpath, 'Val.pkl'))
            Test_list = load_data(os.path.join(fold_pkl_rootpath, 'Test.pkl'))

            # in TCGA the csv subject_id is patient name so we need to write the patient split into it
            if mode == 'TCGA':
                Train_list = [sample[:12] for sample in Train_list]
                Val_list = [sample[:12] for sample in Val_list]
                Test_list = [sample[:12] for sample in Test_list]

            write_csv_data(task_description_csv, subject_id=subject_id, id_data=Train_list,
                           split_target_key=split_target_key + '_{}fold-{}'.format(k, fold), split_val='Train')
            write_csv_data(task_description_csv, subject_id=subject_id, id_data=Val_list,
                           split_target_key=split_target_key + '_{}fold-{}'.format(k, fold), split_val='Val')
            write_csv_data(task_description_csv, subject_id=subject_id, id_data=Test_list,
                           split_target_key=split_target_key + '_{}fold-{}'.format(k, fold), split_val='Test')

    print('done')


# task config tools:
def build_task_config_settings(task_description_df, new_labels,
                               one_hot_table={}, all_task_dict={}, cls_task=None, max_possible_values=100):
    """
    Configure task settings by processing labels from the task description DataFrame.

    This function analyzes each label in the provided list of new labels to determine whether
    it represents a regression or classification task. It updates the `one_hot_table` and
    `all_task_dict` accordingly, while ensuring no duplicate labels exist. It also skips labels
    that are invalid (e.g., containing too many unique values or constant values) and returns
    them for review.

    Args:
        task_description_df (pd.DataFrame): DataFrame containing task description data, where
            each column represents a label.
        new_labels (list of str): List of column names (labels) from `task_description_df` to
            process and include in the task configuration.
        one_hot_table (dict, optional): Dictionary to store one-hot encoding mappings for
            classification tasks. Default is an empty dictionary.
        all_task_dict (dict, optional): Dictionary to store the type of each task, mapping
            label names to 'float' for regression or 'list' for classification. Default is
            an empty dictionary.
        cls_task (list, optional): A list of tasks to be regarded as classification tasks.
            Default to None.
        max_possible_values (int, optional): Maximum number of unique values allowed for a
            label to be considered a classification task. Labels exceeding this limit are
            marked invalid. Default is 100.
    """
    assert all(label in task_description_df.columns for label in new_labels)

    selected_new_labels = []
    invalid_labels = []  # store invalid labels and return

    for label in new_labels:
        # new label should not be in existing config
        if label in one_hot_table or label in all_task_dict:
            raise ValueError(f'Duplicate label: {label}')

        # get the list of all possible values under the current column
        content_list = list(task_description_df[label].value_counts().keys())  # this also removes the duplicates
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
            if cls_task is not None:
                if label in cls_task:
                    str_flag = True
                else:
                    str_flag = False
            else:
                str_flag = False

        # skip if no valid value
        if len(valid_content_list) == 0:
            invalid_labels.append(label)
            continue

        if not str_flag:
            all_task_dict[label] = 'float'
            if len(valid_content_list) == 0:
                continue  # skip if no valid value
            print(f'Regression task added to task settings: {label}')
        else:  # maybe it's a cls task
            # skip if too many possible values or the value is constant
            if len(valid_content_list) > max_possible_values or len(valid_content_list) == 1:
                invalid_labels.append(label)
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


def build_yaml_config_from_csv(task_description_csv, task_settings_path, dataset_name=None,
                               tasks_to_run=None, cls_task=None, mode='TCGA', max_tiles=1000000, shuffle_tiles=True,
                               excluding_list=(
                                       'WSI_name', 'slide_id', 'split', 'Folder', 'File_name', 'Slide', 'Tissue',
                                       'Type', 'Disposition', 'mpp', 'Sub folder', 'Status', 'Date',
                                       'Time', 'Patient ID', 'uniquePatientKey', 'FORM_COMPLETION_DATE',
                                       'OTHER_PATIENT_ID', 'studyId', 'tile_image_path', 'tile_y', 'tile_x',
                                       'tile_name', 'split'), yaml_config_name='task_configs.yaml'):
    """
    Build a YAML configuration file from a CSV file containing task descriptions.

    Parameters:
    task_description_csv (str): Path to the original task_description_df .csv file.
    task_settings_path (str): Output directory for the YAML file. (task-settings path)

    dataset_name (str): Name of the dataset. this is for recording in the yaml
    tasks_to_run (list): A list of tasks to run.
    cls_task (list): A list of tasks to be regarded as classification tasks.
    max_tiles (int): Maximum number of tiles. Default is 1000000.
    shuffle_tiles (bool): Whether to shuffle tiles or not. Default is True.
    excluding_list (tuple): List of columns to exclude. Default is ('WSI_name', ...).
                            the attribute starts with 'split' will be ignored as they are designed for control split
                            EG: 'split_nfold-k', n is the total fold number and k is the fold index

    yaml_config_name (str): Name of the yaml config file in the task_settings_folder
    """

    try:
        task_description_df = read_df_from_file(task_description_csv)
    except:  # no valid label selected
        raise ValueError('Invalid input!', task_description_csv)

    one_hot_table, all_task_dict = {}, {}  # todo we put here for future loading existing configs
    excluding_list = list(excluding_list)

    # select columns in csv to be used as the labels.
    # By default, all columns except subject_id will be used as label.
    tentative_task_labels = [col for col in task_description_df.columns if col not in excluding_list]

    if tasks_to_run is not None:
        for task in tasks_to_run:
            assert task in tentative_task_labels
    else:
        # take all tasks as valid tasks
        tasks_to_run = tentative_task_labels

    one_hot_table, all_task_dict, selected_new_labels, invalid_labels = \
        build_task_config_settings(task_description_df, tentative_task_labels, one_hot_table, all_task_dict, cls_task)

    # remove invalid labels from tasks_to_run
    tasks_to_run = [task for task in tasks_to_run if task not in invalid_labels and task in all_task_dict]

    print(f'#' * 30)
    print(f'Add labels to config: {selected_new_labels}')
    print(f'#' * 30)

    if max_tiles is not None and shuffle_tiles is not None:
        config = {
            'name': dataset_name,
            'tasks_to_run': tasks_to_run,
            'all_task_dict': all_task_dict,
            'one_hot_table': one_hot_table,
            'max_tiles': max_tiles,
            'shuffle_tiles': shuffle_tiles,
            'mode': mode}
    else:
        config = {
            'name': dataset_name,
            'tasks_to_run': tasks_to_run,
            'all_task_dict': all_task_dict,
            'one_hot_table': one_hot_table,
            'mode': mode}
    if not os.path.exists(task_settings_path):
        os.makedirs(task_settings_path)

    yaml_output_path = os.path.join(task_settings_path, yaml_config_name)
    if os.path.exists(yaml_output_path):
        os.remove(yaml_output_path)

    with open(yaml_output_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    return all_task_dict, one_hot_table


def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    return config


def build_split_and_task_configs(data_path, task_description_csv, dataset_name,
                                 tasks_to_run, cls_task=None, subject_key='slide_id', sample_key='slide_id',
                                 split_target_key='fold_information',
                                 test_ratio=0.2, k=1,
                                 task_setting_folder_name='task-settings',
                                 yaml_config_name='task_configs.yaml',
                                 mode='TCGA',
                                 excluding_list=('WSI_name', 'slide_id', 'split', 'Folder', 'File_name',
                                                 'Slide', 'Tissue', 'Type', 'Disposition', 'mpp', 'Sub folder',
                                                 'Status', 'Date', 'Time', 'Patient ID', 'uniquePatientKey',
                                                 'FORM_COMPLETION_DATE', 'OTHER_PATIENT_ID', 'studyId',
                                                 'tile_image_path', 'tile_y', 'tile_x', 'tile_name', 'split')):
    """
    this function generate a task_setting_folder in data_path,
        and there are 1 task_description.csv inside and 1 task_configs.yaml inside
        the task_description.csv is label and split information retrieve/ generated from original task_description_csv
        the yaml_config is a yaml file recording the running task information (such as one-hot mapping for CLS)

    Parameters:
    data_path (str): a root folder of multiple WSI folders,
    task_description_csv (str): an original label csv file

    dataset_name (str): Name of the dataset. this is for recording in the yaml

    tasks_to_run (list): A list of tasks to run.

    cls_task (list): A list of tasks to be regarded as classification tasks.

    subject_key (str, optional): Column name in the CSV that identifies as individual to be shuffled
                (if you need patient wise shuffle, to track the id of patient) Defaults to 'slide_id'.
    sample_key (str, optional): Column name to store slide/sample IDs. Defaults to 'slide_id'.

    split_target_key (str, optional): Column name for storing split labels ('split'). Defaults to 'split'.

    test_ratio (float, optional): Proportion of data allocated to the test set. Defaults to 0.2.
    k (int, optional): Number of folds for k-fold cross-validation. Defaults to 1.

    task_setting_folder_name (str): Output directory for the YAML file. (folder in root path)
    yaml_config_name (str): Name of the yaml config file in the task_settings_folder

    mode (str, optional): Dataset mode; defaults to 'TCGA'. Determines how negative samples
            are handled and patient-sample mapping is constructed.

    excluding_list (tuple): List of columns to exclude. Default is ('WSI_name', ...).
                            the attribute starts with 'split' will be ignored as they are designed for control split
                            EG: 'split_nfold-k', n is the total fold number and k is the fold index

    """
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

    # build excluding list
    excluding_list = list(excluding_list)
    excluding_list.extend([subject_key, split_target_key])
    excluding_list = list(set(excluding_list))

    if k > 1:
        excluding_list.extend([split_target_key + '_{}fold-{}'.format(k, fold) for fold in range(1, k + 1)])

    # build data_split_for_csv
    if mode in ['tile', 'tiles', 'ROI', 'ROIs']:
        # ROI csv processing no need the root path
        build_ROI_data_split_for_csv(task_description_csv, subject_key=subject_key, sample_key=sample_key,
                                     test_ratio=test_ratio, k=k, mode=mode, split_target_key=split_target_key)

        build_yaml_config_from_csv(task_description_csv, output_dir, dataset_name=dataset_name,
                                   tasks_to_run=tasks_to_run,
                                   cls_task=cls_task,
                                   max_tiles=None, shuffle_tiles=None, mode=mode,
                                   excluding_list=excluding_list,
                                   yaml_config_name=yaml_config_name)
    else:
        # WSI csv processing need the root path to ensure the WSI samples are correctly seperated
        build_WSI_data_split_for_csv(task_description_csv, data_path, subject_key=subject_key, sample_key=sample_key,
                                     test_ratio=test_ratio, k=k, mode=mode, split_target_key=split_target_key)
        build_yaml_config_from_csv(task_description_csv, output_dir, dataset_name=dataset_name,
                                   tasks_to_run=tasks_to_run,
                                   cls_task=cls_task,
                                   max_tiles=1000000, shuffle_tiles=True, mode=mode,
                                   excluding_list=excluding_list,
                                   yaml_config_name=yaml_config_name)

    # check
    load_yaml_config(os.path.join(data_path, task_setting_folder_name, yaml_config_name))

    print(f'Finished! task_description_csv: {task_description_csv}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build split and task configs.')

    parser.add_argument('--data_path', type=str, default='/home/BigModel/Archive',
                        help='Root path for the datasets')
    parser.add_argument('--task_description_csv',
                        default='/home/BigModel/Archive/labels.csv', type=str,
                        help='label csv file path, if given it will be copied to the task_setting_folder as '
                             'task_description.csv '
                             'default none will go to find task_description.csv in task_setting_folder')

    parser.add_argument('--subject_key', type=str, default='patientID',
                        help='Column name in the CSV that identifies as individual to be shuffled '
                             '(if you need patient wise shuffle, to track the id of patient) ')
    parser.add_argument('--sample_key', type=str, default='slide_id',
                        help='Column name to store slide IDs, if no slide id, this one should set to subject id')
    parser.add_argument('--split_target_key', type=str, default='fold_information',
                        help='Key to split the dataset')

    parser.add_argument('--task_setting_folder_name', type=str, default='task-settings-5folds',
                        help='Output directory for the YAML file. (folder in root path)')
    parser.add_argument('--mode', type=str, default='TCGA',
                        help='Dataset mode; Determines how negative samples are handled and '
                             'patient-sample mapping is constructed.')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of folds for k-fold cross-validation. Defaults to 1.')

    parser.add_argument("--fix_random_seed", action="store_true", help="fix random seed to enable reproduction")
    parser.add_argument('--dataset_name', type=str, default='TCGA-LUNG',
                        help='Name of the dataset. this is for recording in the yaml')
    parser.add_argument('--cls_task', type=str, default=None,
                        help='Task list, separated by "%", if not None, all tasks under cls_task will be regarded as classification task.')
    parser.add_argument('--tasks_to_run', type=str,
                        default=None,
                        help='Tasks to run, separated by "%", default None will take all tasks in csv')

    args = parser.parse_args()

    if args.fix_random_seed:
        random.seed(42)  # setup random seed to make sure split always keep same

    # Convert to a list
    tasks_to_run = args.tasks_to_run.split('%') if \
        (args.tasks_to_run is not None and args.tasks_to_run != 'None') else None
    cls_task = args.cls_task.split('%') if \
        (args.cls_task is not None and args.cls_task != 'None') else None

    build_split_and_task_configs(
        data_path=args.data_path,
        task_description_csv=args.task_description_csv,
        dataset_name=args.dataset_name,
        tasks_to_run=tasks_to_run,
        cls_task=cls_task,
        subject_key=args.subject_key,
        sample_key=args.sample_key,
        split_target_key=args.split_target_key,
        task_setting_folder_name=args.task_setting_folder_name,
        mode=args.mode,
        k=args.k
    )
    # demo with TCGA
    '''
    python Slide_dataset_tools.py --data_path /data/BigModel/embedded_datasets/ \
    --task_description_csv /home/BigModel/Archive/dataset_csv/TCGA_Log_Transcriptome_Final.csv \
    --subject_key patient_id \
    --sample_key slide_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings-5folds \
    --mode TCGA \
    --dataset_name coad-read \
    --tasks_to_run iCMS%CMS%MSI.status%EPCAM%COL3A1%CD3E%PLVAP%C1QA%IL1B%MS4A1%CD79A \
    --k 5

    python Slide_dataset_tools.py --data_path /data/BigModel/embedded_lung/ \
    --task_description_csv /data/BigModel/TCGA-LUNG-task_description.csv \
    --subject_key WSI_name \
    --sample_key slide_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings \
    --mode None \
    --dataset_name lung-mix \
    --tasks_to_run purity%FRACTION_GENOME_ALTERED%AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%lung-cancer-subtyping \
    --k 1
    '''

    '''
    # load 5 fold pkl to csv (if we need to load 5 fold from previous split)
    load_pickle_data_split_for_csv(
        task_description_csv='/data/BigModel/embedded_datasets/task-settings-5folds/20240827_TCGA_log_marker10.csv',
        subject_key='patient_id', split_target_key='fold_information', input_pkl_rootpath='/data/ai4dd/TCGA_5-folds',
        mode='TCGA', k=5)
    '''

    # demo with SGH data
    '''
    python Slide_dataset_tools.py --data_path '/data/hdd_1/BigModel/qupath_embedded_datasets' \
    --task_description_csv /home/zhangty/Desktop/BigModel/code/Archive/dataset_csv/SGH_Transcriptome_Log_Combined.csv \
    --subject_key slide_id \
    --sample_key slide_id \
    --split_target_key fold_information \
    --task_setting_folder_name task-settings-5folds \
    --mode SGH \
    --dataset_name SGH_aidd \
    --k 5
    '''
    # demo with ROI SO bulk tiles
    '''
    python Slide_dataset_tools.py --data_path '/data/hdd_1/BigModel/SO/tiled_data' \
    --task_description_csv /data/hdd_1/BigModel/SO/tiled_data/filtered_tile_labels.csv \
    --subject_key slide_id \
    --sample_key tile_image_path \
    --task_setting_folder_name task-settings-5folds \
    --mode ROI \
    --dataset_name GIS-SO-pseudo-bulk \
    --k 5
    '''
