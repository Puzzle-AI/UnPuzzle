import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

################################
### Basic ###
################################

def run_map_mp(func, argument_list, num_processes='', is_tqdm=True):
    """Multi-threading with progress bar
    Ref: https://zhuanlan.zhihu.com/p/359369130

    Args:
        func (func): Target function.
        argument_list (list): Argument list. Example format: [(a1, b1), (a2, b2)]
        num_processes (str, optional): The number of processes. Defaults to the number of threads - 3.
        is_tqdm (bool, optional): Whether to display progress bar (using tqdm). Defaults to True.

    Returns:
        result_list_tqdm: Output list of each thread.
    """
    result_list_tqdm = []
    try:
        if not num_processes:
            num_processes = min(cpu_count() - 3, len(argument_list))
        pool = Pool(processes=num_processes)
        print('start running multiprocess using {} threads'.format(num_processes))

        # Use pool.starmap to allow multi-parameter for func
        if is_tqdm:

            # Here because starmap can only return result after fully finished, it is not capable to 
            # operate with tqdm progress bar.

            # In this case, I update the progress bar every num_processes processes, which may slow down 
            # the process a bit but enable observation.

            pbar = tqdm(total=len(argument_list))
            idx = 0
            for idx in range(0, len(argument_list) // num_processes + 1):
                for result in pool.starmap(func=func, iterable=argument_list[idx*num_processes : min((idx+1)*num_processes, len(argument_list))]):
                    result_list_tqdm.append(result)
                pbar.update(min(num_processes, len(argument_list)-idx*num_processes))
                idx += 1
        else:
            for result in pool.starmap(func=func, iterable=argument_list):
                result_list_tqdm.append(result)
        pool.close()
        pool.join()

    except:
        result_list_tqdm = list(map(func, argument_list))   
    return result_list_tqdm


def setup_seed(seed):
    """Fix up the random seed

    Args:
        seed (int): Seed to be applied
    """
    import random
    random.seed(seed)
    np.random.seed(seed)


################################
### Tools ###
################################


def fetch_wsi_folder_list(data_dir='./'):
    # Find all child folder under current data directory
    folder_list = []

    for folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, folder)):
            folder_list.append(folder)

    folder_list = list(set(folder_list))

    return folder_list


def one_hot_to_int(df):
    """Change one-hot label to int value
    """
    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Apply the transformation using apply and a lambda function
        df[col] = df[col].apply(
            lambda attr_value: np.argmax(attr_value) if isinstance(attr_value, list) else attr_value)
        
    return df


def one_hot_to_str(df, one_hot_table):
    """Change the list type one-hot labels within df into str value.

    Args:
        df (pandas.DataFrame): Dataframe with some one hot label values in list type.
        one_hot_table (dict): One hot table from string to list value, e.g., {str: list}.

    Returns:
        _type_: _description_
    """

    # Get one hot to string dict
    one_hot_to_str = {}
    for col in one_hot_table:
        if col not in one_hot_to_str:
            one_hot_to_str[col] = {}
        for str_value in one_hot_table[col]:
            one_hot_to_str[col][str(one_hot_table[col][str_value])] = str_value

    # Replace all the list with string value 
    for col in df.columns:
        df[col] = df[col].apply(
            lambda attr_value: one_hot_to_str[col][str(attr_value)] 
            if isinstance(attr_value, list) else attr_value)

    return df


################################
### Data Retrival ###
################################

def retrieve_df_from_dataset(dataset_path, train_val_test=False):
    # init one_hot_table
    if train_val_test:
        one_hot_table_path = os.path.join(dataset_path, 'Train', 'task-settings', 'one_hot_dict.json')
    else:
        one_hot_table_path = os.path.join(dataset_path, 'task-settings', 'one_hot_dict.json')

    try:
        with open(one_hot_table_path, 'r') as f:
            one_hot_table = json.load(f)
    except:
        raise ValueError('One hot table not found, aborted')

    # fetch all WSI labels within current dataset
    if train_val_test:
        # find WSIs in train, val and test set
        df_tmp = []
        for phase in ['Train', 'Val', 'Test']:
            wsi_folder_list = fetch_wsi_folder_list(os.path.join(dataset_path, phase))
            for wsi_folder in wsi_folder_list:
                if wsi_folder != 'task-settings':
                    with open(os.path.join(dataset_path, phase, wsi_folder, 'task_description.json'), 'r') as f:
                        json_dict = json.load(f)
                        df_tmp.append(json_dict)

    else:
        # only find in current set
        wsi_folder_list = fetch_wsi_folder_list(dataset_path)
        df_tmp = []
        for wsi_folder in wsi_folder_list:
            if wsi_folder != 'task-settings':
                with open(os.path.join(dataset_path, wsi_folder, 'task_description.json'), 'r') as f:
                    json_dict = json.load(f)
                    df_tmp.append(json_dict)

    # change format into dataframe
    df = pd.DataFrame(df_tmp)

    # convert to numeric if possible 
    df = df.apply(pd.to_numeric, errors='ignore')

    # change from one hot label to string
    df = df.copy(deep=True)
    df = one_hot_to_str(df, one_hot_table)

    return df


def retrieve_df_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df