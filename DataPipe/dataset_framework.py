"""
MTL dataset framework       Script  ver: Jan 9th 2025 12:00
"""

import time
import sys
import argparse
from pathlib import Path

# For convenience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent.parent))  # Go up 2 levels
try:
    from DataPipe.Slide_dataset_tools import *
    from DataPipe.wsi_tools import read_tile_name_for_loc_y_x
except:
    from Slide_dataset_tools import *
    from wsi_tools import read_tile_name_for_loc_y_x

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.data.dataloader import default_collate
import h5py


# pseudo Bulk dataset builder
class Bulk_ROI_Dataset(Dataset):
    def __init__(self, root_path: str,
                 task_description_csv: str = None,
                 task_setting_folder_name: str = 'task-settings-5folds',
                 task_name_list: list = None,
                 edge_size=224, transform=None,
                 split_target_key=None, split_name: str = None,
                 stopping_folder_name_list: list = ['thumbnails'], tile_suffix: str = '.jpeg',
                 csv_file_name="task_description.csv",
                 task_excluding_list=('slide_id', 'tile_image_path', 'tile_y', 'tile_x', 'tile_name', 'split')):
        """
        Custom Dataset to load pseudo-bulk image and their gene label data for all slides.

        Notice: Each tile in the CSV is assumed to be existing in its WSI folder,
                because the filtered CSV is build based the available tiles

        Parameters:
        - root_path (str): Path to the folder containing tiles and bulk label for all tiles.
        - task_description_csv (str): label csv path, default None: taking the csv_file_name in root_path
        - task_name_list (list): the assigned task of loading, default None will be loading all tasks in CSV file

        - edge_size (int): Target size for resizing images (default is 224).
        - transform (callable, optional): Image transforms to apply to each ROI.

        - split_target_key : str
            The key that specifies the column name for taking the split_name,
            'split_nfold-k', n is the total fold number and k is the fold index
        - split_name : str
            The key word of patient_ids/labeled_slide_names as the split lists to build dataset
        - stopping_folder_name_list:?

        - tile_suffix (str): File extension of tile images (default is '.jpeg').
        - csv_file_name (str):
        - excluding_list (tuple): List of columns to exclude. Default is ('WSI_name', ...).
                            the attribute starts with 'split' will be ignored as they are designed for control split

        The filled label csv for the valid tiles:
            slide_id, tile_image_path, tile_y, tile_x, tile_name, gene names...

        Get item:
        sample = {
            "image_features": patch_image_tensor,
            "slide_id": slide_id,
            'coord_yx': patch_coord_yx,
            "task_name_list": self.task_name_list,
            "task_description_list": task_description_list,
        }
        """
        self.root_path = root_path
        self.tile_suffix = tile_suffix

        # Extend the list with unique folder names
        stopping_folder_name_list.extend(["task-settings", "task-settings-5folds", task_setting_folder_name])
        stopping_folder_name_list = list(set(stopping_folder_name_list))
        # adjust excluding_list
        task_excluding_list = list(task_excluding_list)
        if split_target_key is not None:
            task_excluding_list.append(split_target_key)
        task_excluding_list = list(set(task_excluding_list))

        self.slide_paths = self.find_slide_paths_and_ids(stopping_folder_name_list=stopping_folder_name_list)
        self.slide_ids = list(self.slide_paths.keys())

        # build teh task configs
        self.task_cfg = load_yaml_config(os.path.join(root_path, task_setting_folder_name, "task_configs.yaml"))
        self.task_dict = self.task_cfg.get("all_task_dict")
        self.one_hot_table = self.task_cfg.get("one_hot_table")

        # build the all label csv
        if task_description_csv:
            # load the specified aggregated file
            task_description_data_df = pd.read_csv(task_description_csv)
        elif os.path.exists(os.path.join(self.root_path, task_setting_folder_name, csv_file_name)):
            # load the aggregated file at root_path
            task_description_data_df = pd.read_csv(os.path.join(self.root_path,
                                                                task_setting_folder_name, csv_file_name))
        else:
            # Aggregate gene expression data across all slides into a single DataFrame
            task_description_data_df = pd.concat(
                [pd.read_csv(os.path.join(self.slide_paths[slide_id], csv_file_name)).assign(slide_id=slide_id)
                 for slide_id in self.slide_ids], ignore_index=True)

        # Set up the task
        self.task_name_list = task_name_list or self.task_cfg.get("tasks_to_run")
        assert self.task_name_list is not None

        # then Filter the csv to only take the specified tasks
        # (reduce memory cost for large scale of tasks such as gene)
        taking_attributes = []
        for attribute in task_description_data_df.columns:
            # find indexing or specified attributes in the csv
            if attribute in self.task_name_list or attribute in task_excluding_list:
                taking_attributes.append(attribute)
        task_description_data_df = task_description_data_df[taking_attributes]

        # Filter the csv to only take the available slides and tiles
        valid_task_description_data_df = task_description_data_df[
            task_description_data_df['slide_id'].isin(self.slide_ids)]
        # notice: the tile in the CSV is assumed to be existing in its WSI folder,
        # because the filtered CSV is build based the all tiles

        # Filter the csv to only take the specified split_name at given csv[split_target_key] (such as 'train').
        if split_target_key is not None:
            self.tile_task_description_df = valid_task_description_data_df[
                valid_task_description_data_df[split_target_key] == split_name]
        else:
            self.tile_task_description_df = valid_task_description_data_df

        # Default transform (resize, to tensor, normalize)
        default_transform = transforms.Compose([
            transforms.Resize((edge_size, edge_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.transform = transform or default_transform

    def find_slide_paths_and_ids(self, stopping_folder_name_list=['thumbnails']):
        """
        Finds slide folder paths, stopping search in specified folders.

        Parameters:
        - stopping_folder_name_list (list): List of folder names to ignore in search.
        """
        slide_paths = {}
        for dirpath, dirnames, _ in os.walk(self.root_path):
            dirnames[:] = [d for d in dirnames if d not in stopping_folder_name_list]
            for dirname in dirnames:
                slide_folder_path = os.path.join(dirpath, dirname)
                if any(fname.endswith(self.tile_suffix) for fname in os.listdir(slide_folder_path)):
                    slide_paths[dirname] = Path(slide_folder_path)
                    break
        return slide_paths

    def __len__(self):
        return len(self.tile_task_description_df)

    def get_MTL_label(self, raw_description):
        """
        raw_description: its a row of values, 'numpy.ndarray' object
        the ordering is following the self.task_name_list
        """
        task_description_list = []
        # Convert to a dictionary with task names as keys
        loaded_task_description = dict(zip(self.task_name_list, raw_description))

        # check if all task is labeled
        trigger_label_missing = True

        for task in self.task_name_list:
            data_type = self.task_dict[task]
            if not pd.isna(loaded_task_description[task]):  # In case of available label
                if data_type == 'float':  # Regression task
                    # Convert the string to a float if it's not already a float
                    value = float(loaded_task_description[task]) if \
                        isinstance(loaded_task_description[task], str) \
                        else loaded_task_description[task]
                    task_description_list.append(torch.tensor(value))
                    trigger_label_missing = False
                    # e.g., torch.tensor(0.69)
                else:  # Classification task
                    label = loaded_task_description[task]  # e.g., label = 'lusc'
                    one_hot_label = torch.tensor(self.one_hot_table[task][label])
                    long_label = one_hot_label.argmax()
                    # e.g., the index of the one-hot label, e.g., torch.LongTensor(1)
                    task_description_list.append(long_label)  # e.g., torch.tensor(1)
                    trigger_label_missing = False
            else:  # In case of missing label
                if data_type == "float":  # Regression task
                    task_description_list.append(torch.tensor(99999999.99))  # Missing label
                else:  # Classification task
                    task_description_list.append(torch.tensor(99999999))  # Missing label

        if trigger_label_missing:
            return None
        else:
            return task_description_list

    def __getitem__(self, idx):
        # Extract information for the current tile
        row = self.tile_task_description_df.iloc[idx]
        slide_id = row['slide_id']
        tile_name = row['tile_name']
        # self.task_name_list set the ORDERING of the retuning values also
        raw_description = row[self.task_name_list].values.astype(float)

        # obtain the MTL label
        task_description_list = self.get_MTL_label(raw_description)
        if task_description_list is None:
            print('In the current running tasks, all intended Labels are missing for Sample:', slide_id)
            return -1

        # Construct image path and coordinates
        img_path = os.path.join(self.slide_paths[slide_id], tile_name)
        y, x = read_tile_name_for_loc_y_x(tile_name, suffix=self.tile_suffix)
        patch_coord_yx_tensor = torch.tensor([y, x], dtype=torch.int32)

        # Load and transform the image
        with open(img_path, "rb") as f:
            patch_image = Image.open(f).convert("RGB")
            patch_image_tensor = self.transform(patch_image)
        # todo: Future: design a error output:   return -1 as sample

        # Prepare sample with image,coordinates, and tasks and slide id
        sample = {
            "image_features": patch_image_tensor,
            'coord_yx': patch_coord_yx_tensor,
            "task_name_list": self.task_name_list,
            "task_description_list": task_description_list,
            "slide_id": slide_id
        }
        return sample


# WSI MTL dataset builder
class SlideDataset(Dataset):
    def __init__(
            self,
            root_path: str,
            task_description_csv: str = None,
            task_setting_folder_name: str = "task_settings",
            slide_id_key="slide_id",
            split_target_key="split",
            split_name: str = "train",
            possible_suffixes=(".h5", ".pt", ".jpeg", ".jpg"),
            stopping_folder_name_list=["thumbnails", ],
            task_type="MTL",
            task_name_list=None,  # this is important to track the ordering
            task_excluding_list=('slide_id', 'tile_image_path', 'split'),
            data_name_mode=None,
            max_tiles=None,
            shuffle_tiles=None,
            padding=False,
            **kwargs,
    ):
        """
        Slide dataset class for retrieving slide_feature samples for different tasks.

        Each WSI is a folder (slide_folder, named by slide_id), and all cropped tiles are embedded as one .h5 file:
            h5file['features'] is a list of numpy features, each feature (can be of multiple dims: dim1, dim2, ...)
                            for transformer embedding, the feature dim is [768]
            h5file['coords_yx'] is a list of coordinates, each item is a [Y, X], Y, X is slide_feature index in WSI

        Arguments:
        ----------
        root_path : str
            The root path of the tile embeddings
        task_description_csv : label csv
        task_setting_folder_name: the folder name of config folder

        slide_id_key : str
            The key name of the slide_folder

        split_target_key : str
            The key that specifies the column name for taking the split_name,
            'split_nfold-k', n is the total fold number and k is the fold index
        split_name : str
            The key word of patient_ids/labeled_slide_names as the split lists to build dataset

        possible_suffixes: supported suffix for taking the samples
        stopping_folder_name_list:

        task_type: 'MTL' for building slide_feature level mtl dataset, 'embedding' for doing slide level embedding

        task_name_list: the assigned task of loading, default None will be loading tasks in config file

        data_name_mode: slide name rule, default is None, 'TCGA' for TCGA names,
                        which obtains the patient name of the slide_folder, and map the label in csv by patient name?
                        by default will try to load the config file

        max_tiles: tile taking maximum number for each slide, default is None,
                        by default will try to load the config file

        padding: if True, will pad all images to max_tiles or 10000, and image_features_lens will equal to
                    previous tile size. You may then revert the image feature using image_features_lens

        everytime it get a sample WSI:
        ----------
        sample = {'image_features': image features [N, D] tensor,
              'image_features_lens': data_dict['image_features_lens'],
              'pad_mask': data_dict['pad_mask'],
              'coords_yx': [N, 2] tensor,
              'slide_id': slide_id,
              'task_name_list': task_name_list,
              'task_description_list': task_description_list}
        """
        # Extend the list with unique folder names
        stopping_folder_name_list.extend(["task-settings", "task-settings-5folds", task_setting_folder_name])
        stopping_folder_name_list = list(set(stopping_folder_name_list))
        # adjust excluding_list
        task_excluding_list = list(task_excluding_list)
        task_excluding_list.append(slide_id_key)
        if split_target_key is not None:
            task_excluding_list.append(split_target_key)
        task_excluding_list = list(set(task_excluding_list))

        super(SlideDataset, self).__init__(**kwargs)
        self.task_type = task_type

        self.root_path = root_path
        self.possible_suffixes = possible_suffixes

        self.padding = padding

        if self.task_type == "embedding":
            # here the slide dataset is called without requirement of task labels,
            # in this case we dont need config and csv label file
            split_name = "all_embedding"
            self.task_name_list = "slide_embedding"
            try:
                self.task_cfg = load_yaml_config(os.path.join(root_path, task_setting_folder_name, "task_configs.yaml"))
                self.data_name_mode = data_name_mode or self.task_cfg.get("mode")
            except:
                self.task_cfg = None
                self.data_name_mode = data_name_mode

            # Find valid slide_feature paths that have tile encodings
            Data_valid_slide_ids, valid_sample_ids = self.get_data_valid_slides(None, stopping_folder_name_list,
                                                                                self.data_name_mode)
            # for embedding and pretraining scenario, we take all valid slide by data availability
            self.slide_ids = Data_valid_slide_ids
            self.setup_task_data(None, None, task_type=self.task_type)

        else:
            # the slide dataset need task labels
            # load task config
            self.task_cfg = load_yaml_config(os.path.join(root_path, task_setting_folder_name, "task_configs.yaml"))

            if slide_id_key == 'patient_id':
                self.data_name_mode = data_name_mode or self.task_cfg.get("mode")
            else:
                self.data_name_mode = None

            self.split_target_key = split_target_key  # the key to record the fold infor
            self.slide_id_key = slide_id_key

            # load label csv
            task_description_csv = task_description_csv or \
                                   os.path.join(root_path, task_setting_folder_name, "task_description.csv")
            task_description_data_df = pd.read_csv(task_description_csv)
            # Get the label from CSV file with WSIs assigned with the target split (such as 'train').
            task_description_data_df = task_description_data_df[
                task_description_data_df[self.split_target_key] == split_name]
            # we use split_target_key to indicate fold

            # Set up the task
            self.task_name_list = task_name_list or self.task_cfg.get("tasks_to_run")
            assert self.task_name_list is not None

            # then Filter the csv to only take the specified tasks
            # (reduce memory cost for large scale of tasks such as gene)
            taking_attributes = []
            for attribute in task_description_data_df.columns:
                # find indexing or specified attributes in the csv
                if attribute in self.task_name_list or attribute in task_excluding_list:
                    taking_attributes.append(attribute)
            task_description_data_df = task_description_data_df[taking_attributes]

            # Find valid slide_id paths that have tile encodings
            print(f'In dataset framework, we pair {self.slide_id_key} as the slide_id_key in label csv')
            Data_valid_slide_ids, valid_sample_ids = (
                self.get_data_valid_slides(task_description_data_df[self.slide_id_key].values,
                                           stopping_folder_name_list, self.data_name_mode))

            # filter the task_description_data_df
            if self.data_name_mode == 'TCGA':
                valid_task_description_data_df = \
                    task_description_data_df[task_description_data_df[self.slide_id_key].isin(valid_sample_ids)]
                # Find valid slide_id_names with labels and set the labels for them
                Label_valid_sample_names = self.setup_task_data(valid_task_description_data_df, self.task_name_list,
                                                                task_type=self.task_type)
                # mapping to obtain the slide ids
                self.slide_ids = self.name_mapping(Data_valid_slide_ids, Label_valid_sample_names, self.data_name_mode)
            else:
                valid_task_description_data_df = \
                    task_description_data_df[task_description_data_df[self.slide_id_key].isin(Data_valid_slide_ids)]
                # filter label valid samples on the data valid samples
                self.slide_ids = self.setup_task_data(valid_task_description_data_df, self.task_name_list,
                                                      task_type=self.task_type)

            # Load from settings or set default value
        self.max_tiles = max_tiles \
                         or (self.task_cfg.get("max_tiles", 10000) if self.task_cfg is not None else 10000)
        self.shuffle_tiles = shuffle_tiles \
                             or (self.task_cfg.get("shuffle_tiles", False) if self.task_cfg is not None else False)
        print("Dataset has been initialized with " + str(len(self.slide_ids)) + " slides for split:", str(split_name))

        """
        # fixme notice this check is very slow when the hard disk is in use
        # check tile distribution 
        self.check_tile_num_distribution(draw_path=os.path.join(root_path, task_setting_folder_name,
                                                                str(split_name) + '.jpeg'))
        """

    def name_mapping(self, Data_valid_slide_ids, Label_valid_sample_names, data_name_mode):
        valid_slide_id_list = []
        for slide_id in Data_valid_slide_ids:
            slide_id_name = slide_id[0:12] if data_name_mode == 'TCGA' else slide_id
            if slide_id_name in Label_valid_sample_names:
                valid_slide_id_list.append(slide_id)
        return valid_slide_id_list

    def find_slide_paths_and_ids(self, stopping_folder_name_list):
        """
        Find slide_feature paths and their corresponding IDs.

        This operation can be slow as there are many '.jpg' files in the slide_folder.
        Therefore, when it detects one slide_folder, all files inside should not be tested again.


        stopping_folder_name_list: a list of the not-taking slide names, default is none
        """
        slide_paths = {}
        for dirpath, dirnames, _ in os.walk(self.root_path):
            # Remove directories in the stopping list from dirnames to avoid descending into them
            dirnames[:] = [d for d in dirnames if d not in stopping_folder_name_list]

            for dirname in dirnames:
                slide_folder_path = os.path.join(dirpath, dirname)
                # Check for the presence of .h5, .pt, or .jpg files and break early
                for fname in os.listdir(slide_folder_path):
                    if fname.endswith(self.possible_suffixes):
                        slide_id = dirname
                        slide_paths[slide_id] = Path(slide_folder_path)
                        break  # Break early once a valid file is found
        return slide_paths

    def get_data_valid_slides(self, labeled_slide_names, stopping_folder_name_list, data_name_mode="TCGA"):
        """Get the slides that have tile encodings stored in the tile directory.

        labeled_slide_names: a list of the slide names, default is none for all slides
        stopping_folder_name_list: a list of the not-taking slide names, default is none

        """
        slide_paths = self.find_slide_paths_and_ids(stopping_folder_name_list=stopping_folder_name_list)

        self.slide_paths = {}

        valid_slide_ids = []
        valid_sample_ids = []

        for slide_id in slide_paths:
            slide_id_name = slide_id[0:12] if data_name_mode == 'TCGA' else slide_id
            slide_path = slide_paths[slide_id]
            if labeled_slide_names is not None:
                # for embedding tasks or other tasks that need to ensure we have label for the slides
                if slide_id_name not in labeled_slide_names:
                    # when this sample is not required in this current split
                    continue
                else:
                    if "pt_files" in self.root_path.split("/")[-1]:
                        embedded_slide_file = slide_id.replace(".svs", "") + ".pt"
                    else:
                        embedded_slide_file = slide_id.replace(".svs", "") + ".h5"
                    embedded_slide_path = os.path.join(slide_path, embedded_slide_file)
                    if not os.path.exists(embedded_slide_path):
                        print("Data Missing for: ", slide_id)
                    else:
                        # Add to valid list
                        valid_slide_ids.append(slide_id)
                        valid_sample_ids.append(slide_id_name)
                        self.slide_paths[slide_id] = embedded_slide_path
            else:
                # for embedding tasks or other tasks that do not need to ensure we have label for the slides
                if "pt_files" in self.root_path.split("/")[-1]:
                    embedded_slide_file = slide_id.replace(".svs", "") + ".pt"
                else:
                    embedded_slide_file = slide_id.replace(".svs", "") + ".h5"
                embedded_slide_path = os.path.join(slide_path, embedded_slide_file)
                if not os.path.exists(embedded_slide_path):
                    print("Data Missing for: ", slide_id)
                else:
                    # Add to valid list
                    valid_slide_ids.append(slide_id)
                    valid_sample_ids.append(slide_id_name)
                    self.slide_paths[slide_id] = embedded_slide_path

        return valid_slide_ids, valid_sample_ids

    def check_tile_num_distribution(self, draw_path):
        # fixme notice this is very slow when the hard disk is in use
        import matplotlib.pyplot as plt

        tile_num_list = []
        for slide_id in self.slide_ids:
            # Get the slide_feature path
            embedded_slide_path = self.slide_paths[slide_id]

            # Read assets from the slide_feature
            assets, _ = self.read_assets_from_h5(embedded_slide_path)
            tile_num = len(assets["coords_yx"])
            tile_num_list.append((slide_id, tile_num))

        # Sort the list based on tile numbers
        tile_num_list.sort(key=lambda x: x[1])

        # Extract slide_feature IDs and corresponding tile counts
        slide_ids_sorted, tile_counts_sorted = zip(*tile_num_list)

        # Plotting the distribution of tile numbers
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(slide_ids_sorted)), tile_counts_sorted)
        plt.xticks(range(len(slide_ids_sorted)), [""] * len(slide_ids_sorted))  # Disable printing the slide_feature IDs
        plt.xlabel("Slide ID")
        plt.ylabel("Number of Tiles")
        plt.title("Distribution of Tile Numbers Across Slides")

        # Adding a horizontal orange line indicating `self.max_tiles`
        plt.axhline(y=self.max_tiles, color="orange", linestyle="--", linewidth=2,
                    label=f"Taking Max Tiles ({self.max_tiles})")

        # Add the value of `self.max_tiles` to the y-tick labels
        y_ticks = plt.yticks()[0]  # Get the current y-tick values
        plt.yticks(list(y_ticks) + [self.max_tiles])  # Add `self.max_tiles` to the y-ticks

        # Add a legend to the plot
        plt.legend()
        # Save the plot to the specified path
        plt.tight_layout()
        plt.savefig(draw_path)
        # Display the plot
        plt.show()

    def prepare_MTL_data_list(self, task_description_csv, task_name_list):
        """Prepare the sample for multi-label task.

        return
        Valid_sample_names
        task_dict,
        one_hot_table,
        labels: a dict recording the labels for each slide_feature by
                loading the corresponding self.slide_id_key in csv
        """
        task_dict = self.task_cfg.get("all_task_dict")
        one_hot_table = self.task_cfg.get("one_hot_table")

        Sample_names = task_description_csv[self.slide_id_key]
        Label_valid_sample_names = []

        labels = {}
        for Sample_name in Sample_names:
            task_description_list = []
            loaded_task_description = (
                task_description_csv[task_description_csv[self.slide_id_key] == Sample_name].to_dict('records'))[0]
            # check if all task is labeled
            trigger_label_missing = True
            for task in task_name_list:
                data_type = task_dict[task]
                if not pd.isna(loaded_task_description[task]):  # In case of available label
                    if data_type == 'float':  # Regression task
                        # Convert the string to a float if it's not already a float
                        value = float(loaded_task_description[task]) if \
                            isinstance(loaded_task_description[task], str) \
                            else loaded_task_description[task]
                        task_description_list.append(torch.tensor(value))
                        trigger_label_missing = False
                        # e.g., torch.tensor(0.69)
                    else:  # Classification task
                        label = str(loaded_task_description[task])  # e.g., label = 'lusc'
                        one_hot_label = torch.tensor(one_hot_table[task][label])
                        long_label = one_hot_label.argmax()
                        # e.g., the index of the one-hot label, e.g., torch.LongTensor(1)
                        task_description_list.append(long_label)  # e.g., torch.tensor(1)
                        trigger_label_missing = False
                else:  # In case of missing label
                    if data_type == "float":  # Regression task
                        task_description_list.append(torch.tensor(99999999.99))  # Missing label
                    else:  # Classification task
                        task_description_list.append(torch.tensor(99999999))  # Missing label

            if trigger_label_missing:
                pass
                print('In the current running tasks, all intended Labels are missing for Sample:', Sample_name)
            else:
                labels[Sample_name] = task_description_list
                Label_valid_sample_names.append(Sample_name)

        return Label_valid_sample_names, task_dict, one_hot_table, labels

    def setup_task_data(self, task_description_csv=None, task_name_list=None, task_type="MTL"):
        """Prepare the sample for single task tasks_to_run or multi-task tasks_to_run.

        old demo self.prepare_single_task_data_list, the split is a list of wsi name
        """

        # todo multiple modality func
        if task_type == "MTL":
            assert task_description_csv is not None and task_name_list is not None
            Label_valid_sample_names, self.task_dict, self.one_hot_table, self.labels = \
                self.prepare_MTL_data_list(task_description_csv, task_name_list)
            return Label_valid_sample_names
        elif task_type == 'embedding':
            self.task_dict, self.one_hot_table, self.labels = None, None, None
        else:
            raise NotImplementedError  # currently we only have task_type == 'MTL'

    def shuffle_data_pairs(self, images: torch.Tensor, coords: torch.Tensor) -> tuple:
        """Shuffle the serialized images and coordinates"""
        indices = torch.randperm(len(images))
        images_ = images[indices]
        coords_ = coords[indices]
        return images_, coords_

    def read_assets_from_h5(self, h5_path: str) -> tuple:
        """Read the assets from the h5 file"""
        assets = {}
        attrs = {}
        with h5py.File(h5_path, "r", swmr=True) as f:
            for key in f.keys():
                assets[key] = f[key][:]
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs

    def pad_tensor(self, tgt_tensor):
        # record tensor length
        tgt_tensor_lens, tgt_tensor_dim = tgt_tensor.shape

        # convert all tensor into certain size. This will enable batch size > 1 for single GPU.
        pad_size = self.max_tiles - tgt_tensor_lens

        # create a zero tensor for padding
        padding = torch.zeros((pad_size, tgt_tensor_dim), dtype=tgt_tensor.dtype, device=tgt_tensor.device)
        tgt_tensor = torch.cat((tgt_tensor, padding), dim=0)

        return tgt_tensor, tgt_tensor_lens

    def get_embedded_data_dict_read(self, embedding_file_path: str) -> dict:
        if ".pt" in embedding_file_path:
            image_features = torch.load(embedding_file_path)
            coords_yx = 0
        elif ".h5" in embedding_file_path:
            assets, _ = self.read_assets_from_h5(embedding_file_path)
            image_features = torch.from_numpy(assets["features"])
            coords_yx = torch.from_numpy(assets["coords_yx"])

            # if shuffle the sample
            if self.shuffle_tiles:
                image_features, coords_yx = self.shuffle_data_pairs(image_features, coords_yx)
            # randomly sample the tils but maintaining the ordering
            if image_features.size(0) > self.max_tiles:
                # Generate random indices while keeping their relative order
                random_indices = torch.randperm(image_features.size(0))[:self.max_tiles]
                random_indices = random_indices.sort().values  # Ensure relative order is maintained
                # Select the tiles based on the sorted random indices
                image_features = image_features[random_indices, :]
                coords_yx = coords_yx[random_indices, :]
        else:
            raise NotImplementedError  # not an intended feature format for embedded tiles
        return image_features, coords_yx

    def get_embedded_data_dict_padding(self, image_features, coords_yx):
        if self.padding:
            # pad image and coords_yx to self.max_tiles with 0
            image_features, image_features_lens = self.pad_tensor(image_features)
            coords_yx, _ = self.pad_tensor(coords_yx)
        else:
            image_features_lens = image_features.size(0)
        return image_features, image_features_lens, coords_yx

    def get_embedded_data_dict(self, embedding_file_path: str) -> dict:
        """Get the image_features from the path"""
        image_features, coords_yx = self.get_embedded_data_dict_read(embedding_file_path)

        image_features, image_features_lens, coords_yx = self.get_embedded_data_dict_padding(image_features, coords_yx)

        # set the input dict
        data_dict = {
            "image_features": image_features,
            "image_features_lens": image_features_lens,
            "pad_mask": 0,  # It may be used for some model design
            "coords_yx": coords_yx}
        return data_dict

    def get_one_embedded_sample(self, idx: int) -> dict:
        """Get one sample from the dataset"""
        # get the slide_id that has label and feature
        slide_id = self.slide_ids[idx]
        # get the slide_feature path (notice the slide_paths is all WSIs with features including no label ones)
        embedded_slide_path = self.slide_paths[slide_id]

        # get the slide_feature tile embeddings
        data_dict = self.get_embedded_data_dict(embedded_slide_path)
        # get the slide_feature label
        slide_id_name = slide_id[0:12] if self.data_name_mode == 'TCGA' else slide_id
        task_description_list = self.labels[slide_id_name] if self.task_type != 'embedding' else 'None'

        # set the sample dict
        sample = {
            "image_features": data_dict["image_features"],
            "image_features_lens": data_dict["image_features_lens"],
            "pad_mask": data_dict["pad_mask"],
            "coords_yx": data_dict["coords_yx"],
            "slide_id": slide_id,
            "task_name_list": self.task_name_list,
            "task_description_list": task_description_list,
        }

        return sample

    def get_embedded_sample_with_try(self, idx, n_try=3):
        """Get the sample with n_try, handles missing/failed sample, but not nicely"""

        for _ in range(n_try):
            try:
                one_embedded_sample = self.get_one_embedded_sample(idx)
                return one_embedded_sample
            except Exception as e:
                print("Error in getting the sample, try another index")
                idx = np.random.randint(0, len(self.slide_ids))
        print("Error in getting one sample with n_try: ", n_try)
        # raise RuntimeError('Failed to get a valid sample after {} tries'.format(n_try))
        return -1

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        """
        everytime it get a sample WSI:
        ----------
        sample = {'image_features': image features [N, D] tensor,
              'image_features_lens': data_dict['image_features_lens'],
              'pad_mask': data_dict['pad_mask'],
              'coords_yx': [N, 2] tensor,
              'slide_id': slide_id,
              'task_name_list': task_name_list,
              'task_description_list': task_description_list}
        """
        # fixme in this current framework the model always trained with wsi batch size of 1
        slide_level_sample = self.get_embedded_sample_with_try(idx)
        return slide_level_sample


def MTL_collate_fn(batch, non_stack_keys=["task_name_list"]):
    """
    Generalized collate function for datasets returning dictionaries with arbitrary keys.

    Parameters:
    - batch: List of samples from the dataset, where each sample is a dictionary.
    - non_stack_keys: List of keys that should not be stacked but instead taken as-is from the first sample.

    Returns:
    - collated_batch: Dictionary with collated data for each key in the samples.
    """
    # Filter out invalid data (-1 for not valid return from dataset)
    cleaned_batch = [data for data in batch if data != -1]

    # If the cleaned batch is empty, return None
    if len(cleaned_batch) == 0:
        return None

    # Initialize collated batch
    collated_batch = {}

    # Get all keys from the first sample
    sample_keys = cleaned_batch[0].keys()

    for key in sample_keys:
        # Extract all values for the current key across the batch
        values = [sample[key] for sample in cleaned_batch]

        if key in non_stack_keys:
            # Use the value from the first sample (assuming all are identical)
            collated_batch[key] = values[0]
        else:
            # Use default_collate to stack or process as needed
            try:
                collated_batch[key] = default_collate(values)
            except Exception as e:
                raise ValueError(f"Error collating key '{key}': {e}")

    return collated_batch


# demo for showcase ROI data loading
def try_ROI_dataset_framework(root_path, task_description_csv=None, tasks_to_run=None, split_target_key=None,
                              batch_size=1024, num_workers=20, pin_memory=False):
    """Try with specified training parameters

    build_split_and_task_configs(root_path, task_description_csv, dataset_name, tasks_to_run,
                                 slide_id_key, split_target_key, task_setting_folder_name, mode)
    """
    import time
    from tqdm import tqdm

    task_name_list = tasks_to_run.split("%") if tasks_to_run else None

    start_time = time.time()

    # check the dataset
    Train_dataset = Bulk_ROI_Dataset(root_path, task_description_csv, task_name_list=task_name_list,
                                     transform=None, split_target_key=split_target_key, split_name="Train")
    Val_dataset = Bulk_ROI_Dataset(root_path, task_description_csv, task_name_list=task_name_list,
                                   transform=None, split_target_key=split_target_key, split_name="Val")

    print('task_name_list:', Train_dataset.task_name_list)

    dataloaders = {
        "Train": torch.utils.data.DataLoader(Train_dataset, batch_size=batch_size,
                                             collate_fn=MTL_collate_fn,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             shuffle=True, drop_last=True),
        "Val": torch.utils.data.DataLoader(Val_dataset, batch_size=batch_size,
                                           collate_fn=MTL_collate_fn,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           shuffle=False, drop_last=True)}

    dataloader_time = time.time()
    print(f'dataloader initialized, time consumed: {dataloader_time - start_time:.2f} sec')

    # data loop for a epoch
    for data_iter_step, sample in enumerate(tqdm(dataloaders["Train"],
                                                 desc="Dataloader Iteration Progress: Train dataloader",
                                                 unit="batch", disable=False)):
        if sample is None:
            continue
        # the staking of dataloader will return a stacked sample
        '''
        "image_features": B,C,e,e
        'coord_yx': B,2
        "task_name_list": a list of task's name
        '''

        # take data and task_description_list from sample
        image_features, coords_yx, task_name_list, task_description_list, slide_id = sample
        '''
            sample = {
            "image_features": patch_image_tensor,
            'coord_yx': patch_coord_yx,
            "task_name_list": self.task_name_list,
            "task_description_list": task_description_list,
            "slide_id": slide_id
        }
        '''
        # image_features is a tensor of [B,3,edge,edge],  coords_yx is tensor of [B,N,2]
        # print(image_features.shape)
        # print(coords_yx.shape)
        # break

    print(f'\n\n')
    finish_time = time.time()
    print(f'dataset verified, time consumed: {finish_time - dataloader_time:.2f} sec')
    print(f'total time consumed: {finish_time - start_time:.2f} sec')


# demo for showcase WSI data loading
def try_WSI_dataset_framework(args):
    """Try with specified training parameters

    build_split_and_task_configs(root_path, task_description_csv, dataset_name, tasks_to_run,
                                 slide_id_key, split_target_key, task_setting_folder_name, mode)
    """
    import time
    from tqdm import tqdm

    task_name_list = args.tasks_to_run.split("%")
    padding = True if args.batch_size > 1 else False

    start_time = time.time()

    # check the dataset
    Train_dataset = SlideDataset(
        args.root_path,
        args.task_description_csv,
        task_setting_folder_name=args.task_setting_folder_name,
        split_name="Train",
        slide_id_key=args.slide_id_key,
        split_target_key=args.split_target_key,
        data_name_mode=args.data_name_mode,
        max_tiles=args.max_tiles,
        task_type=args.task_type,
        task_name_list=task_name_list,
        padding=padding
    )
    Val_dataset = SlideDataset(
        args.root_path,
        args.task_description_csv,
        task_setting_folder_name=args.task_setting_folder_name,
        split_name="Val",
        slide_id_key=args.slide_id_key,
        split_target_key=args.split_target_key,
        data_name_mode=args.data_name_mode,
        max_tiles=args.max_tiles,
        task_type=args.task_type,
        task_name_list=task_name_list,
        padding=padding
    )

    print(Train_dataset.get_embedded_sample_with_try(20))

    dataloaders = {
        "Train": torch.utils.data.DataLoader(Train_dataset, batch_size=args.batch_size,
                                             collate_fn=MTL_collate_fn,
                                             num_workers=args.num_workers,
                                             persistent_workers=args.persistent_workers,
                                             pin_memory=args.pin_memory,
                                             prefetch_factor=args.prefetch_factor,
                                             shuffle=True, drop_last=True),
        "Val": torch.utils.data.DataLoader(Val_dataset, batch_size=args.batch_size,
                                           collate_fn=MTL_collate_fn,
                                           num_workers=args.num_workers,
                                           persistent_workers=args.persistent_workers,
                                           pin_memory=args.pin_memory,
                                           prefetch_factor=args.prefetch_factor,
                                           shuffle=False, drop_last=True)}

    dataloader_time = time.time()
    print(f'dataloader initialized, time consumed: {dataloader_time - start_time:.2f} sec')

    # data loop for a epoch
    for data_iter_step, sample in enumerate(
            tqdm(dataloaders["Train"], desc="Dataloader Iteration Progress: Train dataloader",
                 unit="batch", disable=False)):
        if sample is None:
            continue
        # take data and task_description_list from sample
        image_features, coords_yx, task_description_list, slide_id = sample
        # image_features is a tensor of [B,N,D],  coords_yx is tensor of [B,N,2]
        # print(image_features.shape)
        # print(coords_yx.shape)
        # break

    print(f'\n\n')
    finish_time = time.time()
    print(f'dataset verified, time consumed: {finish_time - dataloader_time:.2f} sec')
    print(f'total time consumed: {finish_time - start_time:.2f} sec')


# fixme this is temp for debugging (by Shangqing)
def get_args_parser():
    parser = argparse.ArgumentParser(description="MTL Training")

    # Dataset
    parser.add_argument("--root_path", default=None, type=str, help="MTL dataset root")
    parser.add_argument("--task_description_csv", default=None, type=str, help="label csv file path")
    parser.add_argument("--slide_id_key", default="patient_id", type=str, help="key for mapping the label")
    parser.add_argument("--split_target_key", default="fold_information_5fold-1",
                        type=str, help="key identifying the split information")
    parser.add_argument("--task_setting_folder_name", default="task-settings-5folds",
                        type=str, help="task-settings folder name")
    parser.add_argument("--data_name_mode", default="TCGA",
                        type=str, help="data name mode")
    parser.add_argument("--tasks_to_run", default=None, type=str,
                        help="tasks to run MTL, split with %, default is None with all tasks in task config to be run")
    parser.add_argument("--task_type", default="MTL", type=str,
                        help="always MTL")

    # Dataloader
    parser.add_argument("--persistent_workers", action='store_true',
                        help="dataloader persistent_workers")
    parser.add_argument("--pin_memory", action='store_true',
                        help="dataloader pin_memory")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="dataloader num_workers")
    parser.add_argument("--prefetch_factor", default=None, type=int,
                        help="dataloader prefetch_factor")  # fixme: cause instability, not recommended
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size , default 1')
    parser.add_argument('--max_tiles', default=2000, type=int,
                        help='max tile for loading')

    return parser


# fixme this is temp for debugging (by Shangqing)
if __name__ == "__WSI__":
    import csv
    import itertools

    # Dataset config
    ## =========== TCGA-BRCA
    root_path = "/data/hdd_1/BigModel/TCGA-BRCA/tiles-embeddings/"
    task_description_csv = "/data/ssd_1/WSI/TCGA-BRCA/tiles-embeddings/task-settings-5folds/task_description_tcga-brca_20241206.csv"
    slide_id_key = "slide_id"
    split_target_key = "fold_information_5fold-1"
    task_setting_folder_name = "task-settings-5folds"
    data_name_mode = "TCGA"
    # tasks_to_run = "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%lung-cancer-subtyping%OS_MONTHS"
    tasks_to_run = 'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced'
    task_type = "MTL"  # MTL or embedding, if its embedding, the task config is not composable

    # Define params searching range
    num_workers_range = [16, 64]
    batch_size_range = [1, 4, 16]
    persistent_workers_range = [True]
    pin_memory_range = [True]

    # csv file name
    output_file = "parameter_tuning_results.csv"

    # Record best config
    best_time = float('inf')
    best_config = None

    # Initialize csv file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["num_workers", "batch_size", "persistent_workers", "pin_memory", "elapsed_time"])

        # Grid search
        for num_workers, batch_size, persistent_workers, pin_memory in itertools.product(
                num_workers_range, batch_size_range, persistent_workers_range, pin_memory_range
        ):
            # Skip by rules
            if num_workers == 0 and (persistent_workers_range or pin_memory_range):
                continue

            args = argparse.Namespace(
                # dataset config
                root_path=root_path,
                task_description_csv=task_description_csv,
                slide_id_key=slide_id_key,
                split_target_key=split_target_key,
                task_setting_folder_name=task_setting_folder_name,
                data_name_mode=data_name_mode,
                tasks_to_run=tasks_to_run,
                task_type=task_type,
                # dataloader config
                num_workers=num_workers,
                batch_size=batch_size,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory,
                max_tiles=2000,
                prefetch_factor=None,
            )

            start_time = time.time()
            try_WSI_dataset_framework(args)
            # time.sleep(np.random.uniform(0.05, 1)) # dummy
            elapsed_time = time.time() - start_time

            # Write to csv
            writer.writerow([num_workers, batch_size, persistent_workers, pin_memory, elapsed_time])

            # Update best config
            if elapsed_time < best_time:
                best_time = elapsed_time
                best_config = {
                    "num_workers": num_workers,
                    "batch_size": batch_size,
                    "persistent_workers": persistent_workers,
                    "pin_memory": pin_memory,
                }

            print(f"Tested config: {vars(args)}, Time: {elapsed_time:.4f}s")

    # Print best result and write to csv
    print("\nBest configuration:")
    print(best_config)
    print(f"Best time: {best_time:.4f}s")

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Best Configuration"])
        writer.writerow(["num_workers", "batch_size", "persistent_workers", "pin_memory", "elapsed_time"])
        writer.writerow([
            best_config["num_workers"],
            best_config["batch_size"],
            best_config["persistent_workers"],
            best_config["pin_memory"],
            best_time
        ])

    ## To run manually through scripts:
    # parser = get_args_parser()
    # args = parser.parse_args()
    # try_dataset_framework(args)

    ## Other dataset config:

    # ## =========== TCGA-demoset
    # root_path = "/data/ssd_1/WSI/TCGA-demoset/tiles-embeddings/"
    # task_description_csv = "/data/ssd_1/WSI/TCGA-demoset/tiles-embeddings/task-settings-5folds/task_description.csv"
    # slide_id_key = "slide_id"
    # split_target_key = "fold_information_5fold-1"
    # task_setting_folder_name = "task-settings-5folds"
    # data_name_mode = "TCGA"  # fixme this is making a name mapping for label
    # dataset_name = ("TCGA-demoset",)
    # # tasks_to_run = "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%lung-cancer-subtyping%OS_MONTHS"
    # tasks_to_run = 'TUMOR_STATUS'
    # task_type = "MTL"  # MTL or embedding, if its embedding, the task config is not composable

    # ## =========== TCGA-COAD-READ
    # root_path = "/data/ssd_1/WSI/TCGA-COAD-READ/tiles-embeddings"
    # task_description_csv = "/home/workenv/BigModel/Archive/dataset_csv/TCGA/tcga-coad-read_marker10.csv"
    # slide_id_key = "patient_id"
    # split_target_key = "fold_information_5fold-1"
    # task_setting_folder_name = "task-settings-5folds"
    # data_name_mode = "TCGA"  # fixme this is making a name mapping for label
    # dataset_name = ("TCGA-COAD-READ",)
    # tasks_to_run = "iCMS%CMS%EPCAM"
    # task_type = "MTL"  # MTL or embedding, if its embedding, the task config is not composable

    # ## =========== TCGA-lung
    # root_path = "/data/hdd_1/BigModel/TCGA-lung/tiles-embeddings/"
    # task_description_csv = "/data/ssd_1/WSI/TCGA-lung/tiles-embeddings/task-settings-5folds/task_description_tcga-lung_20241121.csv"
    # slide_id_key = "slide_id"
    # split_target_key = "fold_information_5fold-1"
    # task_setting_folder_name = "task-settings-5folds"
    # data_name_mode = "TCGA"
    # # tasks_to_run = "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%lung-cancer-subtyping%OS_MONTHS"
    # tasks_to_run = 'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced'
    # task_type = "MTL"  # MTL or embedding, if its embedding, the task config is not composable

    # ## =========== TCGA-BRCA
    # root_path = "/data/hdd_1/BigModel/TCGA-BRCA/tiles-embeddings/"
    # task_description_csv = "/data/ssd_1/WSI/TCGA-BRCA/tiles-embeddings/task-settings-5folds/task_description_tcga-lung_20241121.csv"
    # slide_id_key = "slide_id"
    # split_target_key = "fold_information_5fold-1"
    # task_setting_folder_name = "task-settings-5folds"
    # data_name_mode = "TCGA"
    # # tasks_to_run = "AJCC_PATHOLOGIC_TUMOR_STAGE_reduced%lung-cancer-subtyping%OS_MONTHS"
    # tasks_to_run = 'AJCC_PATHOLOGIC_TUMOR_STAGE_reduced'
    # task_type = "MTL"  # MTL or embedding, if its embedding, the task config is not composable

if __name__ == '__main__':
    
    parser = get_args_parser()
    args = parser.parse_args()
    try_WSI_dataset_framework(args)
    '''
    # ROI running fixme this is temp for debugging
    try_ROI_dataset_framework(root_path='/data/hdd_1/BigModel/SO/tiled_data',
                              task_description_csv='/data/hdd_1/BigModel/SO/tiled_data/filtered_tile_labels.csv',
                              tasks_to_run='ACKR1%ACTA2%ADAM12%ADM',
                              batch_size=256, split_target_key=None)
    '''
