"""
WSI tile cropping dataset tools   Script  ver： Jan 3rd 12:20

# type A is for (ROI+WSI approaches)
# type B is for (Cell+ROI+WSI approaches)



Terminology:
1. WSI / slide_feature: the whole scanned slide_feature
2. ROI: the region of interest. usually it means the region with tissue or a small part of slide_feature,
notice it has two meanings therefor it may be confusing.
3. Tile (as noun) / Patch / ROI: a small part taken from WSI,
usually a non-overlapping griding can generate a series of them for each WSI.
4. Tile (as verb) / Crop: the non-overlapping griding methods to break the WSI into many small patches.
5. ROI sequence / tile sequence / slide_feature sequence: all patches taken from one WSI.

In WSI object, the data is stored in multiple scale, it is called levels (from high to low)
it corresponds to low resolution/magnification (highest level) to high resolution/magnification (lowest level)
for example:
1. you can see all tissues on your screen with 1x magnification,
now its lowest magnification and highest level (level-8), now the resolution for whole slide_feature is super low
2. you can only one cell on your screen with 100x magnification,
now its highest magnification and lowest level (level-0), now the resolution for whole slide_feature is super high

todo
fixme timothy and Dhanav mention some tiff loading issue, tianyi will take a look
"""

import os
import sys
from pathlib import Path

# For convenience, import all path to sys
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir))
sys.path.append(str(this_file_dir.parent))
sys.path.append(str(this_file_dir.parent.parent))
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

import time
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import functools
import logging
import shutil
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import gc

import numpy as np
import pandas as pd
import PIL
from matplotlib import collections, patches, pyplot as plt
from monai.data import Dataset
from monai.data.wsi_reader import WSIReader
from openslide import OpenSlide
from tqdm import tqdm

from DataPipe.wsi_tools import *
from DataPipe.segmentation_and_filtering_tools import *


def get_base_name(file_name: str) -> str:
    """Remove all extensions from a filename."""
    while '.' in file_name:
        file_name = file_name.rsplit('.', 1)[0]  # Remove the last extension
    return file_name


# todo it should be designed for better pretraining management
def prepare_slides_sample_list(slide_root, slide_suffixes=['.svs', '.ndpi', '.tiff', '.tif'], metadata_file_paths=None,
                               image_key: str = "slide_image_path", mode='TCGA'):
    """
    Make a slides_sample_list, each element inside is a
    slide_sample = {"slide_image_path": slide_image_path, "slide_id": slide_id, "metadata": {}}

    Later it will be warped like: dataset = torch.utils.data.Dataset(slides_sample_list)

    Args:
        slide_root: a root folder of multiple WSIs, notice each WSI may be organized in different folder format
            this code will go through all files in the slide_root and find the WSI files with suffix
            in the slide_suffixes list

        slide_suffixes: list, the possible suffixes for WSI image file.

        metadata_file_paths: list, the paths to multiple metadata files, this code will go through the files and
            load them as pd dataframe, if the information for a certain WSI (slide_id) can be found
            in certain pd dataframe, the information will be kept in dictionary in "metadata"

        :param image_key: WSI Image key in the input and output dictionaries. default is 'slide_image_path'

    Returns: slides_sample_list

    slide_sample = {"slide_image_path": str(slide_image_path), "slide_id": slide_id, "metadata": {}}
    """
    slides_sample_list = []

    # Traverse the slide_root directory to find all files with the given suffixes
    for root, _, files in os.walk(slide_root):
        for file in files:
            if "morphology" in file:
                # fixme: this is a temp hack for skipping the morphology annoataion files in aidd samples
                continue
            if any(file.endswith(suffix) for suffix in slide_suffixes):
                slide_image_path = Path(root) / file
                slide_id = get_base_name(slide_image_path.stem)
                # If its TCGA samples, the slide_id is the filename [0:12] without the suffix
                patient_id = slide_id[0:12] if mode == 'TCGA' else slide_id

                # Initialize the slide_sample dictionary
                slide_sample = {image_key: str(slide_image_path), "slide_id": slide_id, "metadata": {}}

                # If metadata file paths are provided, load metadata into the slide_sample object.
                # later they will be recorded automatically in the tile log csv
                if metadata_file_paths:
                    assert mode == 'TCGA', "metadata file only support TCGA for now!"
                    for metadata_file_path in metadata_file_paths:
                        metadata_df = pd.read_csv(metadata_file_path)

                        # Check if the slide_id is in the metadata dataframe
                        if patient_id in metadata_df['patient_id'].values:  # fixme TCGA format only 'patient_id'
                            # Get the metadata row for this slide_id and convert it to a dictionary
                            # fixme TCGA format only
                            metadata = metadata_df.loc[metadata_df['patient_id']
                                                       == patient_id].to_dict(orient='records')[0]
                            slide_sample["metadata"].update(metadata)

                # Append the slide_sample to the slides_sample_list
                slides_sample_list.append(slide_sample)

    return slides_sample_list


def process_one_slide_to_tiles(sample: Dict["SlideKey", Any],
                               output_dir: Path, thumbnail_dir: Path,
                               margin: int = 0, tile_size: int = 224,
                               target_mpp: float = 0.5, manual_mpp: Optional[float] = None,
                               force_read_level: int = None, force_roi_scale: float = None,
                               foreground_threshold: Optional[float] = None, occupancy_threshold: float = 0.1,
                               pixel_std_threshold: int = 5, extreme_value_portion_th: float = 0.5,
                               chunk_scale_in_tiles: int = 0,
                               tile_progress: bool = False,
                               image_key: str = "slide_image_path",
                               ROI_image_key='tile_image_path') -> str:
    """Load and process a slide_feature, saving tile images and information to a CSV file.

    :param sample: Slide information dictionary, returned by the input slide_feature dataset.

    :param output_dir: Root directory for the output dataset; outputs for a single slide_feature will be
    saved inside `task_settings_path/slide_id/`.
    :param thumbnail_dir:

    :param margin: Margin around the foreground bounding box, in pixels at lowest resolution.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param target_mpp: 0.5 for prov-gigapath
    :param manual_mpp: should be None for automatic checking the WSI metadata

    :param force_read_level: None by default. If not None, read from this level by force, ignore target_mpp
    :param force_roi_scale: target mpp/force_read_level mpp, use if force_read_level not None

    :param foreground_threshold: Luminance threshold (0 to 255) to determine if one pixel is foreground
    then the pixels can be used for checking tile occupancy. If `None` (default),
    an optimal threshold will be estimated automatically.

    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.

    :param pixel_std_threshold: The threshold for the pixel variance at one ROI
                                to say this ROI image is too 'empty'
    :param extreme_value_portion_th: The threshold for the ratio of the pixels being 0 of one ROI,
                                    to say this ROI image is too 'empty'

    :param chunk_scale_in_tiles: to speed up the io for loading the WSI regions
    :param tile_progress: Whether to display a progress bar in the terminal.
    :param image_key: Image key in the input and output dictionaries. default is 'slide_image_path'
    :param ROI_image_key: ROI Image key in the input and output dictionaries. default is 'tile_image_path'
    """
    # STEP 0: set up path and log files
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    slide_id: str = sample["slide_id"]
    rel_slide_dir = Path(slide_id)
    output_tiles_dir = output_dir / rel_slide_dir

    logging.info(f">>> Slide dir {output_tiles_dir}")
    if is_already_processed(output_tiles_dir, rel_slide_dir, logging):
        logging.info(f">>> Skipping {output_tiles_dir} - already processed")
        return output_tiles_dir

    else:
        output_tiles_dir.mkdir(parents=True, exist_ok=True)

        # STEP 1: take the WSI and get the ROIs (valid tissue regions)
        slide_image_path = Path(sample[image_key])
        logging.info(f"Loading slide_feature {slide_id} ...\nFile: {slide_image_path}")

        # take the valid tissue regions (ROIs) of the WSI (with monai and OpenSlide loader)
        loader = Loader_for_get_one_WSI_sample(WSIReader(backend="OpenSlide"), image_key=image_key,
                                               target_mpp=target_mpp, manual_mpp=manual_mpp,
                                               force_read_level=force_read_level, force_roi_scale=force_roi_scale, 
                                               margin=margin, foreground_threshold=foreground_threshold,
                                               thumbnail_dir=thumbnail_dir)
        WSI_image_obj, loaded_ROI_samples = loader(sample)

        # STEP 2: prepare log files:
        # prepare a csv log file for ROI tiles
        keys_to_save = ("slide_id", ROI_image_key, "tile_id", "label",
                        "tile_y", "tile_x", "occupancy")
        # Decode the slide_feature metadata (if got)
        slide_metadata: Dict[str, Any] = loaded_ROI_samples[0]["metadata"]
        metadata_keys = tuple("slide_" + key for key in slide_metadata)
        # print('metadata_keys',metadata_keys)
        csv_columns: Tuple[str, ...] = (*keys_to_save, *metadata_keys)
        # print('csv_columns',csv_columns)

        # build a recording file to log the processed files
        dataset_csv_path = output_tiles_dir / "dataset.csv"
        dataset_csv_file = dataset_csv_path.open('w')
        dataset_csv_file.write(','.join(csv_columns) + '\n')  # write CSV header

        failed_tiles_csv_path = output_tiles_dir / "failed_tiles.csv"
        failed_tiles_file = failed_tiles_csv_path.open('w')
        failed_tiles_file.write('tile_id' + '\n')  # write CSV header

        # STEP 3: Tile (crop) the WSI into ROI tiles (patches), save into h5
        logging.info(f"Tiling slide_feature {slide_id} ...")
        # each ROI_sample in loaded_WSI_samples is a valid ROI region
        n_failed_tiles = 0
        for index, ROI_sample in enumerate(loaded_ROI_samples):
            # The estimated luminance (foreground threshold) for whole WSI is applied to ROI here to filter the tiles
            tile_info_list, n_failed_tile = extract_valid_tiles(slide_image_path, ROI_sample, output_tiles_dir,
                                                                tile_size=tile_size,
                                                                foreground_threshold=ROI_sample[
                                                                    "foreground_threshold"],
                                                                occupancy_threshold=occupancy_threshold,
                                                                pixel_std_threshold=pixel_std_threshold,
                                                                extreme_value_portion_th=extreme_value_portion_th,
                                                                chunk_scale_in_tiles=chunk_scale_in_tiles,
                                                                tile_progress=tile_progress,
                                                                ROI_image_key=ROI_image_key,
                                                                num_workers=1,  # for WSI parallel use one CPU for each
                                                                log_file_elements=(dataset_csv_file,
                                                                                   failed_tiles_file,
                                                                                   keys_to_save,
                                                                                   metadata_keys))
            if tile_info_list == None:
                # empty ROi (finished processing earlier)
                break
            # STEP 3: visualize the tile location overlay to WSI
            visualize_tile_locations(ROI_sample, thumbnail_dir / (slide_image_path.name
                                                                  + "_roi_" + str(index) + "_tiles.jpeg"),
                                     tile_info_list, image_key=image_key)
            n_failed_tiles += n_failed_tile

        if n_failed_tiles > 0:
            # what we want to do with slides that have some failed tiles? for now, just drop?
            logging.warning(f"{slide_id} is incomplete. {n_failed_tiles} tiles failed in reading.")

        # Explicitly delete the large objects
        del WSI_image_obj
        del loaded_ROI_samples
        del loader

        # close csv logging
        dataset_csv_file.close()
        failed_tiles_file.close()

        # Force garbage collection
        gc.collect()

        logging.info(f"Finished processing slide_feature {slide_id}")

        return None


# Function to handle exceptions and logging
def safe_process_one_slide_to_tiles(sample: Dict[str, Any], output_dir: Path, thumbnail_dir: Path,
                                    margin: int, tile_size: int, target_mpp: float, manual_mpp,
                                    force_read_level: int, force_roi_scale: float,
                                    foreground_threshold: Optional[float], occupancy_threshold: float,
                                    pixel_std_threshold: int, extreme_value_portion_th: float,
                                    chunk_scale_in_tiles: int,
                                    tile_progress: bool, image_key: str) -> Optional[str]:
    try:
        # Process the slide_feature and return the output directory or some result
        return process_one_slide_to_tiles(
            sample=sample, output_dir=output_dir, thumbnail_dir=thumbnail_dir,
            margin=margin, tile_size=tile_size, target_mpp=target_mpp, manual_mpp=manual_mpp,
            force_read_level=force_read_level, force_roi_scale=force_roi_scale,
            foreground_threshold=foreground_threshold, occupancy_threshold=occupancy_threshold,
            pixel_std_threshold=pixel_std_threshold, extreme_value_portion_th=extreme_value_portion_th,
            chunk_scale_in_tiles=chunk_scale_in_tiles,
            tile_progress=tile_progress, image_key=image_key
        )
    except Exception as e:
        logging.error(f"Error processing slide_feature {sample[image_key]}: {e}")
        print(f"Error processing slide_feature {sample[image_key]}: {e}")
        return sample[image_key]


# Process multiple WSIs for pretraining
def prepare_tiles_dataset_for_all_slides(slides_sample_list: "slides_sample_list", root_output_dir: Union[str, Path],
                                         margin: int = 0, tile_size: int = 224,
                                         target_mpp: float = 0.5, manual_mpp: float = None,
                                         force_read_level: int = None,
                                         force_roi_scale: float = None,
                                         foreground_threshold: Optional[float] = None,
                                         occupancy_threshold: float = 0.1,
                                         pixel_std_threshold: int = 5,
                                         extreme_value_portion_th: float = 0.5,
                                         chunk_scale_in_tiles: int = 4,
                                         image_key: str = "slide_image_path",
                                         parallel: bool = True,
                                         n_processes: Optional[int] = None,
                                         overwrite: bool = False,
                                         n_slides: Optional[int] = None) -> None:
    """Process a slides dataset to produce many folders of tiles dataset.

    :param slides_sample_list: Input slide folders list
    :param root_output_dir: The root directory of the output tiles dataset.

    :param margin: Margin around the foreground bounding box, in pixels at lowest resolution.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param target_mpp: 0.5 for prov-gigapath
    :param manual_mpp: should be None for automatic checking the WSI metadata

    :param force_read_level: None by default, if not None then read WSI from this level and ignore target_mpp
    :param force_roi_scale: target mpp/force_read_level mpp, use if force_read_level not None
    
    :param foreground_threshold: Luminance threshold (0 to 255) to determine if one pixel is foreground
    then the pixels can be used for checking tile occupancy. If `None` (default),
    an optimal threshold will be estimated automatically.

    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.

    :param pixel_std_threshold: The threshold for the pixel variance at one ROI
                                to say this ROI image is too 'empty'
    :param extreme_value_portion_th: The threshold for the ratio of the pixels being 0 of one ROI,
                                    to say this ROI image is too 'empty'

    :param chunk_scale_in_tiles: to speed up the io for loading the WSI regions

    :param image_key: Image key in the input and output dictionaries. default is 'slide_image_path'

    :param parallel: Whether slides should be processed in parallel with multiprocessing.
    :param n_processes: If given, limit the total number of slides for multiprocessing

    :param overwrite: Whether to overwrite an existing output tiles dataset. If `True`, will delete
    and recreate `root_output_dir`, otherwise will resume by skipping already processed slides.
    :param n_slides: If given, limit the total number of slides for debugging.
    """

    # Ignoring some types here because mypy is getting confused with the MONAI Dataset class
    # to select a sub-set of samples use keyword n_slides
    dataset = Dataset(slides_sample_list)[:n_slides]  # type: ignore

    # make sure all slide_feature files exist in the image dir
    for sample in dataset:
        image_path = Path(sample[image_key])
        assert image_path.exists(), f"{image_path} doesn't exist"
        logging.info(f"Detected WSI: {image_path}")

    output_dir = Path(root_output_dir)
    logging.info(f"Creating dataset of mpp-{target_mpp} {tile_size}x{tile_size} "
                 f"{slides_sample_list.__class__.__name__} tiles at: {output_dir}")

    if force_read_level is not None:
        logging.info(f"Warning: detected force_read_level={force_read_level}, read from level {force_read_level} by force!")

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=not overwrite)
    thumbnail_dir = output_dir / "thumbnails"
    thumbnail_dir.mkdir(exist_ok=True)
    logging.info(f"Thumbnail directory: {thumbnail_dir}")

    func = functools.partial(safe_process_one_slide_to_tiles,
                             output_dir=output_dir, thumbnail_dir=thumbnail_dir,
                             margin=margin, tile_size=tile_size,
                             target_mpp=target_mpp, manual_mpp=manual_mpp,
                             force_read_level=force_read_level,
                             force_roi_scale=force_roi_scale,
                             foreground_threshold=foreground_threshold,
                             occupancy_threshold=occupancy_threshold,
                             pixel_std_threshold=pixel_std_threshold,
                             extreme_value_portion_th=extreme_value_portion_th,
                             chunk_scale_in_tiles=chunk_scale_in_tiles,
                             tile_progress=not parallel, image_key=image_key)

    if parallel:
        import multiprocessing
        from multiprocessing import cpu_count

        pool = multiprocessing.Pool(processes=n_processes or cpu_count())
        map_func = pool.imap_unordered  # type: ignore
    else:
        map_func = map  # type: ignore

    error_WSIs = list(
        tqdm(map_func(func, dataset), desc="Slides", unit="wsi", total=len(dataset)))  # type: ignore

    if parallel:
        pool.close()
        pool.join()  # Ensure all processes are cleaned up

    logging.info("Merging slide_feature files into a single file")
    merge_dataset_csv_files(output_dir)

    logging.info(f"Error WSIs: {error_WSIs}")
    print(error_WSIs)  # fixme temp design, better write to system as a file


# for inference
def prepare_tiles_dataset_for_single_slide(slide_image_path: str = '', save_dir: str = '', tile_size: int = 224,
                                           image_key: str = "slide_image_path"):
    """
    This function is used to tile a single slide_feature and save the tiles to a directory.
    -------------------------------------------------------------------------------
    Warnings: pixman 0.38 has a known bug, which produces partial broken images.
    Make sure to use a different version of pixman.
    -------------------------------------------------------------------------------

    Arguments:
    ----------
    slide_file : str
        The path to the slide_feature file.
    save_dir : str
        The directory to save the tiles.
    tile_size : int
        The size of the tiles.
    """
    slide_id = os.path.basename(slide_image_path)
    # slide_sample = {"slide_image_path": slide_image_path, "slide_id": slide_id,
    # "metadata": {'TP53': 1, 'Diagnosis': 'Lung Cancer'}}
    slide_sample = {image_key: slide_image_path, "slide_id": slide_id, "metadata": {}}

    save_dir = Path(save_dir)
    if save_dir.exists():
        print(f"Warning: Directory {save_dir} already exists. ")

    print(f"Processing slide_feature {slide_image_path} with tile size {tile_size}. Saving to {save_dir}.")

    process_one_slide_to_tiles(
        slide_sample,
        output_dir=save_dir,
        margin=0,
        tile_size=tile_size,
        foreground_threshold=None,  # None to use automatic illuminance estimation
        occupancy_threshold=0.1,
        thumbnail_dir=save_dir / "thumbnails",
        chunk_scale_in_tiles=4,
        tile_progress=True,
        image_key=image_key
    )

    output_tiles_dir = save_dir / Path(slide_id)

    dataset_csv_path = os.path.join(output_tiles_dir, "dataset.csv")
    dataset_df = pd.read_csv(dataset_csv_path)
    assert len(dataset_df) > 0
    failed_csv_path = os.path.join(output_tiles_dir, "failed_tiles.csv")
    failed_df = pd.read_csv(failed_csv_path)
    assert len(failed_df) == 0

    print(f"Slide {slide_image_path} has been tiled. {len(dataset_df)} tiles saved to {output_tiles_dir}.")


# todo to support loading ROI level tasks for ROI model
def tile_ROI_loadding_dataset(slide_image_path):
    pass


def main(args):
    os.makedirs(args.tiled_WSI_dataset_path, exist_ok=True)

    # Configure logging
    main_log_file = Path(args.tiled_WSI_dataset_path) / 'wsi_tile_processing.log'
    logging.basicConfig(filename=main_log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # split metadata_file_paths with %
    metadata_file_paths = args.metadata_file_paths.split("%") if args.metadata_file_paths is not None else None
    
    slides_sample_list = prepare_slides_sample_list(slide_root=args.WSI_dataset_path, 
                                                    slide_suffixes=['.svs', '.ndpi', '.tiff', '.tif'],
                                                    metadata_file_paths=metadata_file_paths,
                                                    mode=args.mode)

    prepare_tiles_dataset_for_all_slides(
        slides_sample_list,
        root_output_dir=args.tiled_WSI_dataset_path,
        tile_size=args.edge_size,
        target_mpp=args.target_mpp,
        manual_mpp=args.manual_mpp,
        force_read_level=args.force_read_level,
        force_roi_scale=args.force_roi_scale,
        overwrite=args.overwrite,
        parallel=True)


def get_args_parser():
    """Input parameters
    """
    parser = argparse.ArgumentParser(description='Build split and task configs.')

    parser.add_argument('--WSI_dataset_path', type=str,
                        default='/data/hdd_1/BigModel/TCGA-LUAD-LUSC/TCGA-LUAD-raw',
                        help='Root path for the original HE WSI datasets')

    parser.add_argument('--tiled_WSI_dataset_path', type=str,
                        default='/data/hdd_1/BigModel/TCGA-LUAD-LUSC/tiles_datasets',
                        help='Root path for the Tiled WSI datasets')
    parser.add_argument('--edge_size', type=int, default=224,
                        help='edge size of tile')

    parser.add_argument('--target_mpp', type=float, default=0.5,
                        help='target_mpp')
    parser.add_argument('--manual_mpp', type=float, default=None,
                        help='manual_mpp should be None for automatic checking the WSI metadata')
    parser.add_argument('--force_read_level', type=int, default=None,
                        help='if not None, read from this level of WSI and ignore target_mpp')
    parser.add_argument('--force_roi_scale', type=float, default=None,
                        help='target mpp/force_read_level mpp, use if force_read_level not None')

    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite previous embedding at the path')
    # metadata parts (this is not required for processing, just additional log)
    parser.add_argument('--mode', type=str, default=None,
                        help='Mode (e.g., TCGA), Optional this locate the sample name in metadata')
    parser.add_argument('--metadata_file_paths', type=str, default=None,
                        help='metadata_file_paths, split with % for different paths')
    
    return parser


if __name__ == '__main__':
    # Suppress the specific RuntimeWarning related to overflow
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in reduce")

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
