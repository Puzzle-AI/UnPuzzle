"""
Tools to process WSI     Script  ver： Feb 2nd 09:00
a WSI is a scanned whole slide_feature image of some tissue region and many blank background

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


this code consists many tools to
0.load WSI object
1.select the valid region (with tissue instead of empty)
2.tile(crop) the wsi into ROI sequences

we use
from monai.data.wsi_reader import WSIReader(backend="OpenSlide")
it returns slide_feature shape in (CHW) not (CWH)!
However
slide_feature = openslide.OpenSlide(wsi_path) return slide_feature in (WHC)
"""
import os
import sys
from pathlib import Path

# For convenience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels

import gc
from copy import deepcopy
import openslide
import cv2
from typing import Any, Dict, Iterable, Optional, Sequence
import pandas as pd
import PIL
from matplotlib import collections, patches, pyplot as plt
from monai.data import Dataset
from monai.data.wsi_reader import WSIReader
from tqdm import tqdm
from PIL import ImageDraw, Image
import json
import traceback
import warnings
import multiprocessing

try:
    from .bbox_tools import get_ROI_bounding_box_list, Box
    from .segmentation_and_filtering_tools import *
except:
    from bbox_tools import get_ROI_bounding_box_list, Box
    from segmentation_and_filtering_tools import *


# this is used to process WSI to get slide_feature-level (for OpenSlide) at a target mpp
def get_nearest_level_for_target_mpp(WSI_image_obj, target_mpp, slide_image_path=None, manual_mpp=None):
    '''
    This code is designed to get the nearest level to load the WSI so the image is at target_mpp.
    WSI_image_obj is from:
    from monai.data.wsi_reader import WSIReader
    WSI_image_obj: OpenSlide = WSIReader(backend="OpenSlide").read(WSI_image_path)
    target_mpp: target mpp for resizing the window of cropped tile
    manual_mpp: should be None for auto reading mpp from WSI metadata,
                specify this only incase we don't have openslide.PROPERTY_NAME_MPP

    slide_image_path: optional

    return: target_loading_level, and the ROI_size_scale
    The ROI_size_scale is designed for adjusting the ROI_patch_size.
    For example, if the mpp at the nearest level (1.0) is 4 times the target_mpp (0.25),
    then the ROI_size_scale is 0.25 so the cropped ROI should be 1/4 the size of the original cropping size
    and then resize to the original size.
    '''
    # Try to get the MPP value from the WSI image properties
    mpp_x = manual_mpp or WSI_image_obj.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = manual_mpp or WSI_image_obj.properties.get(openslide.PROPERTY_NAME_MPP_Y)

    # Calculate highest resolutoin relative level (assume level 0 resolution = 0.25)
    if mpp_x is None or mpp_y is None:
        # fixme temp hack for reading qupath output samples
        qupath_json_path = os.path.join(str(slide_image_path), str(slide_image_path) + '_qpath_seg.json')
        if os.path.exists(qupath_json_path):
            # Open the file first
            with open(qupath_json_path, 'r') as f:
                mpp_x = json.load(f)["mpp_x"]
                lowest_mpp = float(mpp_x)
        else:
            raise KeyError("Microns per pixel (MPP) value is missing from the slide_feature properties, "
                           "pls use --manual_mpp to specify the mpp for slides")
    else:
        lowest_mpp = float(mpp_x)

    # Adjust lowest_mpp to common resolutions if within a certain range
    if 0.2 < lowest_mpp < 0.3:  # resolution:0.25um/pixel
        lowest_mpp = 0.25
    elif 0.4 < lowest_mpp < 0.6:  # resolution:0.5um/pixel
        lowest_mpp = 0.5

    # Create a dict for all available resolutions and their level
    mpp_level_dict = {}
    for slide_level in range(len(WSI_image_obj.level_downsamples)):
        ratio = WSI_image_obj.level_downsamples[slide_level]
        mpp = lowest_mpp * int(ratio)
        mpp_level_dict[mpp] = slide_level

    # Find the proper level and resolution
    if target_mpp in mpp_level_dict:  # read from target level directly
        target_level = mpp_level_dict[target_mpp]
        nearest_mpp = target_mpp
    elif target_mpp > lowest_mpp:  # resize from nearest lower level
        min_mpp_diff = float('inf')
        for mpp in mpp_level_dict:
            mpp_diff = target_mpp - mpp if target_mpp > mpp else float('inf')
            if mpp_diff < min_mpp_diff:
                target_level = mpp_level_dict[mpp]
                nearest_mpp = mpp
                min_mpp_diff = mpp_diff
    else:  # resize from nearest higher level
        min_mpp_diff = float('inf')
        for mpp in mpp_level_dict:
            mpp_diff = mpp - target_mpp if mpp > target_mpp else float('inf')
            if mpp_diff < min_mpp_diff:
                target_level = mpp_level_dict[mpp]
                nearest_mpp = mpp
                min_mpp_diff = mpp_diff

    # Calculate the ROI_size_scale
    ROI_size_scale = target_mpp / nearest_mpp

    return target_level, ROI_size_scale


def convert_bbox_to_target_level(WSI_image_obj, level0_bbox, target_level):
    scale = WSI_image_obj.level_downsamples[target_level]
    scaled_bbox = level0_bbox / scale

    return scale, scaled_bbox


# WSI reading tools
class Loader_for_get_one_WSI_sample(MapTransform):
    """
    This process is a pipeline warps openslide and other functions:
    Load a pathology whole slide_feature image (WSI), and find the valid regions (with foreground bounding box).

    This process operates on 'Slide information dictionary'!!!! -> (sample)

    Also adds the following meta-sample entries:
    - 'location' (tuple): top-right coordinates of the bounding box
    - 'size' (tuple): width and height of the bounding box
    - 'level' (int): chosen magnification level
    - 'scale' (float): corresponding scale, loaded from the file
    """

    def __init__(self, reader: WSIReader, image_key: str = "slide_image_path", target_mpp: float = 0.5,
                 manual_mpp: float = None, force_read_level: int = None, force_roi_scale: float = None,
                 margin: int = 0, foreground_threshold: Optional[float] = None,
                 thumbnail_dir: Path = '.') -> None:
        """
        :param reader: An instance of MONAI's `WSIReader`.
        :param image_key: Image key in the input and output dictionaries. default is 'slide_image_path'

        :param target_mpp: use to get Magnification level to load slides at target_mpp from the raw multi-scale files.
        :param manual_mpp: should be None for automatic checking the WSI metadata

        :param force_read_level: if not None, read from this level and ignore target_mpp
        :param force_roi_scale: target mpp/force_read_level mpp, use if force_read_level not None

        :param margin: Amount in level0-pixels by which to enlarge the estimated bounding box for cropping.

        :param foreground_threshold: Pixels with luminance below this value will be considered as foreground.
        If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.
        and it will be passed to the process of each specific WSI/ROI

        :param thumbnail_dir:

        """
        super().__init__([image_key], allow_missing_keys=False)
        self.reader = reader  # WSIReader(backend="OpenSlide"), from monai.data.wsi_reader import WSIReader
        self.image_key = image_key

        self.target_mpp = target_mpp
        self.manual_mpp = manual_mpp  # should be None for automatic checking the WSI metadata

        self.force_read_level = force_read_level
        self.force_roi_scale = force_roi_scale

        self.margin = margin
        self.foreground_threshold = foreground_threshold  # default None for automatic estimation

        self.thumbnail_dir = thumbnail_dir

    # WSI segmentation (high level)
    # + calculate luminance estimation for the threshold (identifying pixel as foreground)
    def WSI_region_detection(self, slide_obj: OpenSlide) -> Box:
        # Estimate bounding box at the lowest resolution/magnification (i.e. highest level)
        highest_level = slide_obj.level_count - 1
        # Load full slide_feature array at the given magnification level. in [C,H,W]
        slide, _ = self.reader.get_data(slide_obj, level=highest_level)

        if slide_obj.level_count == 1:
            # in this case, the sample is not nicely organized into multiple level,
            # the whole slide_feature is stored only once at a very high magnification (eg 100x).
            logging.warning(f"Warning: Only one level found in WSI . "
                            f"segment_foregound at high magnification for whole_slide_feature "
                            f"will use a lot of memory.")
            # if the WSI is tiff file, we can only have the level-0 and will need the json file for mpp information

        # if self.foreground_threshold is None, a threshold will be estimated with skimage Otsu's method.
        # foreground_mask is (1, H, W) boolean array indicating whether the pixel is foreground or not
        foreground_mask, Luminance_threshold = segment_foreground(slide, self.foreground_threshold)
        # threshold: Pixels with luminance below this value will be considered as foreground.
        # visualize_foreground_mask(slide_feature, foreground_mask, Luminance_threshold)

        scale = slide_obj.level_downsamples[highest_level]

        # select multiple valid region instead of one, reduce the calculation cost
        # fixme this maximum_top_n should be notice
        box_list = get_ROI_bounding_box_list(foreground_mask, maximum_top_n=20)
        level0_bbox_list = []
        for bbox in box_list:
            # get bbox at level-0
            level0_bbox = scale * bbox.add_margin(self.margin)
            level0_bbox_list.append(level0_bbox)

        return level0_bbox_list, Luminance_threshold

    def __call__(self, sample: Dict) -> Dict:
        """
        This process open WSI and compose image data and other information
        into the 'Slide information dictionary'

        :param sample: 'Slide information dictionary', dict describing image metadata.

        Example:
        {'slide_id': ['1ca999adbbc948e69783686e5b5414e4'],
        'slide_image_path': ['/tmp/datasets/PANDA/train_images/1ca999adbbc948e69783686e5b5414e4.tiff'],
         'mask': ['/tmp/datasets/PANDA/train_label_masks/1ca999adbbc948e69783686e5b5414e4_mask.tiff'],
         'data_provider': ['karolinska'],
         'isup_grade': tensor([0]),
         'gleason_score': ['0+0']}

        Returns:
            loaded_ROI_samples containing many :
                ROI_sample (each sample is a ROI region of the WSI)

        ROI_sample:
        {'slide_id': ['1ca999adbbc948e69783686e5b5414e4'],
        'slide_image_path': ['/tmp/datasets/PANDA/train_images/1ca999adbbc948e69783686e5b5414e4.tiff'],
         'mask': ['/tmp/datasets/PANDA/train_label_masks/1ca999adbbc948e69783686e5b5414e4_mask.tiff'],
         'data_provider': ['karolinska'],
         'isup_grade': tensor([0]),
         'gleason_score': ['0+0'],

         'ROI_index': ROI_index

         "origin_start_y": the absolute location on level_0 of the valid tissue region top-left point
         "origin_start_x": the absolute location on level_0 of the valid tissue region top-left point
         'origin_h': valid region height on level_0
         'origin_w': valid region width on level_0

         "target_start_y": the absolute location on target_level of the valid tissue region top-left point
         "target_start_x": the absolute location on target_level of the valid tissue region top-left point
         'target_h': scaled region height on target_level
         'target_w': scaled region width on target_level

         'target_level': converted magnification level for loading WSI near target mpp
         "ROI_scale": the scale factor of mpp conversion to make ROI patches
         "foreground_threshold": luminance estimation for the threshold (identifying pixel as foreground)
         }

        """
        # STEP 0: Get WSI Path
        self.slide_image_path = Path(sample[self.image_key])

        # STEP 1: Open WSI and get the loading level for target mpp
        logging.info(f"Loader_for_get_one_WSI_sample: read {self.slide_image_path}")
        WSI_image_obj: OpenSlide = self.reader.read(self.slide_image_path)

        # get target loading level
        if self.force_read_level is not None:
            target_level, ROI_size_scale = self.force_read_level, self.force_roi_scale
        else:
            target_level, ROI_size_scale = get_nearest_level_for_target_mpp(
                WSI_image_obj, self.target_mpp, self.slide_image_path, manual_mpp=self.manual_mpp)

        # STEP 2: Select the valid regions on the WSI, in one bbox
        logging.info("Loader_for_get_one_WSI_sample: get level0_bbox ")

        # Select multiple valid region instead of one big ROI, reduce the calculation cost
        # get WSI bbox's location and the foreground_threshold for Luminance estimation
        level0_bbox_list, Luminance_threshold = self.WSI_region_detection(WSI_image_obj)

        loaded_ROI_samples = []

        for ROI_index, level0_bbox in enumerate(level0_bbox_list):
            ROI_sample = deepcopy(sample)
            logging.info(f"Loader_for_get_one_WSI_sample: level0_bbox: {level0_bbox}")

            # Save original slide_feature thumbnail with bbox
            save_openslide_thumbnail(WSI_image_obj,
                                     self.thumbnail_dir / (self.slide_image_path.name
                                                           + "_original_ROI_" + str(ROI_index) + ".jpeg"),
                                     size_target=1024, level0_bbox=level0_bbox)

            # STEP 3: Calibrate the location for OpenSlide
            # OpenSlide takes absolute location coordinates in the level 0 reference frame,
            # but relative region size in pixels at the chosen level
            WSI_loc_scale, target_level_bbox = convert_bbox_to_target_level(WSI_image_obj, level0_bbox, target_level)

            # compose the WSI information for following steps
            # valid region at level-0
            ROI_sample["ROI_index"] = ROI_index

            ROI_sample["origin_start_y"] = level0_bbox.y
            ROI_sample["origin_start_x"] = level0_bbox.x
            # ROI_sample["origin_h"] = level0_bbox.h
            # ROI_sample["origin_w"] = level0_bbox.w

            # valid region at target_level
            # ROI_sample["target_start_y"] = target_level_bbox.y
            # ROI_sample["target_start_x"] = target_level_bbox.x
            ROI_sample["target_h"] = target_level_bbox.h
            ROI_sample["target_w"] = target_level_bbox.w
            ROI_sample["target_level"] = target_level

            ROI_sample["WSI_loc_scale"] = WSI_loc_scale  # scale of converting the target_level location to level-0
            ROI_sample["ROI_size_scale"] = ROI_size_scale  # scale of changing the slide_feature size
            ROI_sample["foreground_threshold"] = Luminance_threshold

            logging.info(f"Loader_for_get_one_WSI_sample: target_level: {target_level}, "
                         f"target_level_bbox: {target_level_bbox}")

            loaded_ROI_samples.append(ROI_sample)

        return WSI_image_obj, loaded_ROI_samples


# WSI tilling tools
def generate_rel_locations(target_level_tilesize, target_h, target_w):
    '''
    Generate the (y, x) starting coordinates of tiles to cover the target region size of h and w.
    If the [h, w] is not divisible by the target_level_tilesize, the last tile will be dropped.

    Parameters:
    - target_level_tilesize: Size of the tile at the target level (int)
    - target_h: Height of the target region (int)
    - target_w: Width of the target region (int)

    Returns:
    - List of tuples with (y, x) starting coordinates for each tile
    '''
    tile_coords = []

    # Calculate number of full tiles that fit in height and width,
    # drop the last tile instead of padding
    num_tiles_y = target_h // target_level_tilesize
    num_tiles_x = target_w // target_level_tilesize

    # Generate the coordinates
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            y = i * target_level_tilesize
            x = j * target_level_tilesize
            tile_coords.append((y, x))

    return tile_coords


def adjust_tile_locations(rel_tile_locations, loaded_WSI_sample, level0_y=None, level0_x=None):
    """
    Adjust the relative tile locations by scaling and translating based on the WSI sample information.

    Parameters:
    - rel_tile_locations: list of tuples, each containing (y, x) coordinates
    - loaded_WSI_sample: dictionary containing WSI sample information including scale and origin start coordinates

    Returns:
    - tile_locations: numpy array of adjusted tile locations (N, 2) with HW (YX) ordering
    """
    scale = loaded_WSI_sample["WSI_loc_scale"]
    origin_y = level0_y or loaded_WSI_sample["origin_start_y"]
    origin_x = level0_x or loaded_WSI_sample["origin_start_x"]

    # Adjust and translate tile locations
    scaled_locations = [(int(y * scale) + origin_y, int(x * scale) + origin_x) for y, x in rel_tile_locations]

    # Convert to numpy array and ensure integer type
    tile_locations = np.array(scaled_locations, dtype=int)

    return tile_locations


def generate_chunk_and_tile_level0_locations(target_level_tilesize, target_h, target_w,
                                             chunk_scale_in_tiles: int = 20, ROI_sample_from_WSI=None):
    """
    we load the ROI image by chunk of tiles, therefor improve disk reading performance with RAM usage

    chunk_scale_in_tiles : how many tiles to make up the chunk at h and w
    """
    assert ROI_sample_from_WSI is not None

    if chunk_scale_in_tiles == 0 or chunk_scale_in_tiles == 1:
        rel_tile_locations_remains = generate_rel_locations(target_level_tilesize, target_h, target_w)
        tile_locations = adjust_tile_locations(rel_tile_locations_remains, ROI_sample_from_WSI)
        return [], tile_locations

    else:
        # separate the chunk region and remaining tiles
        target_level_chunk_size = target_level_tilesize * chunk_scale_in_tiles

        # step 1: calculate the all tiles location in target ROI
        # List of tuples with (y, x) starting coordinates for each tile, encoded location starting with 0,0
        rel_tile_locations_all = generate_rel_locations(target_level_tilesize, target_h, target_w)

        # step 2: calculate the all tiles location in chuck region
        # Calculate number of full tiles that fit in the height and width of chuck,
        # drop the last chunk and adjust the region to chunk region
        chunk_region_h = target_h - target_h % target_level_chunk_size
        chunk_region_w = target_w - target_w % target_level_chunk_size

        # step 3: get the dropped tiles
        rel_tile_locations_chunk_region = generate_rel_locations(target_level_tilesize, chunk_region_h, chunk_region_w)
        rel_tile_locations_remains = get_remaining_tile_locations(
            rel_tile_locations_all, rel_tile_locations_chunk_region)
        # adjust to real location in level-0
        tile_locations = adjust_tile_locations(rel_tile_locations_remains, ROI_sample_from_WSI)

        # step 4: get and adjust the chunck location in level-0
        rel_chunk_locations = generate_rel_locations(target_level_chunk_size, chunk_region_h, chunk_region_w)
        chunk_locations = adjust_tile_locations(rel_chunk_locations, ROI_sample_from_WSI)

        return chunk_locations, tile_locations


def read_a_region_from_WSI(WSI_image_obj, level0_y, level0_x, target_level_tilesize, target_level,
                           reader=WSIReader(backend="OpenSlide")):
    """
    Read a tile from a WSI image with monai api

    Parameters:
    - WSI_image_obj: The WSI image open_slide object.
    - level0_y: The Y-coordinate of the top-left corner in the level 0 reference frame.
    - level0_x: The X-coordinate of the top-left corner in the level 0 reference frame.
    - target_level_tilesize: The size of the tile at the target level.
    - target_level: The level of the WSI image to read the tile from.
    - reader: The WSIReader object. If None, a default reader with OpenSlide backend will be created.

    Returns:
    - img_data: The image data of the tile as a numpy array in [C, H, W] format.
    """
    # OpenSlide takes absolute location coordinates in the level 0 reference frame,
    # but relative region size in pixels at the chosen target_level

    # notice in Monai reader.get_data: order of location/size arguments is YX (HW)
    get_data_kwargs = dict(location=(level0_y, level0_x),
                           size=(target_level_tilesize, target_level_tilesize),
                           level=target_level)

    # STEP 4: take the valid region from WSI
    img_data, _ = reader.get_data(WSI_image_obj, **get_data_kwargs)  # type: ignore
    # shape of img_data: int 8 numpy in [C, H, W] format RGB

    return img_data


def convert_a_chuck_to_many_tiles(a_chuck_img, level0_y, level0_x, chunk_scale_in_tiles,
                                  target_level_tilesize, ROI_sample_from_WSI):
    """

    a_chuck_img: The image data of the chuck as a numpy array in [C, H, W] int8 RGB format.
    """

    target_size = target_level_tilesize * chunk_scale_in_tiles
    rel_tile_locations = generate_rel_locations(target_level_tilesize, target_size, target_size)
    tile_locations = adjust_tile_locations(rel_tile_locations, ROI_sample_from_WSI, level0_y, level0_x)

    # Split the a_chuck_img into tiles
    tile_images = []

    for y, x in rel_tile_locations:
        tile = a_chuck_img[:, y:y + target_level_tilesize, x:x + target_level_tilesize]
        tile_images.append(tile)

    return tile_images, tile_locations


def resize_tile_to_PIL_target_tile(tile_img, tile_size):
    """
    Resize the selected tile image to a new tile size.

    Parameters:
    - tile_img: numpy array [C, H, W] int8 RGB image, where H=W=target_level_tilesize
    - tile_size: int, the new size to resize the tile to (H=W=tile_size)

    Returns:
    - resized_tile_img_pil: PIL Image (HWC RGB, int8), resized to (tile_size, tile_size)
    """
    # Convert the numpy array to PIL Image, first convert from [C, H, W] to [H, W, C]
    tile_img_pil = Image.fromarray(np.transpose(tile_img, (1, 2, 0)))

    # Ensure the image is in RGB format (useful if the original image might have an alpha channel)
    tile_img_pil = tile_img_pil.convert('RGB')

    # Resize the image using LANCZOS resampling
    resized_tile_img_pil = tile_img_pil.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

    return resized_tile_img_pil


def process_chunk(chuck_index, chuck_location, slide_image_path, target_level_tilesize, chunk_scale_in_tiles,
                  ROI_sample_from_WSI,
                  foreground_threshold, occupancy_threshold, pixel_std_threshold, extreme_value_portion_th, tile_size,
                  output_tiles_dir, ROI_image_key):
    WSI_image_obj: OpenSlide = WSIReader(backend="OpenSlide").read(slide_image_path)
    level0_y, level0_x = chuck_location
    try:
        a_chuck_img = read_a_region_from_WSI(WSI_image_obj, level0_y, level0_x,
                                             target_level_tilesize * chunk_scale_in_tiles,
                                             target_level=ROI_sample_from_WSI["target_level"],
                                             reader=WSIReader(backend="OpenSlide"))

        tile_images, chuck_tile_locations = convert_a_chuck_to_many_tiles(a_chuck_img, level0_y, level0_x,
                                                                          chunk_scale_in_tiles,
                                                                          target_level_tilesize,
                                                                          ROI_sample_from_WSI)

        processed_tiles = []
        failed_tiles = []

        # STEP 3: Filtering the ROIs with foreground occupancy ratio and pixel empty-value and pixel variance
        for tile_idx_in_chuck in range(chunk_scale_in_tiles * chunk_scale_in_tiles):
            a_tile_img = tile_images[tile_idx_in_chuck]
            tile_location = chuck_tile_locations[tile_idx_in_chuck]

            try:
                empty_tile_bool_mark, tile_occupancy = check_an_empty_tile(a_tile_img,
                                                                           foreground_threshold,
                                                                           occupancy_threshold,
                                                                           pixel_std_threshold,
                                                                           extreme_value_portion_th)
                # STEP 4: prepare tile
                if empty_tile_bool_mark:
                    processed_tiles.append((None, tile_location, True))
                else:
                    # Convert the cropped valid tile into PIL image with tile_size
                    PIL_tile = resize_tile_to_PIL_target_tile(a_tile_img, tile_size)

                    # Logging and save the image to h5
                    tile_info = get_tile_info_dict(ROI_sample_from_WSI, tile_occupancy, tile_location,
                                                   target_level_tilesize,
                                                   rel_slide_dir=Path(ROI_sample_from_WSI['slide_id']),
                                                   ROI_image_key=ROI_image_key)
                    save_PIL_image(PIL_tile, output_tiles_dir / tile_info["tile_image_path"])

                    processed_tiles.append((tile_info, None, False))

            except Exception as e:
                failed_tiles.append(tile_location)
                traceback.print_exc()
                warnings.warn(
                    f"An error occurred while saving tile {get_tile_id(ROI_sample_from_WSI['slide_id'], tile_location)}: {e}")

        return processed_tiles, failed_tiles, None  # No exception
    except Exception as e:
        # If an exception occurs, return the chunk location to indicate failure
        traceback.print_exc()
        warnings.warn(f"An error occurred while processing chunk at {chuck_location}: {e}")
        return None, None, chuck_location


def process_chunk_wrapper(args):
    """Wrapper function to handle exceptions and pass results back through a queue."""
    chuck_index, chuck_location, slide_image_path, target_level_tilesize, chunk_scale_in_tiles, \
        ROI_sample_from_WSI, foreground_threshold, occupancy_threshold, pixel_std_threshold, \
        extreme_value_portion_th, tile_size, output_tiles_dir, ROI_image_key = args

    result_queue = multiprocessing.Queue()

    try:
        result = process_chunk(chuck_index, chuck_location, slide_image_path, target_level_tilesize,
                               chunk_scale_in_tiles, ROI_sample_from_WSI, foreground_threshold,
                               occupancy_threshold, pixel_std_threshold, extreme_value_portion_th,
                               tile_size, output_tiles_dir, ROI_image_key)
        result_queue.put(result)
    except Exception as e:
        traceback.print_exc()
        warnings.warn(f"An error occurred while processing chunk {chuck_location}: {e}")
        result_queue.put((None, None, chuck_location))  # Indicate failure

    return result_queue.get()


def extract_chuck(num_workers, chuck_locations, slide_image_path, target_level_tilesize, chunk_scale_in_tiles,
                  ROI_sample_from_WSI, foreground_threshold, occupancy_threshold, pixel_std_threshold,
                  extreme_value_portion_th, tile_size, output_tiles_dir, ROI_image_key, tile_progress,
                  dataset_csv_file, failed_tiles_file, keys_to_save, metadata_keys):
    tile_info_list = []
    n_failed_tiles = 0
    n_discarded = 0

    with dataset_csv_file, failed_tiles_file:
        # Initialize Pool
        with multiprocessing.Pool(num_workers) as pool:
            # Prepare arguments for each chunk
            args_list = [(chuck_index, chuck_location, slide_image_path, target_level_tilesize, chunk_scale_in_tiles,
                          ROI_sample_from_WSI, foreground_threshold, occupancy_threshold, pixel_std_threshold,
                          extreme_value_portion_th, tile_size, output_tiles_dir, ROI_image_key)
                         for chuck_index, chuck_location in enumerate(chuck_locations)]

            # Map arguments to the pool
            results = list(tqdm(pool.imap_unordered(process_chunk_wrapper, args_list), total=len(args_list),
                                desc=f"Processing chunks for ROI index {ROI_sample_from_WSI['ROI_index']}",
                                unit="chunk", disable=not tile_progress))

            # Collect results
            for result in results:
                processed_tiles, failed_tiles, failed_location = result

                if failed_location:
                    n_failed_tiles += chunk_scale_in_tiles * chunk_scale_in_tiles
                    descriptor = get_tile_descriptor(failed_location)
                    failed_tiles_file.write(descriptor + '\n')
                    continue

                for tile_info, tile_location, empty_tile in processed_tiles:
                    if empty_tile:
                        n_discarded += 1
                    else:
                        if tile_info:
                            tile_info_list.append(tile_info)
                            dataset_row = format_csv_row(tile_info, keys_to_save, metadata_keys)
                            dataset_csv_file.write(dataset_row + '\n')

                for failed_tile in failed_tiles:
                    n_failed_tiles += 1
                    descriptor = get_tile_descriptor(failed_tile)
                    failed_tiles_file.write(descriptor + '\n')

    logging.info(
        f"Percentage tiles discarded: {n_discarded / (len(chuck_locations) * chunk_scale_in_tiles * chunk_scale_in_tiles) * 100:.2f}")

    return tile_info_list, n_failed_tiles, n_discarded


def extract_valid_tiles(slide_image_path, ROI_sample_from_WSI, output_tiles_dir: Path, tile_size: int,
                        foreground_threshold: float, occupancy_threshold: float = 0.1,
                        pixel_std_threshold: int = 5, extreme_value_portion_th: float = 0.5,
                        chunk_scale_in_tiles=0, tile_progress=True, ROI_image_key: str = "tile_image_path",
                        num_workers=1,  # todo make this to be parallel in sample
                        log_file_elements=None):
    """
    :param slide_image_path:
    :param ROI_sample_from_WSI:
    :param output_tiles_dir: tiles will be saved here as a h5

    :param tile_size:

    :param foreground_threshold: The threshold for identifying the foreground
    :param occupancy_threshold: The threshold of foreground occupancy to say this ROI image is too 'empty'

    :param pixel_std_threshold: The threshold for the pixel variance at one ROI
                                to say this ROI image is too 'empty'
    :param extreme_value_portion_th: The threshold for the ratio of the pixels being 0 of one ROI,
                                    to say this ROI image is too 'empty'

    :param chunk_scale_in_tiles: to speed up the io for loading the WSI regions

    :param tile_progress: Whether to display a progress bar in the terminal.
    :param ROI_image_key: ROI Image key in the input and output dictionaries. default is 'tile_image_path'

    :param num_workers: int, optional
        Number of subprocesses to use for data loading (default is 1 for not multiple processing).

    :param log_file_elements:
            dataset_csv_file,
            failed_tiles_file,
            keys_to_save,
            metadata_keys

    """
    assert log_file_elements is not None
    dataset_csv_file, failed_tiles_file, keys_to_save, metadata_keys = log_file_elements

    # STEP 0: prepare chuck and tile locations
    target_level_tilesize = int(tile_size * ROI_sample_from_WSI["ROI_size_scale"])

    # generate a list of level-0 tile location starting with "target_h" and "target_w" at target level
    chuck_locations, tile_locations = generate_chunk_and_tile_level0_locations(target_level_tilesize,
                                                                               ROI_sample_from_WSI["target_h"],
                                                                               ROI_sample_from_WSI["target_w"],
                                                                               chunk_scale_in_tiles,
                                                                               ROI_sample_from_WSI)
    # each chuck is chunk_scale_in_tiles^2 of tiles
    n_tiles = len(tile_locations) + len(chuck_locations) * chunk_scale_in_tiles * chunk_scale_in_tiles
    logging.info(f"{n_tiles} tiles found for this ROI")

    if n_tiles == 0:
        # the ROI is empty now
        return None, None
    else:
        pass
    # make a list to record the Tile information dictionary for valid tiles
    logging.info(f"Saving tiles for slide_feature {ROI_sample_from_WSI['slide_id']}  "
                 f"ROI index {ROI_sample_from_WSI['ROI_index']}...")

    '''  # todo multiple process STEP 2 (type a) : load tile at the location inside the chunk region
    tile_info_list, n_failed_tiles, n_discarded = \
        extract_chuck(num_workers, chuck_locations, slide_image_path, target_level_tilesize, chunk_scale_in_tiles,
                      ROI_sample_from_WSI, foreground_threshold, occupancy_threshold, pixel_std_threshold,
                      extreme_value_portion_th, tile_size, output_tiles_dir, ROI_image_key, tile_progress,
                      dataset_csv_file, failed_tiles_file, keys_to_save, metadata_keys)
    '''
    # STEP 2 (type a) : load tile at the location inside the chunk region
    # make a list to record the Tile information dictionary for valid tiles
    tile_info_list = []
    n_failed_tiles = 0
    n_discarded = 0
    # should put WSI_image_obj in the loop for multiple processing
    WSI_image_obj: OpenSlide = WSIReader(backend="OpenSlide").read(slide_image_path)

    for chuck_index in tqdm(range(len(chuck_locations)),  # todo make this to be parallel in sample
                            f"Processing the chuck tiles for ({ROI_sample_from_WSI['slide_id'][:6]}…) "
                            f"ROI index {ROI_sample_from_WSI['ROI_index']}",
                            unit="chuck", disable=not tile_progress):
        chuck_location = chuck_locations[chuck_index]
        level0_y, level0_x = chuck_location
        try:
            # should put WSI_image_obj in the loop for multiple processing
            # WSI_image_obj: OpenSlide = WSIReader(backend="OpenSlide").read(slide_image_path)
            a_chuck_img = read_a_region_from_WSI(WSI_image_obj, level0_y, level0_x,
                                                 target_level_tilesize * chunk_scale_in_tiles,
                                                 target_level=ROI_sample_from_WSI["target_level"],
                                                 reader=WSIReader(backend="OpenSlide"))

            tile_images, chuck_tile_locations = convert_a_chuck_to_many_tiles(a_chuck_img, level0_y, level0_x,
                                                                              chunk_scale_in_tiles,
                                                                              target_level_tilesize,
                                                                              ROI_sample_from_WSI)

        except:
            # sometimes reading certain region is broken (for pixel failure) and we need to skip
            n_failed_tiles += chunk_scale_in_tiles * chunk_scale_in_tiles
            descriptor = get_tile_descriptor(chuck_location)
            # we write down these failed tiles into the log
            failed_tiles_file.write(descriptor + '\n')
            traceback.print_exc()
            warnings.warn(f"An error occurred while saving tile "
                          f"{get_tile_id(ROI_sample_from_WSI['slide_id'], chuck_location)}: {e}")
        else:
            # STEP 3: Filtering the ROIs with foreground occupancy ratio and pixel empty-value and pixel variance
            for tile_idx_in_chuck in range(chunk_scale_in_tiles * chunk_scale_in_tiles):

                a_tile_img = tile_images[tile_idx_in_chuck]
                tile_location = chuck_tile_locations[tile_idx_in_chuck]

                try:
                    empty_tile_bool_mark, tile_occupancy = check_an_empty_tile(a_tile_img,
                                                                               foreground_threshold,
                                                                               occupancy_threshold,
                                                                               pixel_std_threshold,
                                                                               extreme_value_portion_th)
                    # STEP 4: prepare tile
                    if empty_tile_bool_mark:
                        n_discarded += 1
                    else:
                        # convert the cropped valid tile into PIL image with tile_size
                        PIL_tile = resize_tile_to_PIL_target_tile(a_tile_img, tile_size)
                        # todo in future we can use h5 to log tile infor

                        # logging and save the image to h5
                        tile_info = get_tile_info_dict(ROI_sample_from_WSI, tile_occupancy, tile_location,
                                                       target_level_tilesize,
                                                       rel_slide_dir=Path(ROI_sample_from_WSI['slide_id']),
                                                       ROI_image_key=ROI_image_key)
                        save_PIL_image(PIL_tile, output_tiles_dir / tile_info["tile_image_path"])

                except Exception as e:
                    # sometimes certain tiles is broken (for pixel failure) and we need to skip
                    n_failed_tiles += 1
                    descriptor = get_tile_descriptor(tile_location)
                    # we write down these failed tiles into the log
                    failed_tiles_file.write(descriptor + '\n')
                    traceback.print_exc()
                    warnings.warn(f"An error occurred while saving tile "
                                  f"{get_tile_id(ROI_sample_from_WSI['slide_id'], tile_location)}: {e}")
                else:
                    if not empty_tile_bool_mark:
                        # record the tile information into tile_info_list
                        tile_info_list.append(tile_info)
                        dataset_row = format_csv_row(tile_info, keys_to_save, metadata_keys)
                        dataset_csv_file.write(dataset_row + '\n')

    # STEP 2 (type b) : load tile at the location for the tiles outside the chunk region
    for tile_index in tqdm(range(len(tile_locations)),
                           f"Processing the out-of-chuck tiles for ({ROI_sample_from_WSI['slide_id'][:6]}…) "
                           f"ROI index {ROI_sample_from_WSI['ROI_index']}",
                           unit="tile", disable=not tile_progress):
        # should put WSI_image_obj in the loop for multiple processing
        # WSI_image_obj: OpenSlide = WSIReader(backend="OpenSlide").read(slide_image_path)
        tile_location = tile_locations[tile_index]
        level0_y, level0_x = tile_location

        try:
            a_tile_img = read_a_region_from_WSI(WSI_image_obj, level0_y, level0_x,
                                                target_level_tilesize, target_level=ROI_sample_from_WSI["target_level"],
                                                reader=WSIReader(backend="OpenSlide"))

            # STEP 3: Filtering the ROIs with foreground occupancy ratio and pixel empty-value and pixel variance
            empty_tile_bool_mark, tile_occupancy = check_an_empty_tile(a_tile_img,
                                                                       foreground_threshold, occupancy_threshold,
                                                                       pixel_std_threshold, extreme_value_portion_th)
            # STEP 4: prepare tile
            if empty_tile_bool_mark:
                n_discarded += 1
            else:
                # convert the cropped valid tile into PIL image with tile_size
                PIL_tile = resize_tile_to_PIL_target_tile(a_tile_img, tile_size)
                # todo in future we can use h5 to log tile infor

                # logging and save the image to h5
                tile_info = get_tile_info_dict(ROI_sample_from_WSI, tile_occupancy, tile_location,
                                               target_level_tilesize,
                                               rel_slide_dir=Path(ROI_sample_from_WSI['slide_id']),
                                               ROI_image_key=ROI_image_key)

                save_PIL_image(PIL_tile, os.path.join(output_tiles_dir, tile_info["tile_image_path"]))

        except Exception as e:
            # sometimes certain tiles is broken (for pixel failure) and we need to skip
            n_failed_tiles += 1
            descriptor = get_tile_descriptor(tile_location)
            # we write down these failed tiles into the log
            failed_tiles_file.write(descriptor + '\n')
            traceback.print_exc()
            warnings.warn(f"An error occurred while saving tile "
                          f"{get_tile_id(ROI_sample_from_WSI['slide_id'], tile_location)}: {e}")
        else:
            if not empty_tile_bool_mark:
                # record the tile information into tile_info_list
                tile_info_list.append(tile_info)
                dataset_row = format_csv_row(tile_info, keys_to_save, metadata_keys)
                dataset_csv_file.write(dataset_row + '\n')

    # Explicitly delete the large objects
    del WSI_image_obj
    # Force garbage collection
    gc.collect()

    # log ROI selection infor
    logging.info(f"Percentage tiles discarded: {n_discarded / n_tiles * 100:.2f}")

    return tile_info_list, n_failed_tiles


# Other tools
def get_remaining_tile_locations(all_locations, chunk_locations):
    """
    Get the difference between two lists of positions.

    :param all_locations: List of all positions (list of tuples).like [(y1,x1),(y2,x2),...]
    :param chunk_locations: List of positions to be removed (list of tuples).like [(y1,x1),(y2,x2),...]
    :return: List of positions remaining after removal.
    """
    all_locations_set = set(all_locations)
    chunk_locations_set = set(chunk_locations)

    remaining_locations_set = all_locations_set - chunk_locations_set
    remaining_locations = list(remaining_locations_set)

    return remaining_locations


def build_tile_loc_y_x(y, x):
    """
    this is the essencial naming setting for tile name
    """
    tile_loc_y_x = f"{y:05d}y_{x:05d}x"
    return tile_loc_y_x


def read_tile_name_for_loc_y_x(tile_name, suffix='.jpeg'):
    y, x = tile_name.split(suffix)[0].split('_')  # EG: 44004y_11136x.jpeg
    y, x = int(y.replace('y', '')), int(x.replace('x', ''))
    return y, x


def build_tile_name_for_loc_y_x(y, x, suffix='.jpeg'):
    tile_name = build_tile_loc_y_x(y, x) + suffix
    return tile_name


def get_tile_descriptor(tile_location) -> str:
    """Format the YX tile coordinates into a tile descriptor.

    :param tile_location: (y,x) the encoded tile coordinates
    :return: a string describing the tile location like 01234y_05678x

    """
    tile_loc_y_x = build_tile_loc_y_x(tile_location[0], tile_location[1])
    return tile_loc_y_x


def get_tile_id(slide_id: str, tile_location: Sequence[int]) -> str:
    """
    Format the slide_feature ID and YX tile coordinates into a unique tile ID.

    :param slide_id: id name of the WSI
    :param tile_location: Sequence[int] get the encoded tile coordinates

    :return: a string describing the tile WSI_name and location like WSIname_01234y_05678x
    """
    return f"{slide_id}.{get_tile_descriptor(tile_location)}"


def get_tile_info_dict(sample: Dict["SlideKey", Any], occupancy: float, tile_location: Sequence[int],
                       target_level_tilesize, rel_slide_dir: Path,
                       ROI_image_key: str = "tile_image_path", suffix='.jpeg') -> Dict["TileKey", Any]:
    """Map slide_feature information and tiling outputs into tile-specific information dictionary.

    :param sample: Slide dictionary.
    :param occupancy: Estimated tile foreground occuppancy.
    :param tile_location: level-0 Tile YX coordinates.
    :param target_level_tilesize:

    :param rel_slide_dir: Directory where tiles are saved, relative to dataset root.

    :param ROI_image_key: ROI Image key in the input and output dictionaries. default is 'tile_image_path'

    :return: Tile information dictionary.
    """
    slide_id = sample["slide_id"]
    descriptor = get_tile_descriptor(tile_location)
    rel_image_path = f"{rel_slide_dir}/{descriptor}" + suffix

    tile_info = {
        "slide_id": slide_id,
        "tile_id": get_tile_id(slide_id, tile_location),
        ROI_image_key: rel_image_path,
        "label": sample.get("label", None),
        "tile_y": tile_location[0],
        "tile_x": tile_location[1],
        'target_level_tilesize': target_level_tilesize,
        "occupancy": occupancy,
        "metadata": {"slide_" + key: value for key, value in sample["metadata"].items()}
    }

    return tile_info


def format_csv_row(tile_info: Dict["TileKey", Any], keys_to_save: Iterable["TileKey"],
                   metadata_keys: Iterable[str]) -> str:
    """Format tile information dictionary as a row to write to a dataset CSV tile.

    :param tile_info: Tile information dictionary.
    :param keys_to_save: Which main keys to include in the row, and in which order.
    :param metadata_keys: Likewise for metadata keys.
    :return: The formatted CSV row.
    """
    tile_slide_metadata = tile_info.pop("metadata")

    fields = [str(tile_info[key]) for key in keys_to_save]
    fields.extend(str(tile_slide_metadata[key]) for key in metadata_keys)
    dataset_row = ','.join(fields)

    return dataset_row


def is_already_processed(output_tiles_dir, rel_slide_dir, logging=None):
    """
    check whether the slide_feature has been processed with all output log files

    Args:
        output_tiles_dir: output folder for tiles (on WSI name)
        rel_slide_dir: jpeg folder name
        logging: logger
    Returns:

    """
    if not output_tiles_dir.exists():
        if logging:
            logging.info(f">>> no valid sample found in {output_tiles_dir}")
        return False

    dataset_img_path = output_tiles_dir / rel_slide_dir
    if len(list(dataset_img_path.glob("*.jpeg"))) == 0:
        if logging:
            logging.info(f">>> no valid jpeg found in {dataset_img_path}")
        return False

    dataset_csv_path = output_tiles_dir / "dataset.csv"
    try:
        df = pd.read_csv(dataset_csv_path)
    except:
        if logging:
            logging.info(f">>> no valid csv found in {dataset_csv_path}")
        return False

    return len(df) > 0


def merge_dataset_csv_files(dataset_dir: Path) -> Path:
    """Combines all "*/dataset.csv" files into a single "dataset.csv" file in the given directory."""
    full_csv = dataset_dir / "dataset.csv"

    # print("List of files")
    # print([str(file) + '\n' for file in dataset_dir.glob("*/dataset.csv")])
    with full_csv.open('w') as full_csv_file:
        # full_csv_file.write(','.join(CSV_COLUMNS) + '\n')  # write CSV header
        first_file = True
        for slide_csv in tqdm(dataset_dir.glob("*/dataset.csv"), desc="Merging dataset.csv", unit='file'):
            logging.info(f"Merging slide_feature {slide_csv}")
            content = slide_csv.read_text()
            if not first_file:
                content = content[content.index('\n') + 1:]  # discard header row for all but the first file
            full_csv_file.write(content)
            first_file = False
    return full_csv


# visualization tools
def visualize_foreground_mask(image: np.ndarray, mask: np.ndarray, threshold: float) -> None:
    """
    Visualize the original image and its foreground mask side-by-side.

    :param image: The original image array in (C, H, W) format.
    :param mask: The foreground mask array in (H, W) format.
    :param threshold: The threshold value used for segmentation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(np.transpose(image, (1, 2, 0)))
    axes[0].set_title("Original Image")

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f"Foreground Mask (Threshold: {threshold:.2f})")

    plt.show()
    fig.savefig('./foreground_mask.jpeg')
    plt.close()


def assemble_tiles_2d(tiles: np.ndarray, coords: np.ndarray, fill_value: Optional[int] = 255,
                      channels_first: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
    """Assembles a 2D array from sequences of tiles and coordinates.

    :param tiles: Stack of tiles with batch dimension first.
    :param coords: YX tile coordinates, assumed to be spaced by multiples of `tile_size` (shape: [N, 2]).
    :param tile_size: Size of each tile; must be >0.
    :param fill_value: Value to assign to empty elements (default: 255 for white).
    :param channels_first: Whether each tile is in CHW (`True`, default) or HWC (`False`) layout.
    :return: A tuple containing:
        - `array`: The reassembled 2D array with the smallest dimensions to contain all given tiles.
        - `offset`: The lowest YX coordinates.
        - `offset`: YX offset introduced by the assembly.
        Added this as tile coordinates to obtain indices for the assembled array.
    """
    if coords.shape[0] != tiles.shape[0]:
        raise ValueError(f"Tile coordinates and values must have the same length, "
                         f"got {coords.shape[0]} and {tiles.shape[0]}")

    if channels_first:
        n_tiles, channels, tile_size, _ = tiles.shape
    else:
        n_tiles, tile_size, _, channels = tiles.shape
    tile_ys, tile_xs = coords.T

    y_min, y_max = min(tile_ys), max(tile_ys + tile_size)
    x_min, x_max = min(tile_xs), max(tile_xs + tile_size)
    height = y_max - y_min
    width = x_max - x_min

    output_shape = (channels, height, width) if channels_first else (height, width, channels)
    array = np.full(output_shape, fill_value, dtype=tiles.dtype)

    offset = np.array([-y_min, -x_min])  # todo, I don't understand what is this
    for idx in range(n_tiles):
        row = coords[idx, 0] + offset[0]  # y axis
        col = coords[idx, 1] + offset[1]  # x axis
        if channels_first:
            array[:, row:row + tile_size, col:col + tile_size] = tiles[idx]
        else:
            array[row:row + tile_size, col:col + tile_size, :] = tiles[idx]

    return array, offset


# use to check tiles
def downsample_chw_numpy_image(image: np.ndarray, scale_factor: Optional[float] = None) -> np.ndarray:
    """Downsamples a CHW numpy array while maintaining the same aspect ratio.

    :param image: The input image in CHW format.
    :param scale_factor: The factor by which to downsample the image (e.g., 0.5 for half size).
    :return: The downsampled image in CHW format.
    """
    channels, height, width = image.shape
    scale_factor = scale_factor or 2000.0 / max(height, width)

    if not (0 < scale_factor <= 1):
        raise ValueError("Scale factor must be in the range (0, 1].")

    # Calculate new dimensions
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Initialize an array for the downsampled image
    downsampled_image = np.zeros((channels, new_height, new_width), dtype=image.dtype)

    # Downsample each channel
    for c in range(channels):
        downsampled_image[c] = cv2.resize(image[c], (new_width, new_height), interpolation=cv2.INTER_AREA)

    return downsampled_image


def save_openslide_thumbnail(openslide_obj, output_path, size_target=1024, level0_bbox=None):
    # Calculate the scale factor to resize the thumbnail
    scale = size_target / max(openslide_obj.dimensions)
    # Get the thumbnail image
    thumbnail = openslide_obj.get_thumbnail([int(m * scale) for m in openslide_obj.dimensions])

    # Draw the bounding box if provided
    if level0_bbox is not None:
        draw = ImageDraw.Draw(thumbnail)
        # Scale the bounding box coordinates to match the thumbnail size
        scaled_bbox = [int(level0_bbox.x * scale), int(level0_bbox.y * scale),
                       int((level0_bbox.x + level0_bbox.w) * scale),
                       int((level0_bbox.y + level0_bbox.h) * scale)]
        # Draw the rectangle on the thumbnail image
        draw.rectangle(scaled_bbox, outline="red", width=3)

    # Save the thumbnail image
    thumbnail.save(output_path)
    logging.info(f"Saving thumbnail {output_path}, shape {thumbnail.size}")


def visualize_tile_locations(sample, output_path, tile_info_list,
                             reader: WSIReader = WSIReader(backend="OpenSlide"),
                             image_key: str = "slide_image_path", size_thumbnail=1024):
    """
    Visualize tile locations on a downscaled thumbnail of a WSI image.

    Parameters:
    - sample: dictionary containing the WSI image path and other relevant information
    - output_path: path to save the output visualization
    - tile_info_list: list of dictionaries containing tile information with keys 'tile_x', 'tile_y', and 'target_level_tilesize'
    - reader: WSIReader object to read the WSI image
    - image_key: Image key in the input and output dictionaries. default is 'slide_image_path'
    - size_thumbnail: size of the thumbnail image to create for visualization
    """
    # Read the WSI image
    WSI_image_obj: OpenSlide = reader.read(sample[image_key])

    # Calculate the scale factor to resize the thumbnail
    scale = size_thumbnail / max(WSI_image_obj.dimensions)

    # Get the thumbnail image (WSI_image_obj.dimensions is level-0 scale)
    thumbnail = WSI_image_obj.get_thumbnail([int(m * scale) for m in WSI_image_obj.dimensions])

    # Calculate the downscale factor
    downscale_factor = 1 / scale  # converting the level-0 location to thumbnail location

    fig, ax = plt.subplots()
    ax.imshow(thumbnail)
    rects = []
    for tile_info in tile_info_list:
        # Convert level-0 coordinate to the thumbnail coordinate
        xy = (tile_info["tile_x"] / downscale_factor,
              tile_info["tile_y"] / downscale_factor)

        # Calculate the tile size in the thumbnail
        thumbnail_tile_size = int(tile_info['target_level_tilesize'] * sample["WSI_loc_scale"]  # level0 scale
                                  / downscale_factor)

        rects.append(patches.Rectangle(xy, thumbnail_tile_size, thumbnail_tile_size))

    # Paint black to the location of tiles
    pc = collections.PatchCollection(rects, match_original=True, alpha=0.5, edgecolor="black")
    pc.set_array(np.array([100] * len(tile_info_list)))
    ax.add_collection(pc)
    fig.savefig(output_path)
    plt.close()


# visualization for putting back tiles
def visualize_CHW_numpy_image(image_array, output_path):
    """

    # put back tiles for visualization
        assemble_img_array, offset = assemble_tiles_2d(image_tiles, tile_locations)
        assemble_img_array = downsample_chw_numpy_image(assemble_img_array)
        visualize_CHW_numpy_image(assemble_img_array, thumbnail_dir / (slide_image_path.name + "_roi_recompose.jpeg"))


    """
    # Convert from (channels, height, width) to (height, width, channels)
    image_array = np.transpose(image_array, (1, 2, 0))

    # Ensure the values are in the range [0, 255] and convert to uint8
    if image_array.max() <= 1.0:  # Assuming the input array is in range [0, 1]
        image_array = (image_array * 255).astype(np.uint8)
    else:  # Assuming the input array is already in range [0, 255]
        image_array = image_array.astype(np.uint8)

    # Display the image using Matplotlib
    fig, ax = plt.subplots()
    ax.imshow(image_array)
    fig.savefig(output_path)
    plt.close()
