"""
ROI/WSI Segmentation tools   Script  ver： Sep 9th 12:30


"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import PIL
import skimage.filters
from monai.config.type_definitions import KeysCollection
from monai.data.wsi_reader import WSIReader
from monai.transforms.transform import MapTransform
from openslide import OpenSlide

# ROI/WSI Segmentation tools
def get_luminance(image_or_images: np.ndarray) -> np.ndarray:
    """
    Convert the RGB image array to luminance.

    :param image_or_images: The RGB image array in (*, C, H, W) format.
    :return: The luminance image array in (*, H, W) format.
    """
    # Assuming the input is in the format (*, C, H, W) and the channels are RGB
    if image_or_images.ndim == 4:
        # Convert RGB to luminance for each image in the batch
        return np.dot(image_or_images.transpose(0, 2, 3, 1), [0.2126, 0.7152, 0.0722])
    elif image_or_images.ndim == 3:
        # Convert RGB to luminance for a single image
        return np.dot(image_or_images.transpose(1, 2, 0), [0.2126, 0.7152, 0.0722])
    else:
        raise ValueError("Expected an image array with 3 or 4 dimensions.")


def segment_foreground(image_or_images: np.ndarray, threshold: Optional[float] = None) \
        -> Tuple[np.ndarray, float]:
    """Segment the given slide_feature by thresholding its luminance.

    :param image_or_images: The RGB image array in (*, C, H, W) format.
    :param threshold: Pixels with luminance below this value will be considered foreground.
    If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.

    :return: the boolean output array of foreground_mask and the threshold used.
    foreground_mask is (*, H, W) boolean array indicating whether the pixel is foreground or not
    """
    luminance = get_luminance(image_or_images)  # -> (*, H, W)
    if threshold is None:
        threshold = skimage.filters.threshold_otsu(luminance)
    # logging.info(f"Otsu threshold from luminance: {threshold}")

    # Foreground is where luminance is greater than the threshold
    foreground_mask = (luminance < threshold)  # True is foreground

    # Apply binary closing to connect the very near parts
    foreground_mask = skimage.morphology.binary_closing(foreground_mask)

    return foreground_mask, threshold


# ROI filtering tools
def ROIs_occupancy_filtering(foreground_mask: np.ndarray, occupancy_threshold: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Process the occupancy mask for ROI sequence and make selection array
    Idea is to exclude tiles that are mostly background based on estimated occupancy.

    :param foreground_mask: Boolean array of shape (*, H, W). * is the number of ROIs for this WSI

    :param occupancy_threshold: filtering threshold, ROIs with a lower occupancy will be discarded.

    :return: selections, occupancies
        selection: boolean arrays of shape (*,), or scalars if `foreground_mask` is for one single tile.
                    It contains which tiles were selected
        occupancies: float arrays of shape (*,), or scalars if `foreground_mask` is for one single tile.
                    It is the estimated occupancies for each ROI.
    """
    if occupancy_threshold < 0. or occupancy_threshold > 1.:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")

    occupancy = foreground_mask.mean(axis=(-2, -1), dtype=np.float16)
    # boolean arrays of shape (*,), or scalars if `foreground_mask` is for one single tile.
    selections = (occupancy > occupancy_threshold).squeeze()
    # float arrays of shape (*,), or scalars if `foreground_mask` is for one single tile.
    occupancies = occupancy.squeeze()

    return selections, occupancies   # type: ignore


def incremental_std(flattened_tiles, pixel_std_threshold=5, batch_size=1000):
    """
    Args:
        flattened_tiles: NumPy array of shape (N, C, pixels)
        pixel_std_threshold: Threshold for standard deviation to consider an ROI as 'empty'
        batch_size: Number of samples to process in each batch

    Returns:
        low_std_mask: Boolean array of shape (N,) indicating which ROIs are 'empty'
    """
    N = flattened_tiles.shape[0]

    # Initialize an array to store the low std mask results
    low_std_mask = np.empty(N, dtype=bool)

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        flattened_batch = flattened_tiles[i:end]

        # Calculate the standard deviation across the flattened spatial dimensions for each channel
        std_rgb = flattened_batch.std(axis=2)  # [batch_size, C] array

        # Calculate the mean standard deviation across all channels for each tile
        std_rgb_mean = std_rgb.mean(axis=1)  # [batch_size] array

        # Create a boolean mask indicating which ROIs are considered 'empty' based on the threshold
        low_std_mask[i:end] = std_rgb_mean < pixel_std_threshold  # [batch_size] boolean array

    return low_std_mask


# this is designed for chunk process, but it is not needed now
def check_empty_tiles(tiles: np.ndarray,
                      foreground_threshold: float, occupancy_threshold: float,
                      pixel_std_threshold: int = 5,
                      extreme_value_portion_th: float = 0.5) -> np.ndarray:
    """
    Determine which ROI image (tile) is empty for a sequence.  # fixme change to a ROI

    fixme: Hacky. This is depends on the value variance and ratio of pixels being 0

    :param tiles: The tile array in (N, C, H, W) format.

    :param foreground_threshold: The threshold for identifying the foreground
    :param occupancy_threshold: The threshold of foreground occupancy to say this ROI image is too 'empty'

    :param pixel_std_threshold: The threshold for the pixel variance at one ROI
                                to say this ROI image is too 'empty'
    :param extreme_value_portion_th: The threshold for the ratio of the pixels being 0 of one ROI,
                                    to say this ROI image is too 'empty'

    :return: Boolean array of shape (N,). an union of the two filtering methods
    """
    # STEP 1: Filtering the ROIs with foreground ratio
    # generate the foreground mask for each ROI in the sequence, based on the value of luminance at each pixel
    foreground_masks, _ = segment_foreground(tiles, foreground_threshold)
    # filtering the ROI tiles
    foreground_selected, occupancies = ROIs_occupancy_filtering(foreground_masks, occupancy_threshold)

    # STEP 2: Filtering the ROIs with pixel variance ratio
    # calculate standard deviation of rgb ROI image series
    N, C, H, W = tiles.shape
    flattened_tiles = tiles.reshape(N, C, H * W)  # -> N,C,pixels

    low_std_mask = incremental_std(flattened_tiles, pixel_std_threshold=pixel_std_threshold, batch_size=1000)

    '''
    # not by batch
    std_rgb = flattened_tiles[:, :, :].std(axis=2)  # [N,C] value is std across all locations in each channel
    std_rgb_mean = std_rgb.mean(axis=1)  # [N] value is std across all channel and all locations
    # [N] boolean array, value to say which ROIs are 'empty' (by having low variance)
    low_std_mask = std_rgb_mean < pixel_std_threshold
    '''

    # STEP 3: Filtering the ROIs with extreme_value ratio
    # count 0 pixel values
    extreme_value_count = ((flattened_tiles == 0)).sum(axis=2)  # -> [N,C], value is zero_pixels number
    # -> [N,C], value is zero_pixels ratio for all locations in each channel
    extreme_value_proportion = extreme_value_count / (H * W)
    # [N] value is zero_pixels ratio across all channel and all locations
    extreme_value_proportion_mean = extreme_value_proportion.max(axis=1)
    # [N] boolean array, value to say which ROIs are 'empty' (by having alot of empty values)
    extreme_value_mask = extreme_value_proportion_mean > extreme_value_portion_th

    # the conclusion for selection is the union of the two filtering methods
    empty_tile_bool_mask = (low_std_mask | extreme_value_mask)
    selected = foreground_selected & (~empty_tile_bool_mask)
    return selected, occupancies


def check_an_empty_tile(tile: np.ndarray,
                        foreground_threshold: float, occupancy_threshold: float,
                        pixel_std_threshold: int = 5,
                        extreme_value_portion_th: float = 0.5) -> np.ndarray:
    """
    Determine which ROI image (tile) is empty for a sequence.

    fixme: Hacky. This is depends on the value variance and ratio of pixels being 0

    :param tile: The tile array in (C, H, W) format.

    :param foreground_threshold: The threshold for identifying the foreground
    :param occupancy_threshold: The threshold of foreground occupancy to say this ROI image is too 'empty'

    :param pixel_std_threshold: The threshold for the pixel variance at one ROI
                                to say this ROI image is too 'empty'
    :param extreme_value_portion_th: The threshold for the ratio of the pixels being 0 of one ROI,
                                    to say this ROI image is too 'empty'

    :return: True (tile is empty) or False (tile is not empty)
    """
    # generate the foreground mask for each ROI in the sequence, based on the value of luminance at each pixel
    foreground_mask, _ = segment_foreground(tile, foreground_threshold)
    # foreground_mask is (1, H, W) boolean array indicating whether the pixel is foreground or not
    occupancy = float(foreground_mask.mean(axis=(-2, -1), dtype=np.float16))
    # occupancy_threshold
    empty_occupancy_bool_mark = occupancy < occupancy_threshold
    if empty_occupancy_bool_mark:
        return True,occupancy  # early stop

    # calculate standard deviation of rgb ROI image series
    C, H, W = tile.shape
    flattened_tiles = tile.reshape(C, H * W)  # -> C,pixels

    std_rgb = flattened_tiles[:, :].std(axis=-1)  # [C] value is std across all locations in each channel
    std_rgb_mean = float(std_rgb.mean(axis=0))  # [1] value is std across all channel and all locations
    # boolean, value to say if ROI is 'empty' (by having low variance)
    low_std_mark = std_rgb_mean < pixel_std_threshold
    if low_std_mark:
        return True,occupancy  # early stop

    # count 0 pixel values
    extreme_value_count = ((flattened_tiles == 0)).sum(axis=-1)  # -> [C], value is zero_pixels number
    # -> [C], value is zero_pixels ratio for all locations in each channel
    extreme_value_proportion = extreme_value_count / (H * W)
    # [1] value is zero_pixels ratio across all channel and all locations
    extreme_value_proportion_mean = float(extreme_value_proportion.max(axis=0))  # fixme max ?
    # boolean, value to say if ROI is 'empty' (by having alot of empty values)
    extreme_value_mark = extreme_value_proportion_mean > extreme_value_portion_th

    # the conclusion for selection is the union of the three filtering methods
    if extreme_value_mark:
        return True,occupancy
    else:
        # only when the three steps all say False, a tile can pass the process
        return False,occupancy


# General image tools
def save_chw_image(array_chw: np.ndarray, path: Path) -> PIL.Image:
    """
    Save an image array in (C, H, W) format to disk.

    :param array_chw: numpy image in CHW RGB format
    :param path: path of saving a .jpeg ROI
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = PIL.Image.fromarray(array_hwc)
    pil_image.convert('RGB').save(path)
    return pil_image


def save_PIL_image(pil_image: np.ndarray, path: Path) -> PIL.Image:
    """
    Save an image array in (C, H, W) format to disk.

    :param pil_image: numpy image in PIL.Image format
    :param path: path of saving a .jpeg ROI
    """
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.convert('RGB').save(path)
    return pil_image


