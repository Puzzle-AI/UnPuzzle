"""
tools for managing bbox    Script  ver： Aug 8th 17:00
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Box:
    """Utility class representing rectangular regions in 2D images.

    :param x: Horizontal coordinate of the top-left corner.
    :param y: Vertical coordinate of the top-left corner.
    :param w: Box width.
    :param h: Box height.
    :raises ValueError: If either `w` or `h` are <= 0.
    """
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0:
            raise ValueError(f"Width must be strictly positive, received {self.w}")
        if self.h <= 0:
            raise ValueError(f"Height must be strictly positive, received {self.w}")

    def __add__(self, shift: Sequence[int]) -> 'Box':
        """Translates the box's location by a given shift.

        :param shift: A length-2 sequence containing horizontal and vertical shifts.
        :return: A new box with updated `x = x + shift[0]` and `y = y + shift[1]`.
        :raises ValueError: If `shift` does not have two elements.
        """
        if len(shift) != 2:
            raise ValueError("Shift must be two-dimensional")
        return Box(x=self.x + shift[0],
                   y=self.y + shift[1],
                   w=self.w,
                   h=self.h)

    def __mul__(self, factor: float) -> 'Box':
        """Scales the box by a given factor, e.g. when changing resolution.

        :param factor: The factor by which to multiply the box's location and dimensions.
        :return: The updated box, with location and dimensions rounded to `int`.
        """
        return Box(x=int(self.x * factor),
                   y=int(self.y * factor),
                   w=int(self.w * factor),
                   h=int(self.h * factor))

    def __rmul__(self, factor: float) -> 'Box':
        """Scales the box by a given factor, e.g. when changing resolution.

        :param factor: The factor by which to multiply the box's location and dimensions.
        :return: The updated box, with location and dimensions rounded to `int`.
        """
        return self * factor

    def __truediv__(self, factor: float) -> 'Box':
        """Scales the box by a given factor, e.g. when changing resolution.

        :param factor: The factor by which to divide the box's location and dimensions.
        :return: The updated box, with location and dimensions rounded to `int`.
        """
        return self * (1. / factor)

    def add_margin(self, margin: int) -> 'Box':
        """Adds a symmetric margin on all sides of the box.

        :param margin: The amount by which to enlarge the box.
        :return: A new box enlarged by `margin` on all sides.
        """
        return Box(x=self.x - margin,
                   y=self.y - margin,
                   w=self.w + 2 * margin,
                   h=self.h + 2 * margin)

    def clip(self, other: 'Box') -> Optional['Box']:
        """Clips a box to the interior of another.

        This is useful to constrain a region to the interior of an image.

        :param other: Box representing the new constraints.
        :return: A new constrained box, or `None` if the boxes do not overlap.
        """
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        try:
            return Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
        except ValueError:  # Empty result, boxes don't overlap
            return None

    def merge(self, other: 'Box') -> 'Box':
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x + self.w, other.x + other.w)
        y2 = max(self.y + self.h, other.y + other.h)
        return Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    def to_slices(self) -> Tuple[slice, slice]:
        """Converts the box to slices for indexing arrays.

        For example: `my_2d_array[my_box.to_slices()]`.

        :return: A 2-tuple with vertical and horizontal slices.
        """
        return (slice(self.y, self.y + self.h),
                slice(self.x, self.x + self.w))

    @staticmethod
    def from_slices(slices: Sequence[slice]) -> 'Box':
        """Converts a pair of vertical and horizontal slices into a box.

        :param slices: A length-2 sequence containing vertical and horizontal `slice` objects.
        :return: A box with corresponding location and dimensions.
        """
        vert_slice, horz_slice = slices
        return Box(x=horz_slice.start,
                   y=vert_slice.start,
                   w=horz_slice.stop - horz_slice.start,
                   h=vert_slice.stop - vert_slice.start)


def calculate_area(slice_obj):
    """Calculate the area of an object given its slice."""
    y_slice, x_slice = slice_obj
    height = y_slice.stop - y_slice.start
    width = x_slice.stop - x_slice.start
    return height * width


def get_top_n_slices_by_size(mask, maximum_top_n=5):
    """Return the top N largest slices by size."""
    labeled_mask, num_features = ndimage.label(mask > 0)  # Label connected components

    slices = ndimage.find_objects(labeled_mask)  # Find slices for labeled objects

    if not slices:
        raise RuntimeError("No objects found in the mask")

    # Calculate area for each slice
    slice_areas = [(s, calculate_area(s)) for s in slices]

    # Sort slices by area in descending order
    sorted_slices = sorted(slice_areas, key=lambda x: x[1], reverse=True)

    # Check if there are fewer slices than top_n
    num_slices_to_return = min(maximum_top_n, len(sorted_slices))

    # Select the top N largest slices or all available if less than top_n
    top_n_slices = [s for s, area in sorted_slices[:num_slices_to_return]]

    return top_n_slices


def merge_overlapping_boxes(box_list):
    """Merge overlapping bounding boxes."""
    merged_boxes = []
    while box_list:
        box = box_list.pop(0)
        merge_with = [b for b in box_list if (box.x < b.x + b.w and box.x + box.w > b.x and box.y < b.y + b.h and box.y + box.h > b.y)]
        for b in merge_with:
            box = box.merge(b)
            box_list.remove(b)
        merged_boxes.append(box)
    return merged_boxes


def get_bounding_box_from_slices(slices):
    box_list = []
    for slice in slices:
        box_list.append(Box.from_slices(slice))

    return box_list


def get_ROI_bounding_box_list(mask: np.ndarray, maximum_top_n=20) -> Box:
    """Extracts a bounding box from a binary 2D array.

    :param mask: A 2D array with 0 (or `False`) as background and >0 (or `True`) as foreground.
    top_n=5

    :return: The smallest box covering all non-zero elements of `mask`.
    :raises TypeError: When the input mask has more than two dimensions.
    :raises RuntimeError: When all elements in the mask are zero.
    """
    if mask.ndim != 2:
        raise TypeError(f"Expected a 2D array but got an array with shape {mask.shape}")

    slices = get_top_n_slices_by_size(mask, maximum_top_n=maximum_top_n)

    box_list = get_bounding_box_from_slices(slices)

    # merge overlapping bbox to reduce repeat calculation in future steps
    box_list = merge_overlapping_boxes(box_list)

    return box_list
