""" Fuctions for manipulating images """

from collections.abc import Iterable
import numpy as np
from matplotlib import pyplot as plt
import cv2


def blend_images(images: Iterable, weights: list = None):
    """
    Blends images together into single image. IF wieghts is none, all images
    are weighted equally

    Args:
        images: list of np.ndarray images to blend together (must be same shape)
        weights: list of weights for each image (must sum to 1)
    """
    # Validate that images are iterable with same shape
    if not isinstance(images, Iterable):
        raise ValueError("Images must be iterable")
    for image in images:
        if not image.shape == images[0].shape:
            raise ValueError("Images must have the same shape")

    # Set weights if not already set, make sure they sum to 1, and have same
    # number of elements as in images
    n_images = len(images)
    if weights is None:
        weights = [1 / n_images] * n_images
    if np.sum(np.array(weights)) != 1:
        raise ValueError("Weights must sum to 1")
    if len(images) != len(weights):
        raise ValueError("Number of images and weights must be same")

    # Iterate through images blending with weights
    # breakpoint()
    blended = images[0].astype(np.float32, copy=True)
    for i in range(1, n_images):
        blended = cv2.addWeighted(
            src1=blended,
            alpha=weights[i - 1],
            src2=images[i].astype(np.float32, copy=True),
            beta=weights[i],
            gamma=0,
        )
    blended = blended.astype(np.uint8)
    return blended


def binary_imfill(mask: np.ndarray, region: str):
    """
    Fill in holes like MATLAB's imfill
    """
    vals = np.unique(mask)
    for val in vals:
        if val not in [0, 1]:
            raise ValueError("Mask must be 0 or 1")
    if region not in ["fg", "bg"]:
        raise ValueError(f'region must be "fg" or "bg". {region} is invalid')

    # Convert mask to FSR uint8
    mask8 = (np.copy(mask) * 255).astype(np.uint8)

    # Fill in regions
    pad = np.zeros((mask8.shape[0] + 2, mask8.shape[1] + 2), dtype=mask8.dtype)
    if region == "fg":
        pad[1:-1, 1:-1] = mask8
        cv2.floodFill(pad, mask=None, seedPoint=(0, 0), newVal=255)
        fill = pad[1:-1, 1:-1]
        not_fill = np.bitwise_not(fill)
        out = not_fill | mask8
    else:
        mask_inv = np.bitwise_not(mask8)
        pad[1:-1, 1:-1] = mask_inv
        cv2.floodFill(pad, mask=None, seedPoint=(0, 0), newVal=255)
        fill = pad[1:-1, 1:-1]
        out = fill & mask8

    # Recovert to binary
    out = (out / 255).astype(mask.dtype)
    for val in np.unique(out):
        assert val in [0, 1]
    return out


def rm_components_by_size(
    mask: np.ndarray, size_frac: float, smaller: bool, region: str
):
    """
    Remove components in mask image based on size.

    Args:
        mask: mask image to process
        size_frac: size threshold as frac to size of mask
        smaller: if true, removes components smaller than size_frac, else removes larger
        region: which region to remove from mask ['fg', 'bg']
    """
    if region not in ["fg", "bg"]:
        raise ValueError(f'region must be "fg" or "bg". {region} is invalid')

    # Get connnected components in mask (invert mask if removing bg)
    tmp_mask = np.copy(mask).astype(bool)
    if region == "bg":
        tmp_mask = np.invert(tmp_mask)
    tmp_mask = tmp_mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tmp_mask)

    # Find regions that satisfy condictions for removal by size and polarity
    rm_inds = []
    for ind in range(1, n_labels):
        area = stats[ind, cv2.CC_STAT_AREA] / tmp_mask.size
        if area < size_frac and smaller is True:
            rm_inds.append(ind)
        elif area < size_frac and smaller is False:
            rm_inds.append(ind)
    # set the removeal labels to 0
    for label in range(n_labels):
        if label in rm_inds:
            labels[labels == label] = 0

    # reinvert bg before returning
    if region == "bg":
        labels = labels > 0
        labels = np.invert(labels)

    return (labels > 0).astype(mask.dtype)


def n_largest_components(mask: np.ndarray, n: int, exc_bg: bool = True):
    """
    Return labeled image of the n largest components

    Args:
        mask: binary mask from which to extract connected components
        n: number of largest area connected components to return
        exc_bg: if True, doesn't count background as part of n components
    """
    # connected components on image to get labels
    n_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask)

    # get the n largest labels in the labeled image
    if n == -1:
        return list(range(cc_stats.shape[0]))
    else:
        areas = cc_stats[:, cv2.CC_STAT_AREA]
        if exc_bg is True:
            areas[0] = 0
        largest_labels = []
        for i in range(n):
            ind = np.argmax(areas)
            largest_labels.append(ind)
            areas[ind] = 0

    # set the non-largest labels to 0 while sequentially labeling largest labels
    seq_label = 1
    for label in range(n_labels):
        if label in largest_labels:
            labels[labels == label] = seq_label
            seq_label += 1
        else:
            labels[labels == label] = 0

    return labels


def overlay_image(
    image: np.ndarray,
    edge_label_dict: dict = None,
    mask: np.ndarray = None,
    edge_labels=None,
):
    """
    Overlays an image with edge labels

    Args:
        image: image to overlay
        edge_label_dict: dictionary of edge labels
        mask: mask to overlay
        edge_labels: edge label mask to overlay
    """
    # Convert to rgb
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    else:
        rgb = np.copy(image).astype(np.uint8)

    # rgb triplets for different masks
    MASK_RGB = [216, 27, 96]
    APICAL_RGB = [30, 136, 229]
    BASAL_RGB = [255, 193, 7]

    # Create colored mask
    overlay = np.zeros_like(rgb)
    if mask is not None:
        overlay[mask == 1] = MASK_RGB
    if edge_labels is not None and edge_label_dict is not None:
        for label, edge_type in edge_label_dict.items():
            edge_mask = (edge_labels == label).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            edge_mask = cv2.dilate(edge_mask, kernel)
            if edge_type == "apical":
                overlay[edge_mask.astype(bool)] = APICAL_RGB
            elif edge_type == "basal":
                overlay[edge_mask.astype(bool)] = BASAL_RGB

    # blend the image and mask
    blended = blend_images([rgb, overlay], [0.7, 0.3])
    return blended
