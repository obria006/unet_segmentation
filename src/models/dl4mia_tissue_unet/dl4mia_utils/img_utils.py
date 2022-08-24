""" Fuctions to preprocess images for input to Unet """
import numpy as np


def normalize(
    pic: np.ndarray,
    pmin: float = 1,
    pmax: float = 99.8,
    axis=(1, 2),
    clip: bool = False,
    eps: float = 1e-20,
    dtype: np.dtype = np.float32,
):
    """
    Normalize the input image according to:

       pic_intensity - pmin_intensity
    -------------------------------------
    pmax_intensity - pmin_intensity + eps

    Args:
        pic: Input image
        pmin: Minimum percentile of the input image
        pmax: Maximum percentile of the input image
        axis: Axis to normalize
        clip: if true, `clips` values to 0-1
        eps: Small epsilon for numerical stability (prevent 0 division)
        dtype: Data type for normalization
    """
    min_int = np.percentile(pic, pmin, axis=axis, keepdims=True)
    max_int = np.percentile(pic, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(pic, min_int, max_int, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(img, min_int, max_int, clip=False, eps=1e-20, dtype=np.float32):
    """Does image normalization for normalize()"""
    if dtype is not None:
        img = img.astype(dtype, copy=False)
        min_int = (
            dtype(min_int)
            if np.isscalar(min_int)
            else min_int.astype(dtype, copy=False)
        )
        max_int = (
            dtype(max_int)
            if np.isscalar(max_int)
            else max_int.astype(dtype, copy=False)
        )
        eps = dtype(eps)

    img = (img - min_int) / (max_int - min_int + eps)

    if clip:
        img = np.clip(img, 0, 1)

    return img

def preprocess_image(image:np.ndarray):
    """
    Preprocess image for input into unet model.

    Steps:
        1. Adds new axis so img shape is (C, H, W)
        2. Applies normalization on the image
     """
    return normalize(image[np.newaxis, ...], axis=(1, 2))


def preprocess_mask(mask:np.ndarray):
    """
    Preprocess mask for input into unet model

    Steps:
        1. Adds new axis so img shape is (C, H, W)
    """
    return mask[np.newaxis, ...]
