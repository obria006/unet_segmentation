""" Fucntions for evaluating data for edge classification """
import glob
import time
import numpy as np
import tifffile
import cv2


def mask_statistics(train_dir: str):
    """
    Get statistics of the training data masks. All images in train dir
    must maintain equal scaling otherwise size will be incorrect.

    Args:
        train_dir (str): The path of the training data mask directory.

    Returns:
        dict: A dictionary containing statistics of the training data
    """
    fg_sizes = []
    bg_sizes = []
    mask_paths = glob.glob(f"{train_dir}/*.tif")
    img_shape = tifffile.imread(mask_paths[0]).shape
    for mask_path in mask_paths:
        fg_img = tifffile.imread(mask_path).astype(bool)
        assert fg_img.shape == img_shape
        bg_img = np.invert(fg_img)
        n_fg, fg_labels, fg_stats, _ = cv2.connectedComponentsWithStats(
            fg_img.astype(np.uint8)
        )
        n_bg, bg_labels, bg_stats, _ = cv2.connectedComponentsWithStats(
            bg_img.astype(np.uint8)
        )
        fg_sizes.extend(fg_stats[1:, cv2.CC_STAT_AREA])  # ignore the bg
        bg_sizes.extend(bg_stats[1:, cv2.CC_STAT_AREA])  # ignore the bg
    fg_sizes = np.array(fg_sizes)
    bg_sizes = np.array(bg_sizes)
    # preprocess to remove really small regions
    fg_sizes = np.delete(fg_sizes, np.where(fg_sizes < 10))
    bg_sizes = np.delete(bg_sizes, np.where(bg_sizes < 10))
    isize = img_shape[0] * img_shape[1]
    mask_stats = {
        "fg_min": np.amin(fg_sizes) / isize,
        "bg_min": np.amin(bg_sizes) / isize,
        "fg_1st": np.percentile(fg_sizes, 1) / isize,
        "bg_1st": np.percentile(bg_sizes, 1) / isize,
        "fg_5th": np.percentile(fg_sizes, 5) / isize,
        "bg_5th": np.percentile(bg_sizes, 5) / isize,
    }
    return mask_stats


if __name__ == "__main__":
    MASK_DIR = "data/processed/uncropped/train/masks"
    t0 = time.time()
    stats = mask_statistics(MASK_DIR)
    print(stats)
    print(f"Duration: {time.time() - t0}")
    """
    results on 08/26/2022 with 128x128 uncropped data:

    {'fg_min': 0.00909423828125,
    'bg_min': 0.0076904296875,
    'fg_1st': 0.0202532958984375,
    'bg_1st': 0.015189208984375,
    'fg_5th': 0.043450927734375006,
    'bg_5th': 0.037866210937500006}
    """
