""" Scripts to test and evaluate the segementation model """
import glob
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import tifffile
from src.models.dl4mia_tissue_unet.dl4mia_utils.metrics import binary_sem_seg_metrics
from src.models.dl4mia_tissue_unet.predict import Predicter, SegmenterWrapper


def load_model(ckpt_path):
    """Loads the model from the given path"""
    return SegmenterWrapper(Predicter.from_ckpt(ckpt_path), in_size=(128, 128))


def get_data_paths(image_dirs: list, mask_dirs: list):
    """Returns image paths in `image_dirs` and `mask_paths` in mask paths"""
    img_paths = []
    mask_paths = []
    for image_dir in image_dirs:
        imgs = glob.glob(f"{image_dir}/*.tif")
        img_paths = img_paths + imgs
    for mask_dir in mask_dirs:
        masks = glob.glob(f"{mask_dir}/*.tif")
        mask_paths = mask_paths + masks
    return img_paths, mask_paths


def read_images(img_path: str, mask_path: str):
    """Returns np.array of image and mask"""
    assert os.path.exists(img_path)
    assert os.path.exists(mask_path)
    assert os.path.basename(img_path) == os.path.basename(
        mask_path
    ), "Mask and image basemane don't match"
    mask = tifffile.imread(mask_path)
    img = tifffile.imread(img_path)
    return img, mask


def main():
    """Runs segmentation model on image data to evaluate its performance"""
    # Get data
    IMAGE_DIRS = [
        "data/processed/uncropped/test/images",
        "data/processed/uncropped/val/images",
    ]
    MASK_DIRS = [dirname.replace("images", "masks") for dirname in IMAGE_DIRS]
    img_paths, mask_paths = get_data_paths(IMAGE_DIRS, MASK_DIRS)

    # Load model
    UNET_CKPT = (
        "src/models/dl4mia_tissue_unet/results/20220824_181000_Colab_gpu/best.pth"
    )
    unet_model = load_model(UNET_CKPT)

    # Segmentation results
    columns_names = ["img", "gt", "acc", "dice", "prec", "spec", "rec"]
    seg_df = pd.DataFrame(columns=columns_names)

    # Evaluate each image
    for img_path, mask_path in tqdm(
        list(zip(img_paths, mask_paths)), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    ):
        img, mask = read_images(img_path, mask_path)
        img, mask = read_images(img_path, mask_path)

        # Segment the image
        seg, _ = unet_model.predict(img)

        # Compute segmentation metrics
        acc, dice, prec, spec, rec = binary_sem_seg_metrics(
            y_true=mask,
            y_pred=seg,
        )

        # Add to dataframe
        tmp_dict = {
            "img": [img_path],
            "gt": [mask_path],
            "acc": [acc],
            "dice": [dice],
            "prec": [prec],
            "spec": [spec],
            "rec": [rec],
        }
        tmp_df = pd.DataFrame(tmp_dict)
        seg_df = pd.concat([seg_df, tmp_df], ignore_index=True)

    print(seg_df)


if __name__ == "__main__":
    main()
