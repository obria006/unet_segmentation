""" Scripts to test and evaluate the segementation model """
import glob
from datetime import datetime
import os
import git
from tqdm import tqdm
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from src.models.dl4mia_tissue_unet.dl4mia_utils.metrics import binary_sem_seg_metrics
from src.models.dl4mia_tissue_unet.predict import Predicter, SegmenterWrapper
from src.visualization.visualize import plot_seg_boxplot
from src.models.dl4mia_tissue_unet.dl4mia_utils.general import load_yaml
from src.utils.paths import list_images


def load_model(ckpt_path: str, in_size: tuple = (128, 128)) -> SegmenterWrapper:
    """
    Loads the model from the given path

    Args:
        ckpt_path (str): Path to checkpoint for model
        in_size (tuple): Size which images will be resized to before input into model
    """
    return SegmenterWrapper(Predicter.from_ckpt(ckpt_path), in_size=in_size)


def get_data_paths(image_dirs: list, mask_dirs: list) -> tuple[list[str], list[str]]:
    """
    Returns image paths in `image_dirs` and `mask_paths` in mask paths

    Args:
        img_dirs (list): List of directories containing images
        mask_dirs (list): List of directories containing masks
    """
    img_paths = []
    mask_paths = []
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]
    if isinstance(mask_dirs, str):
        mask_dirs = [mask_dirs]
    for image_dir in image_dirs:
        imgs = list(list_images(image_dir))
        img_paths = img_paths + imgs
    for mask_dir in mask_dirs:
        masks = list(list_images(mask_dir))
        mask_paths = mask_paths + masks
    return img_paths, mask_paths


def read_images(img_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns np.array of image and mask

    Args:
        img_path (str): Path to image to read
        mask_path (str): Path to mask to read
    """
    assert os.path.exists(img_path)
    assert os.path.exists(mask_path)
    assert os.path.basename(img_path) == os.path.basename(
        mask_path
    ), "Mask and image basemane don't match"
    mask = imageio.v2.imread(mask_path)
    img = imageio.v2.imread(img_path)
    return img, mask


def compile_binary_seg_results(
    segmenter: SegmenterWrapper, img_paths: list[str], mask_paths: list[str]
) -> pd.DataFrame:
    """
    Return Dataframe of segmentation results for binary images

    Args:
        segmenter (SegmenterWrapper): interface with `predict` method that returns segmentation
        img_paths (list): Paths to images to segment. Nth index image corresponds to nth index mask
        mask_paths (list): Paths to masks to segment. Nth index mask corresponds to nth index image
    """
    # Segmentation results
    columns_names = ["img", "gt", "acc", "dice", "prec", "spec", "rec"]
    seg_df = pd.DataFrame(columns=columns_names)

    # Evaluate each image
    for img_path, mask_path in tqdm(
        list(zip(img_paths, mask_paths)), bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    ):
        img, mask = read_images(img_path, mask_path)

        # Segment the image
        seg, _ = segmenter.predict(img)

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

    return seg_df


def save_seg_results(seg_df: pd.DataFrame, output_dir: str, inc_datetime: bool = False):
    """
    Save the numerical segmentation results to a csv file

    Args:
        seg_df (pd.DataFrame): Dataframe of segmentaion results
        output_dir (str): Directory where to save file
        inc_datetime (bool): If true, prepends datetime to filename
    """
    # Get the git sha
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # Filename to save
    fname = f"{sha[0:7]}_seg_metrics.csv"

    # Get current datetime
    if inc_datetime:
        now = datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        fname = f"{date_str}_{fname}"

    fpath = f"{output_dir}/{fname}"
    seg_df.to_csv(fpath)


def compare_binary_images(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Return RGB image showing TP (white), FP (red), FN (blue), and TN (black)
    comparing ground truth versus prediction for binary images.

    Args:
        y_true (np.ndarray): Ground truth binary mask
        y_pred (np.ndarray): Predicted segmentation mask
    """
    # Confirm that images are binary
    assert len(np.unique(y_true)) <= 2
    assert len(np.unique(y_pred)) <= 2

    # Convert binary images to 0-1 labeled images
    if np.amax(y_true) > 0:
        y_true = (np.copy(y_true) * (1 / np.amax(y_true))).astype(bool)
    if np.amax(y_pred) > 0:
        y_pred = (np.copy(y_pred) * (1 / np.amax(y_pred))).astype(bool)

    # Binary masks for TP, FP, and FN
    TP = (y_true == True) & (y_pred == True)
    FP = (y_true == False) & (y_pred == True)
    FN = (y_true == True) & (y_pred == False)

    # Construct rgb image from predicted classes
    rgb = np.zeros((y_true.shape[0], y_true.shape[1], 3)).astype(np.uint8)
    rgb[TP, :] = (255, 255, 255)
    rgb[FP, :] = (255, 0, 0)
    rgb[FN, :] = (0, 0, 255)
    return rgb


def main(
    src_dir: str,
    data_dir: str,
    ckpt_name: str = "best.pth",
    compare: bool = True,
    max_compare: int = 100,
):
    """
    Runs segmentation model on image data to evaluate its performance

    Args:
        src_dir (str): Directory to output directory containing .pth checkpoints
        data_dir (str): Directory containing 'train', 'test', 'val' folders
        ckpt_name (str): Name of checkpoint file to load into model
        compare (bool): If true, saves rgb images showing TP, FP, FN, and TN
        max_compare (int): Maximum number of images to copmare and save from each data type
    """
    # Initialize evaluation output location
    eval_out_dir = f"{src_dir}/evaluation"
    if not os.path.isdir(eval_out_dir):
        os.makedirs(eval_out_dir)
        print(f"Created evaluation directory at: {eval_out_dir}")

    # Compile directories containing images and masks
    data_names = ["train", "val", "test"]
    img_dirs = {}
    mask_dirs = {}
    for data in data_names:
        dict_path = f"{src_dir}/{data}_dataset_dict.yaml"
        dataset_dict = load_yaml(dict_path)
        data_dir = dataset_dict["kwargs"]["data_dir"]
        if not os.path.isdir(data_dir):
            data_dir = data_dir.replace("../", "")
            if not os.path.isdir(data_dir):
                raise ValueError(f"No directory at: {data_dir}")
        img_dirs[data] = f"{data_dir}/{data}/images"
        mask_dirs[data] = f"{data_dir}/{data}/masks"

    # test_dict_path = f"{src_dir}/test_dataset_dict.yaml"
    # test_dataset_dict = load_yaml(test_dict_path)
    # data_dir = test_dataset_dict["kwargs"]["data_dir"]
    # if not os.path.isdir(data_dir):
    #     data_dir = data_dir.replace("../", "")
    #     if not os.path.isdir(data_dir):
    #         raise ValueError(f"No directory at: {data_dir}")
    # data_type = test_dataset_dict["kwargs"]["data_type"]
    # img_paths = list(list_images(f"{data_dir}/{data_type}/images"))
    # mask_paths = list(list_images(f"{data_dir}/{data_type}/masks"))

    # img_dirs = {data: f"{data_dir}/{data}/images" for data in data_names}
    # mask_dirs = {data: f"{data_dir}/{data}/masks" for data in data_names}

    # Get test and validation image paths
    test_val_img_dirs = [img_dirs["test"], img_dirs["val"]]
    test_val_mask_dirs = [mask_dirs["test"], mask_dirs["val"]]
    tv_img_paths, tv_mask_paths = get_data_paths(test_val_img_dirs, test_val_mask_dirs)

    # Load model
    ckpt_path = f"{src_dir}/{ckpt_name}"
    mdl = load_model(ckpt_path)

    # Compile the segmentation metrics for test and validation paths and save csv and plot
    seg_metrics_df = compile_binary_seg_results(
        segmenter=mdl, img_paths=tv_img_paths, mask_paths=tv_mask_paths
    )
    save_seg_results(seg_df=seg_metrics_df, output_dir=eval_out_dir)
    plot_seg_boxplot(seg_metrics_df, save=False)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    plot_name = f"{sha[:7]}_seg_metrics.svg"
    plot_path = f"{eval_out_dir}/{plot_name}"
    plt.savefig(plot_path)
    plt.show()

    if compare:
        # Process each train, val, test directory
        for data in data_names:
            print(
                f"Comparing GT to prediction for up to {max_compare} images in {data}."
            )
            img_paths, mask_paths = get_data_paths(img_dirs[data], mask_dirs[data])

            # Only evaluate at most "max_compare" images
            if len(img_paths) > max_compare:
                img_paths = img_paths[:max_compare]
                mask_paths = mask_paths[:max_compare]

            # Predict on each image and compare
            for img_path, mask_path in tqdm(
                list(zip(img_paths, mask_paths)),
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ):
                # Read image and mask pair
                img, mask = read_images(img_path, mask_path)

                # Segment the image and compare against ground truth
                seg, _ = mdl.predict(img)
                rgb = compare_binary_images(mask, seg)
                save_dir = f"{eval_out_dir}/comparison/{data}"
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                filename = img_path.split(os.path.sep)[-1]
                basename = os.path.splitext(filename)[0]
                save_path = f"{save_dir}/{basename}.png"
                imageio.imwrite(save_path, rgb)


if __name__ == "__main__":
    data_dir = (
        "C:/VirEnvs/OCT_segmentation/unet_segmentation/data/processed/OCT_scans_128x128"
    )
    src_dir = "src/models/dl4mia_tissue_unet/results/20230403_102456"
    main(src_dir=src_dir, data_dir=data_dir, compare=True, max_compare=5)
