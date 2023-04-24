# import the necessary packages
import os
import time
import random
import imageio
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from src.models.dl4mia_tissue_unet.dl4mia_utils.img_utils import preprocess_image
from src.models.dl4mia_tissue_unet.dl4mia_utils.general import load_yaml
from src.models.dl4mia_tissue_unet import (model as v1, model_v2 as v2)
from src.utils.paths import list_images


class Predicter:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_model(cls, model_path: str):
        model = torch.load(model_path, map_location=cls.device)
        model.to(cls.device)
        return cls(model)

    @classmethod
    def from_ckpt(cls, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location=cls.device)
        state_dict = checkpoint["model_state_dict"]
        model_dict = checkpoint["model_dict"]
        print(f"Loaded checkpoint: {ckpt_path}")
        for key in checkpoint:
            if "state_dict" not in key and "logger_data" not in key:
                print(f"\t{key} = {checkpoint[key]}")
        model = v2.UNet(**model_dict["kwargs"])
        model.load_state_dict(state_dict, strict=True)
        model.to(cls.device)
        return cls(model)

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, image, gt=None):
        with torch.no_grad():
            im = preprocess_image(image)
            multiple_y = im.shape[1] // 8
            multiple_x = im.shape[2] // 8

            if im.shape[1] % 8 != 0:
                diff_y = 8 * (multiple_y + 1) - im.shape[1]
            else:
                diff_y = 0
            if im.shape[2] % 8 != 0:
                diff_x = 8 * (multiple_x + 1) - im.shape[2]
            else:
                diff_x = 0
            p2d = (
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            )  # last dim, second last dim

            # Process and predict differently for differeent models
            if isinstance(self.model, v2.UNet):
                im = np.array(im)
                im = im[np.newaxis,:]
                im = torch.from_numpy(im).to(self.device)
                output = self.model(im)  # B 3 Y X
                output_softmax = torch.sigmoid(output[0,0,:])
            elif isinstance(self.model, v1.UNet):
                im = F.pad(torch.from_numpy(im), p2d, "reflect").to(self.device)
                output = self.model(im)  # B 3 Y X
                output_softmax = torch.sigmoid(output[0])

            # Threshold prediction to genrate binary labeled map
            seed_map = output_softmax.cpu().detach().numpy()  # Y X
            pred_fg_thresholded = seed_map > 0.5

            if (diff_y - diff_y // 2) != 0:
                pred_fg_thresholded = pred_fg_thresholded[
                    diff_y // 2 : -(diff_y - diff_y // 2), ...
                ]
                seed_map = seed_map[diff_y // 2 : -(diff_y - diff_y // 2), ...]
            if (diff_x - diff_x // 2) != 0:
                pred_fg_thresholded = pred_fg_thresholded[
                    ..., diff_x // 2 : -(diff_x - diff_x // 2)
                ]
                seed_map = seed_map[..., diff_x // 2 : -(diff_x - diff_x // 2)]

        return pred_fg_thresholded, seed_map


class SegmenterWrapper:
    """
    Wraps the segmenter class to handle input and output image resizing
    for the segmentation network when used for inference.
    """

    def __init__(self, segmenter: Predicter, in_size: tuple = (128, 128)):
        """
        Args:
            segmenter: Model to perform tissue segmentation
            in_size: size to scale image to before segmentation
        """
        self.segmenter = segmenter
        self.in_size = in_size

    def predict(self, img: np.ndarray, scale_out: bool = True):
        """
        Predict the segmentation of an input image after resizing to `in_size`
        attribute. If `scale_out` is true, scales segmentation to match the size
        of the `img`.

        Args:
            img: Image to segment
            scale_out: If true, scales segmentation to match the of `img`

        Returns
            segmented image
        """
        # Get size of image and resize for input to model
        img_shape = img.shape
        if img_shape != self.in_size:
            in_img = cv2.resize(
                np.copy(img), dsize=self.in_size, interpolation=cv2.INTER_LINEAR
            )
        else:
            in_img = np.copy(img)

        # Predict segmentation
        mask, activation = self.segmenter.predict(image=in_img)

        # Re scale to input image size if desired
        if scale_out is True and mask.shape != img_shape:
            mask = cv2.resize(
                mask.astype(np.uint8), dsize=img_shape, interpolation=cv2.INTER_NEAREST
            ).astype(mask.dtype)
            activation = cv2.resize(
                activation, dsize=img_shape, interpolation=cv2.INTER_LINEAR
            )

        return mask, activation


def prepare_plot(
    img: np.ndarray,
    gt_mask: np.ndarray,
    act_mask: np.ndarray,
    pred_mask: np.ndarray,
    title=None,
):
    """
    Construct image showing og image, gt mask, activations, and the predicted mask.

    Args:
        img (np.ndarray): Orginal image used for prediction
        gt_mask (np.ndarray): ground truth mask
        act_mask (np.ndarray): activation mask from prediction
        pred_mask (np.ndarray): prediction mask
        title (str): title of image
    """
    # initialize our figure
    figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7), dpi=100)

    # plot the original image, its mask, and the predicted mask
    ax[0, 0].imshow(img, cmap="gray")
    ax[0, 1].imshow(gt_mask)
    pos = ax[1, 0].imshow(act_mask)
    ax[1, 1].imshow(pred_mask)

    # set the titles of the subplots
    ax[0, 0].set_title("Image")
    ax[0, 1].set_title("Original Mask")
    ax[1, 0].set_title("Activation Mask")
    ax[1, 1].set_title("Predicted Mask")

    # set the layout of the figure and display it
    figure.tight_layout()
    figure.colorbar(pos, ax=ax[1, 0])
    if title is not None:
        figure.suptitle(title)
    figure.show()


def make_predictions(
    predicter: Predicter, img_path: str, mask_path: str, save_dir: str = None
):
    """
    Construct image showing og image, gt mask, activations, and the predicted mask.
    Image can be saved to file.

    Args:
        predicter (Predicter): Segemntation predictor interface
        img_path (str): Path to orginal image
        mask_path (str): Path to ground truth mask
        save_dir (str): Path to directory where to save images
    """
    # find the filename and generate the path to ground truth mask
    filename = img_path.split(os.path.sep)[-1]
    basename = os.path.splitext(filename)[0]

    # load the ground-truth segmentation mask in grayscale mode
    # and resize it
    gt_mask = imageio.v2.imread(mask_path)

    # MATCH HOW THE DATASET "READS IN" FILES
    image = imageio.v2.imread(img_path)
    orig = image.copy()
    t0 = time.time()
    pred_mask, act_mask = predicter.predict(image, gt=gt_mask)
    print(f"Predict time: {time.time() - t0}")
    pred_mask = (pred_mask * 255).astype(np.uint8)

    # prepare a plot for visualization
    prepare_plot(orig, gt_mask, act_mask, pred_mask, title=filename)
    if save_dir:
        fname = f"{save_dir}/{basename}_prediction.png"
        plt.savefig(fname)


def main(
    src_dir: str,
    ckpt_name: str = "best.pth",
    deploy_dir: str = None,
    max_deploy: int = 100,
):
    """
    Args:
        src_dir (str): Directory containing the training results (namely the .pth checkpoints)
        ckpt_name (str): Checkpoint name to load like "best.pth"
        deploy_dir (str): If not None, conducts predictions on images in deploy_dir
        max_deploy
    """
    # Compile image and mask paths in test dataset
    print("[INFO] loading up test image paths...")
    test_dict_path = f"{src_dir}/test_dataset_dict.yaml"
    test_dataset_dict = load_yaml(test_dict_path)
    data_dir = test_dataset_dict["kwargs"]["data_dir"]
    if not os.path.isdir(data_dir):
        data_dir = data_dir.replace("../", "")
        if not os.path.isdir(data_dir):
            raise ValueError(f"No directory at: {data_dir}")
    data_type = test_dataset_dict["kwargs"]["data_type"]
    img_paths = list(list_images(f"{data_dir}/{data_type}/images"))
    mask_paths = list(list_images(f"{data_dir}/{data_type}/masks"))

    # Randomly select image/masks to predict
    n_predict = min(5, len(img_paths))
    img_paths, mask_paths = zip(
        *random.sample(list(zip(img_paths, mask_paths)), n_predict)
    )

    # Load the model
    print("Loading model...")
    ckpt_path = f"{src_dir}/{ckpt_name}"
    t0 = time.time()
    predicter = Predicter.from_ckpt(ckpt_path=ckpt_path)
    print(f"Load time: {time.time() - t0}")

    # iterate over the randomly selected test image path
    save_dir = f"{src_dir}/predictions"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print(f"Created prediction directory: {save_dir}")
    for img_path, mask_path in list(zip(img_paths, mask_paths)):
        # make predictions and visualize the results
        make_predictions(predicter, img_path, mask_path, save_dir=save_dir)
    plt.show()

    if deploy_dir:
        deploy_imgs = list(list_images(deploy_dir))
        if len(deploy_imgs) > max_deploy:
            deploy_imgs = deploy_imgs[:max_deploy]
        segmenter = SegmenterWrapper(predicter)
        print(f"Deploying model on {len(deploy_imgs)} images in {deploy_dir}")
        for ind in tqdm(range(len(deploy_imgs))):
            img_path = deploy_imgs[ind]
            img_name = img_path.split(os.path.sep)[-1]
            img = imageio.v2.imread(img_path)
            pred, act = segmenter.predict(img)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            save_deploy_dir = f"{save_dir}/deployment"
            save_path = f"{save_deploy_dir}/{img_name}"
            pred = pred.astype(np.uint8) * 255
            imageio.v2.imwrite(save_path, pred)


if __name__ == "__main__":
    src_dir = "src/models/dl4mia_tissue_unet/results/20230403_102456"
    src_dir = "src/models/dl4mia_tissue_unet/results/20230421_171049"
    ckpt_name = "best.pth"
    deploy_dir = "data/raw/OCT_scans/images"
    max_deploy = 100
    main(
        src_dir=src_dir,
        ckpt_name=ckpt_name,
        # deploy_dir=deploy_dir,
        max_deploy=max_deploy,
    )
