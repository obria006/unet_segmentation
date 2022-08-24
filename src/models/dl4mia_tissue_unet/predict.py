# import the necessary packages
import os
import time
import random
import glob
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from src.models.dl4mia_tissue_unet.dl4mia_utils.img_utils import preprocess_image
from src.models.dl4mia_tissue_unet.dl4mia_utils.general import load_yaml
from src.models.dl4mia_tissue_unet.model import UNet

class Predicter():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_model(cls, model_path:str):
        model = torch.load(model_path, map_location=cls.device)
        model.to(cls.device)
        return cls(model)

    @classmethod
    def from_ckpt(cls, ckpt_path:str):
        checkpoint = torch.load(ckpt_path, map_location=cls.device)
        state_dict = checkpoint['model_state_dict']
        model_dict = checkpoint['model_dict']
        print(f"Loaded checkpoint: {ckpt_path}")
        for key in checkpoint:
            if 'state_dict' not in key and 'logger_data' not in key:
                print(f"\t{key} = {checkpoint[key]}")
        model = UNet(**model_dict['kwargs'])
        model.load_state_dict(state_dict, strict=True)
        model.to(cls.device)
        return cls(model)

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, image,gt=None):
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
            p2d = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)  # last dim, second last dim

            im = F.pad(torch.from_numpy(im), p2d, "reflect")

            output = self.model(im)  # B 3 Y X
            # output_softmax = F.softmax(output[0], dim=0)
            output_softmax = torch.sigmoid(output[0])
            seed_map = output_softmax.cpu().detach().numpy()  # Y X
            pred_fg_thresholded = seed_map > 0.5

            if (diff_y - diff_y // 2) != 0:
                pred_fg_thresholded = pred_fg_thresholded[diff_y // 2:-(diff_y - diff_y // 2), ...]
                seed_map = seed_map[diff_y // 2:-(diff_y - diff_y // 2), ...]
            if (diff_x - diff_x // 2) != 0:
                pred_fg_thresholded = pred_fg_thresholded[..., diff_x // 2:-(diff_x - diff_x // 2)]
                seed_map = seed_map[..., diff_x // 2:-(diff_x - diff_x // 2)]

        return pred_fg_thresholded, seed_map

        

def prepare_plot(origImage, origMask, actMask, predMask, title=None):
    # initialize our figure
    figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7), dpi=100)

    # plot the original image, its mask, and the predicted mask
    ax[0,0].imshow(origImage,cmap='gray')
    ax[0,1].imshow(origMask)
    pos = ax[1,0].imshow(actMask)
    ax[1,1].imshow(predMask)

    # set the titles of the subplots
    ax[0,0].set_title("Image")
    ax[0,1].set_title("Original Mask")
    ax[1,0].set_title("Activation Mask")
    ax[1,1].set_title("Predicted Mask")

    # set the layout of the figure and display it
    figure.tight_layout()
    figure.colorbar(pos, ax=ax[1,0])
    if title is not None:
        figure.suptitle(title)
    figure.show()

def make_predictions(predicter, imagePath, maskPath):
    # find the filename and generate the path to ground truth
    # mask
    filename = imagePath.split(os.path.sep)[-1]

    # load the ground-truth segmentation mask in grayscale mode
    # and resize it
    gtMask = cv2.imread(maskPath, 0)

    # MATCH HOW THE DATASET "READS IN" FILES
    image = tifffile.imread(imagePath)
    orig = image.copy()
    t0 = time.time()
    predMask, actMask = predicter.predict(image,gt = gtMask)
    print(f"Predict time: {time.time() - t0}")
    predMask = (predMask*255).astype(np.uint8)
    
    # prepare a plot for visualization
    prepare_plot(orig, gtMask, actMask, predMask,title=filename)

def main(src_dir:str = "src/models/dl4mia_tissue_unet/results/20220824_144635", ckpt_name:str = 'best.pth'):
    print("[INFO] loading up test image paths...")
    test_dict_path = f"{src_dir}/test_dataset_dict.yaml"
    test_dataset_dict = load_yaml(test_dict_path)
    data_dir = test_dataset_dict['kwargs']['data_dir']
    data_type = test_dataset_dict['kwargs']['data_type']
    imagePaths = glob.glob(f"{data_dir}/{data_type}/images/*.tif")
    maskPaths = glob.glob(f"{data_dir}/{data_type}/masks/*.tif")

    imagePaths, maskPaths = zip(*random.sample(list(zip(imagePaths, maskPaths)), 5))

    print("[INFO] loading up model...")
    ckpt_path = f"{src_dir}/{ckpt_name}"
    t0 = time.time()
    P = Predicter.from_ckpt(ckpt_path=ckpt_path)
    print(f"Load time: {time.time() - t0}")

    # iterate over the randomly selected test image paths
    for ind in range(len(imagePaths)):
        imagePath = imagePaths[ind]
        maskPath = maskPaths[ind]
        # make predictions and visualize the results
        make_predictions(P, imagePath, maskPath)
    plt.show()


if __name__ == '__main__':
    main()