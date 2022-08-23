# USAGEconfig 
# python predict.py

# import the necessary packages
from src.models.dl4mia_tissue_unet.config import Config
from src.models.dl4mia_tissue_unet.model import UNet
import glob
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import time
import torch.nn.functional as F

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
            im = self.normalize(image[np.newaxis, ...], axis=(1, 2))  # added new axis already for channel
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

    def normalize(self, pic, pmin=1, pmax=99.8, axis=(1, 2), clip=False, eps=1e-20, dtype=np.float32):
        ''' From dataset class '''
        mi = np.percentile(pic, pmin, axis=axis, keepdims=True)
        ma = np.percentile(pic, pmax, axis=axis, keepdims=True)
        return self.normalize_mi_ma(pic, mi, ma, clip=clip, eps=eps, dtype=dtype)

    def normalize_mi_ma(self, x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
        ''' From dataset class '''
        if dtype is not None:
            x = x.astype(dtype, copy=False)
            mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
            ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
            eps = dtype(eps)

        x = (x - mi) / (ma - mi + eps)

        if clip:
            x = np.clip(x, 0, 1)

        return x
        

def prepare_plot(origImage, origMask, actMask, predMask, title=None):
    # initialize our figure
    figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7), dpi=100)

    # plot the original image, its mask, and the predicted mask
    ax[0,0].imshow(origImage)
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

def make_predictions(predicter, imagePath, config):
    # find the filename and generate the path to ground truth
    # mask
    filename = imagePath.split(os.path.sep)[-1]
    groundTruthPath = os.path.join(f"{config.DATASET_PATH}/test/masks",
        filename)

    # load the ground-truth segmentation mask in grayscale mode
    # and resize it
    gtMask = cv2.imread(groundTruthPath, 0)

    # MATCH HOW THE DATASET "READS IN" FILES
    image = tifffile.imread(imagePath)
    orig = image.copy()
    t0 = time.time()
    predMask, actMask = predicter.predict(image,gt = gtMask)
    print(f"Predict time: {time.time() - t0}")
    predMask = (predMask*255).astype(np.uint8)
    
    # prepare a plot for visualization
    prepare_plot(orig, gtMask, actMask, predMask,title=filename)

def main():
    config = Config()
    # load the image paths in our testing file and randomly select 10
    # image paths
    print("[INFO] loading up test image paths...")
    # imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
    imagePaths = glob.glob(f"{config.DATASET_PATH}/test/images/*.tif")
    # imagePaths = np.random.choice(imagePaths, size=2)

    # load our model from disk and flash it to the current device
    print("[INFO] load up model...")

    model_path = "src/models/dl4mia_tissue_unet/results/20220823_103000_Colab_cpu/model.pth"
    ckpt_train_path = "src/models/dl4mia_tissue_unet/results/20220823_103000_Colab_cpu/trainable_last.pth"
    ckpt_best_path = "src/models/dl4mia_tissue_unet/results/20220823_103000_Colab_cpu/best.pth"
    ckpt_path = "src/models/dl4mia_tissue_unet/results/20220823_103000_Colab_cpu/last.pth"

    # model_path = "src/models/dl4mia_tissue_unet/results/20220823_111400_Colab_gpu/model.pth"
    # ckpt_train_path = "src/models/dl4mia_tissue_unet/results/20220823_111400_Colab_gpu/trainable_last.pth"
    # ckpt_best_path = "src/models/dl4mia_tissue_unet/results/20220823_111400_Colab_gpu/best.pth"
    # ckpt_path = "src/models/dl4mia_tissue_unet/results/20220823_111400_Colab_gpu/last.pth"

    # model_path = "src/models/dl4mia_tissue_unet/results/20220823_111700_Colab_gpu/model.pth"
    # ckpt_train_path = "src/models/dl4mia_tissue_unet/results/20220823_111700_Colab_gpu/trainable_last.pth"
    # ckpt_best_path = "src/models/dl4mia_tissue_unet/results/20220823_111700_Colab_gpu/best.pth"
    # ckpt_path = "src/models/dl4mia_tissue_unet/results/20220823_111700_Colab_gpu/last.pth"

    # model_path = "src/models/dl4mia_tissue_unet/results/20220823_101118/model.pth"
    # ckpt_train_path = "src/models/dl4mia_tissue_unet/results/20220823_101118/trainable_last.pth"
    # ckpt_best_path = "src/models/dl4mia_tissue_unet/results/20220823_101118/best.pth"
    # ckpt_path = "src/models/dl4mia_tissue_unet/results/20220823_101118/last.pth"

    t0 = time.time()
    # P = Predicter.from_model(model_path)
    P = Predicter.from_ckpt(ckpt_path=ckpt_best_path)
    print(f"Load time: {time.time() - t0}")

    # iterate over the randomly selected test image paths
    for path in imagePaths:
        # make predictions and visualize the results
        make_predictions(P, path, config)
    plt.show()





if __name__ == '__main__':
    main()