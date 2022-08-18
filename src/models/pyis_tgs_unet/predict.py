# USAGE
# python predict.py

# import the necessary packages
from src.models.pyis_tgs_unet import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

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


def make_predictions(model, imagePath):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (128, 128))
        orig = image.copy()

        # find the filename and generate the path to ground truth
        # mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
            filename)

        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = cv2.imread(groundTruthPath, 0)
        gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_HEIGHT))

        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        actMask = model(image).squeeze()
        actMask = torch.sigmoid(actMask)
        actMask = actMask.cpu().numpy()

        # filter out the weak predictions and convert them to integers
        predMask = (actMask > config.THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)

        # prepare a plot for visualization
        prepare_plot(orig, gtMask, actMask, predMask,title=filename)

# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

# iterate over the randomly selected test image paths
for path in imagePaths:
    # make predictions and visualize the results
    make_predictions(unet, path)
plt.show()