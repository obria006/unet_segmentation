# USAGE
# python train.py
# import the necessary packages
from src.models.pyis_tissue_unet.dataset import SegmentationDataset
from src.models.pyis_tissue_unet.model import UNet
from src.models.pyis_tissue_unet import config
from src.utils import paths
from src.utils.logging import StandardLogger as SL
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision as tv
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import torch
import time
import os

def train():
    logger = SL(__name__)

    trainImages = list(paths.list_images(f"{config.TRAIN_DATASET_PATH}/images"))
    trainMasks = list(paths.list_images(f"{config.TRAIN_DATASET_PATH}/masks"))
    testImages = list(paths.list_images(f"{config.TEST_DATASET_PATH}/images"))
    testMasks = list(paths.list_images(f"{config.TEST_DATASET_PATH}/masks"))
    valImages = list(paths.list_images(f"{config.VAL_DATASET_PATH}/images"))
    valMasks = list(paths.list_images(f"{config.VAL_DATASET_PATH}/masks"))

    

    
    # write the testing image paths to disk so that we can use then
    # when evaluating/testing our model
    logger.info("saving testing image paths...")
    f = open(config.TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()

    # define transformations
    transforms = tv.transforms.Compose([tv.transforms.ToPILImage(),
        tv.transforms.Resize((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        tv.transforms.ToTensor()])

    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
        transforms=transforms)
    valDS = SegmentationDataset(imagePaths=valImages, maskPaths=valMasks,
        transforms=transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
        transforms=transforms)
    logger.info(f"found {len(trainDS)} examples in the training set...")
    logger.info(f"found {len(valDS)} examples in the validation set...")
    logger.info(f"found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=max([1,int(os.cpu_count()/2)]))
    valLoader = DataLoader(valDS, shuffle=False,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=max([1,int(os.cpu_count()/2)]))

    # initialize our UNet model
    unet = UNet().to(config.DEVICE)

    # initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR)

    # calculate steps per epoch for training and test set
    trainSteps = math.ceil(len(trainDS) / config.BATCH_SIZE)
    valSteps = math.ceil(len(valDS) / config.BATCH_SIZE)

    # initialize a dictionary to store training history
    H = {"train_loss": [], "val_loss": []}

    # loop over epochs
    logger.info("training the network...")
    startTime = time.time()
    for e in range(config.NUM_EPOCHS):
        # set the model in training mode
        unet.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # loop over the training set
        logger.info(f"EPOCH {e+1} of {config.NUM_EPOCHS}")
        for (i, (x, y)) in tqdm(enumerate(trainLoader), total=trainSteps):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            # perform a forward pass and calculate the training loss
            pred = unet(x)
            loss = lossFunc(pred, y)

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            # add the loss to the total training loss so far
            totalTrainLoss += loss

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()

            # loop over the validation set
            for (x, y) in valLoader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

                # make the predictions and calculate the validation loss
                pred = unet(x)
                totalValLoss += lossFunc(pred, y)

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())

        # print the model training and validation information
        logger.info("Train loss: {:.6f}, Val loss: {:.4f}".format(
            avgTrainLoss, avgValLoss))

    # display the total time needed to perform the training
    endTime = time.time()
    logger.info("total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(config.PLOT_PATH)
    # serialize the model to disk
    torch.save(unet, config.MODEL_PATH)

if __name__ == "__main__":
    train()