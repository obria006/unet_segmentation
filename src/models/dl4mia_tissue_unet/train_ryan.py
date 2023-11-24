import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import pickle
import glob
import math
import time
import cv2
import sys
import os

import tifffile as tiff

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import List, Callable, Union, Any, TypeVar, Tuple

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn

def ConvBlock(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )

def FinalBlock(in_channels, out_channels):
    return nn.Sequential(
        #ConvBlock(in_channels, in_channels,  kernel_size = 1),
        ConvBlock(in_channels, in_channels,  kernel_size = 1, stride = 1, padding = 0),
        nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
    )

def MiddleBlock(in_channels, out_channels):
    return nn.Sequential(
        ConvBlock(in_channels, out_channels),
        nn.Dropout(p = 0.2),
        ConvBlock(out_channels, out_channels)
    )

class ResidualBlock(nn.Module):
    """ Residual encoder block. """
    def __init__(self, in_channels, feature_maps, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = (2, 2), stride = None)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, feature_maps, kernel_size = (1, 1), stride = stride, bias = False),
            nn.BatchNorm2d(feature_maps)
        )

        self.conv1 = nn.Conv2d(in_channels, feature_maps,  kernel_size = (3, 3), stride = stride, padding = 1, bias = False)
        self.bn1   = nn.BatchNorm2d(feature_maps)

        self.conv2 = nn.Conv2d(feature_maps, feature_maps, kernel_size = (3, 3), stride = 1,      padding = 1, bias = False)
        self.bn2   = nn.BatchNorm2d(feature_maps)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        skip_connection = self.relu(x)

        x = self.maxpool(skip_connection)

        return x, skip_connection

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.Encoder_0 = ResidualBlock(in_channels = in_channels, feature_maps = 32)
        self.Encoder_1 = ResidualBlock(in_channels = 32,          feature_maps = 64)
        self.Encoder_2 = ResidualBlock(in_channels = 64,          feature_maps = 128)

        self.Middle = MiddleBlock(in_channels = 128, out_channels = 256)

    def forward(self, x):
        x, x0 = self.Encoder_0(x)
        x, x1 = self.Encoder_1(x)
        x, x2 = self.Encoder_2(x)

        x3 = self.Middle(x)

        return [x0, x1, x2, x3]
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.up = nn.Upsample(scale_factor = 2)

        self.conv_block_0 = ConvBlock(in_channels + out_channels, out_channels)
        self.conv_block_1 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)

        x = torch.cat((x, skip_connection), 1)

        x = self.conv_block_0(x)
        x = self.conv_block_1(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()

        self.decoder_0 = DecoderBlock(256, 128) 
        self.decoder_1 = DecoderBlock(128, 64)
        self.decoder_2 = DecoderBlock(64, 32)
        
        self.FinalBlock = FinalBlock(in_channels = 32, out_channels = out_channels)
        
    def forward(self, x0, x1, x2, x3):
        x = self.decoder_0(x3, x2)
        x = self.decoder_1(x,  x1)
        x = self.decoder_2(x,  x0)
        
        x = self.FinalBlock(x)

        return x 

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels = 1):
        super(UNet, self).__init__()

        self.backbone = Encoder(in_channels)
        self.head     = Decoder(out_channels)
        
    def forward(self, x):
        x0, x1, x2, x3 = self.backbone(x)

        x = self.head(x0, x1, x2, x3)

        return x
    
class EndToEndUNet(UNet):
    """End-to-end model that handles pre and post processing for the UNet"""

    def __init__(self, in_channels, out_channels=1, pre:Callable=None, post:Callable=None):
        """
        Args:
            in_channels (int): Number of channels in input image
            out_channels (int): Number of classes (channels in output image)
            pre (Callable): Pre-processing function that operates on Tensors
                of same shape as expected by the UNet input
            post (Callable): Post-processing function that operates on Tensors
                of same shape as expected by the UNet output
        """
        super().__init__(in_channels, out_channels)
        self._preprocess = pre
        self._postprocess = post

    def forward(self, x):
        if self._preprocess:
            x = self._preprocess(x)

        x = super().forward(x)

        if self._postprocess:
            x = self._postprocess(x)

        return x
    
class DiceLossBinary(nn.Module):
    """
    Binary dice loss for semantic segmentation. This code is reworked from
    these GitHub repos:
        - https://github.com/qubvel/segmentation_models.pytorch
        - https://github.com/BloodAxe/pytorch-toolbelt

    """
    def __init__(self, from_logits = True, log_loss = False, smooth = 0.0, eps = 1e-7):
        """
        Args:
            log_loss (bool): If True, the loss is computed as `-log(dice_coeff)`,
                otherwise `1 - dice_coeff`.

            from_logits (bool): If True, assumes y_pred are raw logits.
            smooth (float): Smoothness constant for dice coefficient.
            eps (float): For numerical stability to avoid zero division error.

        """
        super(DiceLossBinary, self).__init__()

        self.from_logits = from_logits
        self.log_loss = log_loss
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Of shape (B, C, H, W).
            y_true (torch.Tensor): Of shape (B, C, H, W).

        Returns:
            torch.Tensor: The loss.

        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = F.logsigmoid(y_pred).exp()

        bs   = y_true.size(0)
        dims = (0, 2)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        scores = self._compute_score(
            y_pred, 
            y_true.type_as(y_pred), 
            smooth = self.smooth, 
            eps = self.eps, 
            dims = dims
        )

        if self.log_loss: loss = -torch.log(scores.clamp_min(self.eps))
        else:             loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss = loss * mask.to(loss.dtype)

        return self._reduction(loss)

    def _reduction(self, loss):
        return loss.mean()

    def _compute_score(self, y_pred, y_true, smooth = 0.0, eps = 1e-7, dims = ()):
        assert y_pred.size() == y_true.size()

        intersection = torch.sum(y_pred * y_true, dim = dims)
        cardinality = torch.sum(y_pred + y_true, dim = dims)

        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

        return dice_score


class TverskyLossBinary(DiceLossBinary):
    """
    This code is reworked from this GitHub repo:
        - https://github.com/qubvel/segmentation_models.pytorch
        - https://github.com/BloodAxe/pytorch-toolbelt

    Tversky loss for semantic segmentation. Notice this class inherits
    `DiceLoss` and adds a weight to the value of each TP and FP given by
    constants alpha and beta. With alpha == beta == 0.5, this loss becomes
    equal to the Dice loss. `y_pred` and `y_true` must be torch tensors of
    shape (B, C, H, W).

    """
    def __init__(self, from_logits = True, log_loss = False, smooth = 0.0, eps = 1e-7, 
                 alpha = 0.5, beta = 0.5, gamma = 1.0):
        """
        Args:
            from_logits (bool): If True, assumes y_pred are raw logits.
            log_loss (bool): If True, the loss is computed as `-log(dice_coeff)`,
                otherwise `1 - dice_coeff`.

            smooth (float): Smoothness constant for dice coefficient.
            eps (float): For numerical stability to avoid zero division error.
            alpha (float): Weight constant that penalize model for FPs.
            beta (float): Weight constant that penalize model for FNs.
            gamma (float): Constant that squares the error function. Defaults to `1.0`.

        """
        super(TverskyLossBinary, self).__init__(from_logits, log_loss, smooth, eps)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reduction(self, loss):
        return loss.mean() ** self.gamma

    def _compute_score(self, y_pred, y_true, smooth = 0.0, eps = 1e-7, dims = ()):
        assert y_pred.size() == y_true.size()

        intersection = torch.sum(y_pred * y_true, dim = dims)  
        fp = torch.sum(y_pred * (1.0 - y_true), dim = dims)
        fn = torch.sum((1 - y_pred) * y_true, dim = dims)

        tversky_score = (intersection + smooth) / (intersection + self.alpha * fp + self.beta * fn + smooth).clamp_min(eps)

        return tversky_score
    
def preprocess_image(image, max_value = 65_533):
    """ Normalize values between -1 and 1. """
    return ((image / max_value) - 1) * 2

def preprocess_0to1(image, max_value = 2** 16 -1):
    """ Normalize values 0-1 based on `max_value`"""
    return image/max_value

class TwoDimensionalDataset(Dataset):
    """
    Creates a 2D dataset to be used with torch DataLoader in training the
    Unet.
    """

    def __init__(
        self,
        data_dir: str,
        data_type: str,
        bg_id: int = 0,
        size: int = None,
        transform=None,
    ):
        """
        Args:
            data_dir: directory containing data (with 'images' and 'masks' subdirectories)
            data_type: type of the data (like 'train', 'val', 'test')
            bg_id: value of background pixels in segmentation masks
            size: size of the data?
            transform: tranforms to apply for augmentation
        """
        # get list of dataset images and masks
        image_list = sorted(glob.glob(f"{data_dir}/{data_type}/images/*.tif"))
        self.image_list = image_list
        instance_list = sorted(glob.glob(f"{data_dir}/{data_type}/masks/*.tif"))
        self.instance_list = instance_list

        # set dataset attributes
        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.data_type = data_type
        self.transform = transform

        print(f"2D `{data_type}` Dataset created.")

    def __getitem__(self, index: int):
        """
        Returns dictionary of image, mask, and filename corresponding to the index
        in image_list and instance_list.

        Args:
            index: index in image_list and instance_list
        """
        # initialize dictionary output
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # read image and mask
        image = tiff.imread(self.image_list[index])  # Y X
        mask = tiff.imread(self.instance_list[index])  # Y X

        image = preprocess_0to1(image)
        
        if self.transform is not None:
            transformed = self.transform(image, mask)
            image = transformed["image"]
            mask  = transformed["mask"]

        return image, mask

        # # training in torch expects image to be tensor valued 0-1 and have the shape
        # # (B, C, Z, Y, X) B = batch, C = channel, Z = z-dimension, Y = height, X = width
        # # So normalize image 0-1 and add axis to (Y, X) image to turn into a (C, Y, X)
        # # ndarray. Likewise add axis to (Y, X) mask to turn into a (C, Y, X) mask.
        # # Include these modified ndarrays in the output dictionary with image filename.
        # sample["image"] = preprocess_image(image)
        # sample["semantic_mask"] = preprocess_mask(mask)
        # sample["im_name"] = self.image_list[index]

        # # if data augmentation/transforms desired, then transform the image and mask before output
        # if self.transform is not None:
        #     img_new, mask_new = self.transform(sample["image"], sample["semantic_mask"])
        #     sample["image"] = img_new
        #     sample["semantic_mask"] = mask_new

        # return sample

    def __len__(self):
        return self.real_size if self.size is None else self.size

class ZahraDataset(Dataset):
    def __init__(self,         
        data_dir: str,
        preprocess:Callable = None,
        transformations = None,
        remove_ids: List[int] = [], 
        image_path: str = "images/*.tif", 
        mask_path: str = "masks/*.tif", 
    ): 
        self.images_dir = sorted(glob.glob(os.path.join(data_dir, image_path)))
        self.masks_dir  = sorted(glob.glob(os.path.join(data_dir, mask_path)))
        self.transformations = transformations
        
        self.masks = []
        self.images = []
        for i, (image_dir, mask_dir) in enumerate(zip(self.images_dir, self.masks_dir)):
            if i not in remove_ids:
                image = tiff.imread(image_dir).astype(np.float32)[:, :, None]
                mask  = tiff.imread(mask_dir).astype(np.float32)[:, :, None]
                
                # Currently I am removing the skin all together.
                mask[mask == 2] = 0

                if preprocess is not None:
                    image = preprocess(image)
                
                # Also, do to the class imbalance, anytime there is
                # skin present I double it in the training set.
                if np.any(mask == 2):
                    self.images.append(image)
                    self.masks.append(mask)
                    
                self.images.append(image)
                self.masks.append(mask)
        print(f"{len(self.images)} in dataset")

    def __getitem__(self, idx):
        print("getting items")
        mask  = self.masks[idx]
        image = self.images[idx]
        print("got image and mask")
        if self.transformations is not None:
            transformed = self.transformations(image = image, mask = mask)
            image = transformed["image"]
            mask  = transformed["mask"]
                    
        return image, mask

    def __len__(self):
        return len(self.images)
    
from albumentations.core.transforms_interface import DualTransform
import random

def shift_image_vertical(img, amount):
    return np.roll(img, amount, axis = 0)

class RandomVerticalShift(DualTransform):
    """Randomly shift the data horizontally.

    Args:
        max_shift (int): The maximum amount of shift possible.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
        
    """
    def __init__(self, max_shift = [0, 0], always_apply = False, p = 0.5):
        super().__init__(always_apply, p)
        self.max_shift = max_shift

    def apply(self, img, **params):
        return shift_image_vertical(img, params["shift"])
    
    def get_params(self):
        return {"shift": np.random.randint(self.max_shift[0], self.max_shift[1])}
    
    def get_transform_init_args_names(self):
        return ("max_shift")

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transformations = A.Compose([
    RandomVerticalShift(p = 0.5, max_shift = [-12, 256]),
    A.ElasticTransform(p = 0.2, alpha_affine = 25, border_mode = cv2.BORDER_REFLECT),
    A.VerticalFlip(p = 1.0 / 3.0),
    A.HorizontalFlip(p = 1.0 / 3.0),
    ToTensorV2(transpose_mask = True),])

class cfg:
    BATCH_SIZE = 1
    EPOCHS = 100
    LOG_EVERY = 3
    DEVICE = "cuda"

    # setting device on GPU if available, else CPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

model = UNet(1, 1).to(cfg.DEVICE)

binary_tversky = TverskyLossBinary(from_logits=True)
# loss_function_0 = TverskyLoss(from_logits = True)
# loss_function_1 = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.0003)

# Training and validation directories
data_dir = "data/processed/OCT_scans_original_and_20230419_512x512"
train_dir = f"{data_dir}/train"
val_dir = f"{data_dir}/val"

train_dataset = TwoDimensionalDataset(
    data_dir=data_dir,
    data_type="train",
    bg_id=0,
    transform=train_transformations,
)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=12,
    pin_memory=True if cfg.DEVICE == "cuda" else False,
)

for i, stuff in enumerate(train_dataloader):
    print(f"ITER: {i}")
# train_dataset = ZahraDataset(
#     train_dir,
#     preprocess=preprocess_0to1,
#     transformations = train_transformations, 
#     image_path = "images/*.tif", 
#     mask_path = "masks/*.tif",
# )
# train_dataloader = DataLoader(
#     train_dataset,
#     batch_size = cfg.BATCH_SIZE,
#     pin_memory = True,
#     drop_last = True,
#     shuffle = True,
#     num_workers = 1
# )
print("Dataset/Dataloader created")

ITERS = len(train_dataloader)
delimiter = " | "

model.train(True)
total_time = time.time()
print("Model in training mode")
for epoch in range(1, cfg.EPOCHS + 1):
    epoch_time = time.time()
    print(f"Epoch: {epoch}")

    for step, (images, masks) in enumerate(train_dataloader):
        step_time = time.time()
        print(f"Image: {step}")
