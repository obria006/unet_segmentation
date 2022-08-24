"""
Defines configuration to train U-Net segmentation network
"""

import os
from datetime import datetime
import torch

class Config:
    """
    Sets parameters to train U-Net segmentation network. Defines parameters
    that dictate structure of the Unet, how the unet is to be trained, and how
    to load data and save results.
    """

    # DATASET_PATH = os.path.join("data","processed","uncropped")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WORKERS = max([1, int(os.cpu_count() / 2)])

    def __init__(
        self,
        data_dir: str = "data/processed/uncropped",
        output_dir: str = "src/models/dl4mia_tissue_unet/results",
        n_channels: int = 1,
        n_classes: int = 1,
        n_levels: int = 3,
        in_size: tuple = (128, 128),
        init_lr: float = 5e-4,
        n_epochs: int = 15,
        batch_size: int = 16,
        save: bool = True,
    ):
        """
        Args:
            data_dir: path to the directory containting 'train', 'val', 'test' dirs
            output_dir: path to the directory for where to place training results
            n_channels: number of channels in the input image
            n_classes: number of classes for segmentation
            n_levels: number of max pooling levels in unet
            in_size: size of the input image to model (image will be resized to match)
            init_lr: initial learning rate
            n_epochs: number of epochs to train
            batch_size: size of training/validation batches
            save: if True, save the trained model checkpoints
        """

        # define path to data and training output directories
        self.DATASET_PATH = data_dir
        self.BASE_OUTPUT = output_dir
        now = datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        self.OUTPUT_PATH = f"{self.BASE_OUTPUT}/{date_str}"

        # define the number of channels in the input, number of classes,
        # and number of levels in the U-Net model
        self.NUM_CHANNELS = n_channels
        self.NUM_CLASSES = n_classes
        self.NUM_LEVELS = n_levels

        # initialize learning rate, number of epochs to train for, and the
        # batch size
        self.INIT_LR = init_lr
        self.NUM_EPOCHS = n_epochs
        self.BATCH_SIZE = batch_size

        # transformations for data augmentation
        self.TRANSFORMS = {
            'NumpyToTensor': {'img_dtype':'float', 'mask_dtype':'short'},
            'RandomJitter': {'brightness':0.3, 'contrast':0.3, 'p':0.3},
            'RandomRotationTransform': {'angles':[90]},
            'RandomFlip': {},
            'ResizeTransform': {'size':in_size},
        }

        # Create dataset dictionaries for the dataloader
        self.TRAIN_DATASET_DICT = self._create_dataset_dict(data_type="train", transforms=self.TRANSFORMS)
        self.VAL_DATASET_DICT = self._create_dataset_dict(data_type="val", transforms=self.TRANSFORMS)
        self.TEST_DATASET_DICT = self._create_dataset_dict(data_type="test", transforms=None)

        # create the model dictionary used to intialize/create the unet
        self.MODEL_DICT = self._create_model_dict(name="unet")

        # create configuration for training the model
        self.CONFIG_DICT = self._create_config_dict(save=save)


    def _create_dataset_dict(self, data_type: str, transforms: dict = None):
        """
        Creates dictionary with dataset information to be used by the
        model's Dataset/DataLoader.

        Args:
            data_type: The type of dataset to be created (train, val, test)
        """
        assert data_type in ["train", "val", "test"], "Data must be 'train', 'val', or 'test'"
        dataset_dict = {
            "kwargs": {
                "data_dir": self.DATASET_PATH,
                "data_type": data_type,
                "size": len(
                    os.listdir(os.path.join(self.DATASET_PATH, data_type, "images"))
                ),
                "transform": transforms,
            },
            "batch_size": self.BATCH_SIZE,
            "workers": self.WORKERS,
        }

        return dataset_dict

    def _create_model_dict(self, name: str = "unet"):
        """
        Creates dictionary with model information to be used to create the
        model.

        Args:
            name: The name of the model to be created
        """
        model_dict = {
            "name": name,
            "kwargs": {
                "num_classes": self.NUM_CLASSES,
                "depth": self.NUM_LEVELS,
                "in_channels": self.NUM_CHANNELS,
            },
        }

        return model_dict

    def _create_config_dict(self, save: bool = True):
        """
        Created dictionary with information used to configure model
        training.

        Args:
            save: Whether to save training output checkpoints
        """
        config_dict = {
            "train_lr": self.INIT_LR,
            "n_epochs": self.NUM_EPOCHS,
            "cuda": self.DEVICE == "cuda",
            "save": save,
            "save_dir": self.OUTPUT_PATH,
            "resume_path": None,
        }

        return config_dict