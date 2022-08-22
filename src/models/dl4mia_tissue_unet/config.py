import os
import torch
from datetime import datetime

class Config():

    DATASET_PATH = os.path.join("data","processed","uncropped")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WORKERS = max([1, int(os.cpu_count()/2)])

    def __init__(self, n_channels:int=1, n_classes:int=1, n_levels:int=3, init_lr:float=5e-4, n_epochs:int=15, batch_size:int=16, save:bool=True):
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

        # define the path to the base output directory
        now = datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        self.BASE_OUTPUT = os.path.join("src","models","dl4mia_tissue_unet","results",date_str)

        # DL4MIA specific settings. Number of workers for the dataloader,
        # the data tranformations, and the type of unet
        self.TRANSFORMS = None
        self.UNET_TYPE = '2d'

        # the training dataset dictionary for the DataLoader
        self.TRAIN_DATASET_DICT = {
                            'name': self.UNET_TYPE,
                            'kwargs': {
                                'data_dir': self.DATASET_PATH,
                                'type': 'train',
                                'size': len(os.listdir(os.path.join(self.DATASET_PATH, 'train', 'images'))),
                                'transform': self.TRANSFORMS,
                            },
                            'batch_size': self.BATCH_SIZE,
                            'workers': self.WORKERS,
        }

        # the validation dataset dictionary for the DataLoader
        self.VAL_DATASET_DICT = {
                            'name': self.UNET_TYPE,
                            'kwargs': {
                                'data_dir': self.DATASET_PATH,
                                'type': 'val',
                                'size': len(os.listdir(os.path.join(self.DATASET_PATH, 'val', 'images'))),
                                'transform': self.TRANSFORMS,
                            },
                            'batch_size': self.BATCH_SIZE,
                            'workers': self.WORKERS,
        }

        # the model dictionary for specifying model structure
        self.MODEL_DICT = {
            'name': 'unet',
            'kwargs': {
                'num_classes': self.NUM_CLASSES,
                'depth': self.NUM_LEVELS,
                'in_channels': self.NUM_CHANNELS,
            }
        }

        # configurations for the workspace and training
        self.CONFIG_DICT = {
            "train_lr":self.INIT_LR,
            "n_epochs": self.NUM_EPOCHS,
            "cuda": self.DEVICE=="cuda",
            "save": save,
            "save_dir": self.BASE_OUTPUT,
            "resume_path": None,
        }
        