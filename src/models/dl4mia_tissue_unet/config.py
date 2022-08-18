import os
import torch
               
# base path of the dataset)
DATASET_PATH = os.path.join("data","processed","uncropped")

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 5e-4
NUM_EPOCHS = 6 # was 40
BATCH_SIZE = 8 # was 64

# define the path to the base output directory
BASE_OUTPUT = os.path.join("src","models","dl4mia_tissue_unet","results")
if not os.path.isdir(BASE_OUTPUT):
    os.makedirs(BASE_OUTPUT)

# DL4MIA specific settings. Number of workers for the dataloader,
# the data tranformations, and the type of unet
WORKERS = max([1, int(os.cpu_count()/2)])
TRANSFORMS = None
UNET_TYPE = '2d'

# the training dataset dictionary for the DataLoader
TRAIN_DATASET_DICT = {
                    'name': UNET_TYPE,
                    'kwargs': {
                        'data_dir': DATASET_PATH,
                        'type': 'train',
                        'size': len(os.listdir(os.path.join(DATASET_PATH, 'train', 'images'))),
                        'transform': TRANSFORMS,
                    },
                    'batch_size': BATCH_SIZE,
                    'workers': WORKERS,
}

# the validation dataset dictionary for the DataLoader
VAL_DATASET_DICT = {
                    'name': UNET_TYPE,
                    'kwargs': {
                        'data_dir': DATASET_PATH,
                        'type': 'val',
                        'size': len(os.listdir(os.path.join(DATASET_PATH, 'val', 'images'))),
                        'transform': TRANSFORMS,
                    },
                    'batch_size': BATCH_SIZE,
                    'workers': WORKERS,
}

# the model dictionary for specifying model structure
MODEL_DICT = {
    'name': 'unet',
    'kwargs': {
        'num_classes': NUM_CLASSES,
        'depth': NUM_LEVELS,
        'in_channels': NUM_CHANNELS,
    }
}

# configurations for the workspace and training
CONFIG_DICT = {
    "train_lr":INIT_LR,
    "n_epochs": NUM_EPOCHS,
    "cuda": DEVICE=="cuda",
    "save": True,
    "save_dir": BASE_OUTPUT,
    "resume_path": None,
}
