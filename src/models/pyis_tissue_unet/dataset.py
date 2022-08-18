# import the necessary packages
from src.models.pyis_tissue_unet import config
from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]

        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        #FIXME READ IN GRAYSCALE
        assert config.NUM_CHANNELS in [1, 3]
        if config.NUM_CHANNELS == 1:
            image = cv2.imread(imagePath,0)
        else:
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.astype("float32") / 255.0
        mask = cv2.imread(self.maskPaths[idx], 0)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)
            
        # return a tuple of the image and its mask
        return (image, mask)