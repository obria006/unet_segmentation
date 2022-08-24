""" Defines dataset class to be used with torch DataLoader"""
import random
import glob
import tifffile
from torch.utils.data import Dataset
from src.models.dl4mia_tissue_unet.dl4mia_utils.img_utils import preprocess_image, preprocess_mask
from src.models.dl4mia_tissue_unet.dl4mia_utils.transforms import Compose


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
        transform: dict=None,
    ):
        """
        Args:
            data_dir: directory containing data (with 'images' and 'masks' subdirectories)
            data_type: type of the data (like 'train', 'val', 'test')
            bg_id: value of background pixels in segmentation masks
            size: size of the data?
            transform: dictionary of tranforms to apply for augmentation
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

        # Convert string keyed transform dictionary to Compose transformation object
        if transform is not None:
            self.transform = Compose.from_dict(transform)
        else:
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
        image = tifffile.imread(self.image_list[index])  # Y X
        mask = tifffile.imread(self.instance_list[index])  # Y X

        # training in torch expects image to be tensor valued 0-1 and have the shape
        # (B, C, Z, Y, X) B = batch, C = channel, Z = z-dimension, Y = height, X = width
        # So normalize image 0-1 and add axis to (Y, X) image to turn into a (C, Y, X)
        # ndarray. Likewise add axis to (Y, X) mask to turn into a (C, Y, X) mask.
        # Include these modified ndarrays in the output dictionary with image filename.
        sample["image"] = preprocess_image(image)
        sample["semantic_mask"] = preprocess_mask(mask)
        sample["im_name"] = self.image_list[index]

        # if data augmentation/transforms desired, then transform the image and mask before output
        if self.transform is not None:
            img_new, mask_new = self.transform(sample["image"], sample["semantic_mask"])
            sample["image"] = img_new
            sample["semantic_mask"] = mask_new

        return sample

    def __len__(self):
        return self.real_size if self.size is None else self.size
