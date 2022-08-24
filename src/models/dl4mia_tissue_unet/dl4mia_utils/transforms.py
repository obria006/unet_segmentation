import sys
import torch
import torchvision.transforms.functional as TF
import numpy as np
import random

class ResizeTransform:
    """ Resize image (H,W) to a new size (H',W')"""

    def __init__(self, size:list):
        """
        Arguments:
            size: Sequence of transformed size as (h, w)
        """
        self.size = size

    def __call__(self, img:torch.Tensor, mask:torch.Tensor):
        return (TF.resize(img, self.size), TF.resize(mask, self.size))

class NumpyToTensor:
    """ Convert numpy image array to Tensor """

    def __init__(self, img_dtype:str='float', mask_dtype:str='short'):
        """
        Arguments:
            dtype: Tensor dtype as torch.FloatTensor or torch.ShortTensor
        """
        if img_dtype not in ('float', 'short'):
            raise NotImplementedError(f"Unsuppported dtype: {img_dtype}. Must be in ('float', 'short').")
        if mask_dtype not in ('float', 'short'):
            raise NotImplementedError(f"Unsuppported dtype: {mask_dtype}. Must be in ('float', 'short').")
        str_to_dtype = {"float":torch.float, "short":torch.short}
        self.img_dtype = str_to_dtype[img_dtype]
        self.mask_dtype = str_to_dtype[mask_dtype]

    def __call__(self, img:np.ndarray, mask:np.ndarray):
        img = torch.from_numpy(img).type(self.img_dtype)
        # img = TF.convert_image_dtype(img, self.img_dtype)
        mask = torch.from_numpy(mask).type(self.mask_dtype)
        # mask = TF.convert_image_dtype(mask, self.mask_dtype)
        return (img, mask)

class RandomFiveCrop:
    """ Returns a random crop from a 5 crop (corners and center). Crops to size
    of half of input size (h/2, w/2) """

    def __init__(self, p:float=0.5):
        """
        Arguments:
            p: probability of applying the crop
        """
        self.p = p

    def __call__(self, img:torch.Tensor, mask:torch.Tensor):
        if random.random() < self.p:
            c, h, w = TF.get_dimensions(img)
            size = (int(h/2), int(w/2))
            img_crops = TF.five_crop(img=img, size=size)
            mask_crops = TF.five_crop(img=mask, size=size)
            ind = random.choice(range(len(img_crops)))
            img = img_crops[ind]
            mask = mask_crops[ind]
        return (img, mask)

class RandomRotationTransform:
    """ Rotate by +/- an angle from the given angles """

    def __init__(self, angles:list=[0, 30, 45, 60, 90]):
        self.angles = angles

    def __call__(self, img:torch.Tensor, mask:torch.Tensor):
        angle = random.choice(self.angles)
        return (TF.rotate(img, angle), TF.rotate(mask, angle))


class RandomFlip:
    """ Randomly horizontally or vertically flips image """
    def __call__(self, img:torch.Tensor, mask:torch.Tensor):
        flip_ind = random.choice(['vflip','hflip',None])
        if flip_ind == 'vflip':
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if flip_ind == 'hflip':
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        return (img, mask)


class RandomJitter:
    """ Randomly change brightness and contrast"""

    def __init__(self, brightness:float=0.25, contrast:float=0.25, p:float=0.33):
        """
        Arguments:
            brightness: brightness factor
            contrast: contrast factor
            p: probability of applying the transform
        """
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img:torch.Tensor, mask:torch.Tensor):
        if random.random() < self.p:
            brightness_factor = 1 + self.brightness*random.randint(-1,1)
            contrast_factor = 1 + self.contrast*random.randint(-1,1)
            img = TF.adjust_brightness(img, brightness_factor)
            img = TF.adjust_contrast(img, contrast_factor)
        return (img, mask)

class Compose:
    """
    Composes several functional transforms together for data
    augmentation/transformations. Composed transforms will be
    applied sequentially to the image, and mask.
    """

    @classmethod
    def from_dict(cls, dict_:dict):
        """
        Instantiate from string keyed dictionary of tranform classes
        and their kwargs. Dictionary keys must be strings that match
        class names in this file. Dictionary values must be a dictionary
        with each key being a string kwarg name for the class and its
        associated arugument value.

        dict_ should take the form:
        {'ClassName1': {'kwarg1': value1, ...}, ...}

        Example:
        ```
            dict_ = {
                'NumpyToTensor': {'img_dtype':'float', 'mask_dtype':'short'},
                'RandomRotationTransform': {'angles':[90]},
                'RandomFlip': {},
                'ResizeTransform': {'size':(128, 128)},
            }
            my_transforms = Compose.from_dict(dict_)
        ```
        
        Args:
            dict_: dictionary of tranform classes and their kwargs
        """
        # init list of transforms
        transforms = []

        # iterate through dict and append transform classes with args to list
        for classname, kwargs in dict_.items():
            tform = getattr(sys.modules[__name__], classname)
            transforms.append(tform(**kwargs))

        return cls(transforms)

    def __init__(self, transforms:list):
        """
        Example:
        ```
            transforms = [
                NumpyToTensor(img_dtype='float', mask_dtype='short'),
                RandomJitter(brightness=0.3, contrast=0.3, p=0.3),
                RandomRotationTransform(angles=[90]),
                RandomFlip(),
                ResizeTransform(size=(128, 128)),
            ]
            Compose(transforms)
        ```

        Arguments:
            transforms: list of tranform classes
        """
        self.transforms = transforms

    def __call__(self, image:np.ndarray, target:np.ndarray):
        """ Apply composed transformations to the image and target """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target