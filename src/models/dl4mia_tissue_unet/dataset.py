import glob
import numpy as np
import os
import random
import tifffile
from skimage.segmentation import find_boundaries
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset

class TwoDimensionalDataset(Dataset):
    def __init__(self, data_dir, type, bg_id=0, size=None, transform=None):
        print('2D `{}` Dataset created.'.format(type))
        # get image and instance list
        image_list = sorted(glob.glob(os.path.join(data_dir, '{}/'.format(type), 'images/*.tif')))
        self.image_list = image_list

        instance_list = sorted(glob.glob(os.path.join(data_dir, '{}/'.format(type), 'masks/*.tif')))
        self.instance_list = instance_list

        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform
        self.type = type
        
    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}
        image = tifffile.imread(self.image_list[index]) # Y X
        mask = tifffile.imread(self.instance_list[index])  # Y X

        sample['image'] = self.normalize(image[np.newaxis, ...], axis=(1, 2))  # added new axis already for channel
        assert self.bg_id in np.unique(mask)
        sample['semantic_mask'] = mask[np.newaxis, ...]
        # if self.transform is None:
        #     sample['semantic_mask'] = mask[np.newaxis, ...].astype(np.float32)
        # else:
        #     sample['semantic_mask'] = mask[np.newaxis, ...]
        sample['im_name'] = self.image_list[index]
        if (self.transform is not None):
            return self.transform(sample)
        else:
            return sample

    def __len__(self):
        return self.real_size if self.size is None else self.size

    @classmethod
    def normalize(cls, pic, pmin=1, pmax=99.8, axis=(1, 2), clip=False, eps=1e-20, dtype=np.float32):
        mi = np.percentile(pic, pmin, axis=axis, keepdims=True)
        ma = np.percentile(pic, pmax, axis=axis, keepdims=True)
        return cls.normalize_mi_ma(pic, mi, ma, clip=clip, eps=eps, dtype=dtype)

    @classmethod
    def normalize_mi_ma(cls, x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
        if dtype is not None:
            x = x.astype(dtype, copy=False)
            mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
            ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
            eps = dtype(eps)

        x = (x - mi) / (ma - mi + eps)

        if clip:
            x = np.clip(x, 0, 1)

        return x


# class TwoDimensionalDataset(Dataset):
#     def __init__(self, data_dir, type, bg_id=0, size=None, transform=None):
#         print('2D `{}` Dataset created.'.format(type))
#         # get image and instance list
#         image_list = sorted(glob.glob(os.path.join(data_dir, '{}/'.format(type), 'images/*.tif')))
#         self.image_list = image_list

#         instance_list = sorted(glob.glob(os.path.join(data_dir, '{}/'.format(type), 'masks/*.tif')))
#         self.instance_list = instance_list

#         self.bg_id = bg_id
#         self.size = size
#         self.real_size = len(self.image_list)
#         self.transform = transform
#         self.type = type
        
#     def __getitem__(self, index):
#         index = index if self.size is None else random.randint(0, self.real_size - 1)
#         sample = {}
#         image = tifffile.imread(self.image_list[index]) # Y X
#         mask = tifffile.imread(self.instance_list[index])  # Y X
#         if self.type=='train' or self.type=='val':
#             sample['image'] = image[np.newaxis, ...] # already normalized while generating crops
#         else:
#             sample['image'] = self.normalize(image[np.newaxis, ...], axis=(1, 2))  # added new axis already for channel
#         class_map, instance_map = self.convert_instance_to_class_ids(mask)
#         sample['semantic_mask'] = class_map[np.newaxis, ...]
#         sample['instance_mask'] = instance_map[np.newaxis, ...]
#         sample['im_name'] = self.image_list[index]
#         if (self.transform is not None):
#             return self.transform(sample)
#         else:
#             return sample

#     def __len__(self):
#         return self.real_size if self.size is None else self.size

    
    
#     @classmethod
#     def convert_instance_to_class_ids(cls, pic, bg_id=0):
#         class_map = convert_to_class_labels(pic)
#         instance_map = np.zeros((pic.shape[0], pic.shape[1]), dtype=np.int16)
#         mask_fg = pic > bg_id
#         if mask_fg.sum() > 0:
#             ids, _, _ = relabel_sequential(pic[mask_fg])
#             instance_map[mask_fg] = ids
#         return class_map, instance_map

#     @classmethod
#     def normalize(cls, pic, pmin=1, pmax=99.8, axis=(1, 2), clip=False, eps=1e-20, dtype=np.float32):
#         mi = np.percentile(pic, pmin, axis=axis, keepdims=True)
#         ma = np.percentile(pic, pmax, axis=axis, keepdims=True)
#         return cls.normalize_mi_ma(pic, mi, ma, clip=clip, eps=eps, dtype=dtype)

#     @classmethod
#     def normalize_mi_ma(cls, x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
#         if dtype is not None:
#             x = x.astype(dtype, copy=False)
#             mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
#             ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
#             eps = dtype(eps)

#         try:
#             import numexpr
#             x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
#         except ImportError:
#             x = (x - mi) / (ma - mi + eps)

#         if clip:
#             x = np.clip(x, 0, 1)

#         return x

    
# def convert_to_class_labels(lbl):
#     b = find_boundaries(lbl, mode='outer')
#     res = (lbl > 0).astype(np.uint8)
#     res[b] = 2
#     return res