import os
import glob
import numpy as np
import shutil
from src.utils.logging import StandardLogger as SL

def list_labeled_images(data_dir:str,img_dir_name:str, mask_dir_name:str, ext='.tif'):
    ''' Get a dicionary associating a root image directory with number of images in file, mask root, and the filenames
    
    data_dir: direcotry containing images and amsks
    img_dir_name: name of directory containing just images
    mask_dir_name: name of directory containing just masks
    ext: extension of images
    '''
    file_dict= {}
    img_dir = f"{data_dir}/{img_dir_name}"
    for img_root, dirs, files, in os.walk(img_dir):
        file_list = []
        for fname in files:
            mask_root = img_root.replace(f"{data_dir}/{img_dir_name}",f"{data_dir}/{mask_dir_name}")
            if fname.endswith(ext) and os.path.exists(f"{mask_root}/{fname}"):
                file_list.append(fname)
        if len(file_list) > 0:
            file_dict[img_root] = {'num files':len(file_list), 'files':file_list, 'mask root':mask_root} 
    return file_dict


def split_train_val_test(data_dir:str, img_dir_name:str, mask_dir_name:str, split_vals=(70, 20, 10), seed=1000, ext='.tif'):
    '''
    Splits all images in data_dir and its subdirectories according to split vals. Split data located in data_dir/split

    data_dir: dirctory contining images to split
    img_dir_name: name of directory containing just images
    mask_dir_name: name of directory containing just masks
    spilt_vals: 3 tuple that adds to 100 of (train split, val spli, test split)
    seed: random seed
    ext: extension of images
    '''
    logger = SL(__name__)
    assert np.sum(np.array(split_vals)) == 100
    file_dict = list_labeled_images(data_dir=data_dir, img_dir_name=img_dir_name, mask_dir_name=mask_dir_name, ext=ext)
    unique_dirs = list(file_dict.keys())
    n_unique = len(unique_dirs)
    indices = np.arange(n_unique)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_len = int(split_vals[0]/100 * n_unique)
    val_len = int(split_vals[1]/100 * n_unique)
    test_len = int(split_vals[2]/100 * n_unique)
    train_inds = indices[:train_len]
    val_inds = indices[train_len:train_len+val_len] # val draws from front of randmized images
    test_inds = indices[-test_len:] # test draws from back of randmized images
    train_dir = f"{data_dir}/split/train"
    val_dir = f"{data_dir}/split/val"
    test_dir = f"{data_dir}/split/test"
    train_exist = False
    val_exist = False
    test_exist = False 
    for dirname in [img_dir_name, mask_dir_name]:
        if not os.path.isdir(f"{train_dir}/{dirname}"):
            os.makedirs(f"{train_dir}/{dirname}")
        else:
            train_exist = True
        if not os.path.isdir(f"{val_dir}/{dirname}"):
            os.makedirs(f"{val_dir}/{dirname}")
        else:
            val_exist = True
        if not os.path.isdir(f"{test_dir}/{dirname}"):
            os.makedirs(f"{test_dir}/{dirname}")
        else:
            test_exist = True
    if not train_exist and not val_exist and not test_exist:
        logger.info(f'copying data from {len(train_inds)} training instances')
        for train_index in train_inds:
            img_root = unique_dirs[train_index]
            mask_root = file_dict[img_root]['mask root']
            file_list = file_dict[unique_dirs[train_index]]['files']
            for fname in file_list:
                img_src = f"{img_root}/{fname}"
                mask_src = f"{mask_root}/{fname}"
                img_dst = f"{train_dir}/{img_dir_name}/{fname}"
                mask_dst = f"{train_dir}/{mask_dir_name}/{fname}"
                shutil.copy(img_src, img_dst)
                shutil.copy(mask_src, mask_dst)
        logger.info(f'copying data from {len(val_inds)} validation instances')
        for val_index in val_inds:
            img_root = unique_dirs[val_index]
            mask_root = file_dict[img_root]['mask root']
            file_list = file_dict[unique_dirs[val_index]]['files']
            for fname in file_list:
                img_src = f"{img_root}/{fname}"
                mask_src = f"{mask_root}/{fname}"
                img_dst = f"{val_dir}/{img_dir_name}/{fname}"
                mask_dst = f"{val_dir}/{mask_dir_name}/{fname}"
                shutil.copy(img_src, img_dst)
                shutil.copy(mask_src, mask_dst)
        logger.info(f'copying {len(test_inds)} testing instances')
        for test_index in test_inds:
            img_root = unique_dirs[test_index]
            mask_root = file_dict[img_root]['mask root']
            file_list = file_dict[unique_dirs[test_index]]['files']
            for fname in file_list:
                img_src = f"{img_root}/{fname}"
                mask_src = f"{mask_root}/{fname}"
                img_dst = f"{test_dir}/{img_dir_name}/{fname}"
                mask_dst = f"{test_dir}/{mask_dir_name}/{fname}"
                shutil.copy(img_src, img_dst)
                shutil.copy(mask_src, mask_dst)
        
    else:
        logger.info("No data split because test/train/val already exsited.")
        
    
def split_train_test(data_dir, project_name, train_test_name, subset=0.5, by_fraction=True, seed=1000):
    """
        Splits the `train` directory into `test` directory using the partition percentage of `subset`.
        Parameters
        ----------
        data_dir: string
            Indicates the path where the `project` lives.
        project_name: string
            Indicates the name of the sub-folder under the location identified by `data_dir`.
        train_test_name: string
            Indicates the name of the sub-directory under `project-name` which must be split
        subset: float
            Indicates the fraction of data to be reserved for evaluation
        seed: integer
            Allows for the same partition to be used in each experiment.
            Change this if you would like to obtain results with different train-test partitions.
    """

    image_dir = os.path.join(data_dir, project_name, 'download', train_test_name, 'images')
    instance_dir = os.path.join(data_dir, project_name, 'download', train_test_name, 'masks')
    image_names = sorted(glob(os.path.join(image_dir, '*.tif')))
    instance_names = sorted(glob(os.path.join(instance_dir, '*.tif')))
    indices = np.arange(len(image_names))
    np.random.seed(seed)
    np.random.shuffle(indices)
    if (by_fraction):
        subset_len = int(subset * len(image_names))
    else:
        subset_len = int(subset)
    test_indices = indices[:subset_len]
    # make_dirs(data_dir=data_dir, project_name=project_name)
    test_images_exist = False
    test_masks_exist = False
    if not os.path.exists(os.path.join(data_dir, project_name, 'download', 'test', 'images')):
        os.makedirs(os.path.join(data_dir, project_name, 'download', 'test', 'images'))
        print("Created new directory : {}".format(os.path.join(data_dir, project_name, 'download', 'test', 'images')))
    else:
        test_images_exist = True
    if not os.path.exists(os.path.join(data_dir, project_name, 'download', 'test', 'masks')):
        os.makedirs(os.path.join(data_dir, project_name, 'download', 'test', 'masks'))
        print("Created new directory : {}".format(os.path.join(data_dir, project_name, 'download', 'test', 'masks')))
    else:
        test_masks_exist = True
    if not test_images_exist and not test_masks_exist:
        for test_index in test_indices:
            shutil.move(image_names[test_index], os.path.join(data_dir, project_name, 'download', 'test', 'images'))
            shutil.move(instance_names[test_index], os.path.join(data_dir, project_name, 'download', 'test', 'masks'))
        print("Train-Test Images/Masks saved at {}".format(os.path.join(data_dir, project_name, 'download')))
    else:
        print(
            "Train-Test Images/Masks already available at {}".format(os.path.join(data_dir, project_name, 'download')))

if __name__ == '__main__':
    data_dir = "data/interim/imagej_converted"
    split_train_val_test(data_dir=data_dir, img_dir_name='images', mask_dir_name='masks', split_vals=(70,20,10), ext='.tif')