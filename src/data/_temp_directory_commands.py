"""
For translating MATLAB labeled images from image J to match directory and name structure in data/interim/converted

1. Used mimic tree to match directory structure of converted in imagej_converted
2. created masks directory in imagej_converted
3. Copied .tif masks and images from C:/Users/jacob.obrien/envs/DL_segmentation/DL4MIA_Unet/data/download
    to the masks and images dirs in imagej_converted
4. Run move_and_rename_files on the images and masks dirs in imagej_converted
"""
import os
import shutil
from glob import glob
import imageio
import numpy as np
from src.utils.logging import StandardLogger as SL

def _ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]
 
def mimic_tree(input_dir:str, mimic_dir:str):
    """
    Copies directory structure from input directory to mimic directory

    Arguments:
        input_dir: Location of desired directory tree
        mimic_dir: location of where to copy tree structure
    """
    if not os.path.isdir(mimic_dir):
        shutil.copytree(input_dir,
                        mimic_dir,
                        ignore=_ignore_files)

def move_and_rename_files(input_dir:str):
    ''' Move .tif files in input dir to the correct directory 
    
    Moves file with name: NAMEXXXX.tif
    
    Where XXXX denotes the zfilled stack number and NAME should match a directory
    '''
    logger = SL(__name__)
    fnames = os.listdir(input_dir)
    for fname in fnames:
        basename = fname[:-8]
        for root, dirs, files in os.walk(input_dir):
            if basename in dirs:
                src = f"{input_dir}/{fname}"
                new_name = rename_file(fname)
                dst = f"{root}/{basename}/{new_name}"
                shutil.move(src,dst)
                logger.info(f"Moved {src} to {dst}")

def rename_file(in_name):
    ''' Returns file name of in_name = NAMEXXXX.tif as NAME_XXXXXX.tif'''
    root_name = in_name[:-8]
    zfill_ext = in_name[-8:]
    new_name = f"{root_name}_00{zfill_ext}"
    return new_name

def relabel_binary_masks(mask_dir:str, bg_label:int, fg_label:int, ext='.tif'):
    '''
    Relabel binary mask to bg=0, fg = 1. My matlab labels have fg=1, bg = 2

    Arguments:
        mask_dir: Directory containing binary image masks
        bg_label: Number of background in mask
        fg_label: Number of foreground in mask
        ext: File extension of images
    '''
    logger = SL(__name__)
    files = glob(f"{mask_dir}/**/*{ext}", recursive=True)
    for fpath in files:
        img_np = np.asarray(imageio.v2.imread(fpath))
        labels = np.unique(img_np)
        if len(labels) > 2:
            logger.error(f"More than 2 labels ({labels}) for {fpath}.")
        else:
            if bg_label in labels.tolist() and fg_label in labels.tolist():
                fg_mask = img_np == fg_label
                new_mask = np.copy(img_np)
                new_mask[new_mask==bg_label] = 0
                new_mask[fg_mask] = 1
                logger.info(f'Changing labels for {fpath}')
                imageio.imwrite(fpath, new_mask)


if __name__ == '__main__':
    mimic_tree("data/interim/converted", "data/interim/imagej_converted")
    move_and_rename_files("data/interim/imagej_converted/images")
    move_and_rename_files("data/interim/imagej_converted/masks")
    relabel_binary_masks(mask_dir = "C:/Users/jacob.obrien/envs/HT_segmentation/unet_segmentation/data/interim/imagej_converted/masks", bg_label=2, fg_label=1)