''' Functions for cropping images '''
import os
import math
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.utils.logging import StandardLogger as SL

def crop_and_resize_dir(input_dir:str, output_dir:str, ncrops=4, size=(256,256), keep_original:bool=False, overwrite:bool=False, ext:str=".tif"):
    '''
    Walk down directories starting at input_dir crop all images with ext into ncrops
    sub images and save each crop as size.

    Arguments:
        input_dir: root directory to search for images
        output_dir: desired location of to mimic root directory and place crop images
        ncrops: Number of crops to create from the input image
        size: Size to save each crop
        overwrite: whether to overwrite the converted image if it already exists
        keep_original: whether to keep the original image in the output directory without cropping
        ext: File extension of images 
    '''
    logger = SL(__name__)
    # Validate desired converted extension
    assert ext in [".tif", ".jpg", ".jpeg", ".tiff", ".png", ".bmp"]
    for root, dirs, files in os.walk(input_dir):
        # Check if file is czi
        for fname in files:
            if fname.endswith(ext):
                inter_dirs = root.replace(input_dir,'') # form of '\\dir1\\dir2'
                mirrored_output_dir = f"{output_dir}{inter_dirs}"
                logger.info(f'Cropping and resizing {root}/{fname}')
                crop_and_resize_image(basename=fname,
                                    input_dir=root,
                                    output_dir=mirrored_output_dir,
                                    ncrops=ncrops,
                                    size=size,
                                    keep_original=keep_original,
                                    overwrite=overwrite,)

def crop_and_resize_image(basename:str, input_dir:str, output_dir:str, ncrops=4, size=(256,256), keep_original:bool=False, overwrite:bool=False, ext:str=".tif"):
    '''
    For single channel image, crop file into ncrops images with size=size and save as image in 
    output dir as og_name_cropX. Crops files from the corners.

    Arguments:
        basename: name of file to crop
        input_dir: root directory containing images to crop
        output_dir: desired location of to mimic root directory and place crop images
        ncrops: Number of crops to create from the input image
        size: Size to save each crop
        overwrite: whether to overwrite the converted image if it already exists
        keep_original: whether to keep the original image in the output directory without cropping
        ext: File extension of images
    '''
    logger = SL(__name__)
    assert math.sqrt(ncrops)%1 == 0, 'Number of crop images must be square' 
    in_name = basename.replace(ext,'')
    img_path = f"{input_dir}/{basename}"
    img = np.asarray(imageio.v2.imread(img_path))
    assert len(img.shape) == 2
    height, width = img.shape
    n_inc = math.sqrt(ncrops)
    crop_w = width/n_inc
    crop_h = height/n_inc
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")
    
    # Determine the interpolation method for resizing
    # Use nearest interpolation for masks so no additional labels are generated
    is_mask = f"masks/{in_name}" in img_path.replace("\\","/")
    if is_mask:
        inter_flag = cv2.INTER_NEAREST
    # Use area interpolation if downsizing image
    elif np.mean(img.shape) > np.mean(size):
        inter_flag = cv2.INTER_AREA
    # Otherwise use bilinear for zooming image
    else:
        inter_flag = cv2.INTER_LINEAR
    
    if keep_original is True or ncrops <=1:
        rsz_img = cv2.resize(np.copy(img),dsize=size,interpolation=inter_flag)
        if is_mask:
            assert np.all(np.unique(img) == np.unique(rsz_img)), "Interpolation added segmentation labels"
        out_name = f"{in_name}{ext}"
        out_path = f"{output_dir}/{out_name}"
        if os.path.exists(out_path) and overwrite is False:
            logger.info(f"Overwrite disabled, so not saving {out_path} beacause it already exists")
        else:
            imageio.imwrite(out_path, rsz_img)
    if ncrops >1:
        for i in range(ncrops):
            x0 = int(i//n_inc*crop_w)
            x1 = int(x0 + crop_w)
            y0 = int(i%n_inc*crop_h)
            y1 = int(y0 + crop_h)
            img_crop = img[x0:x1, y0:y1]
            rsz_crop = cv2.resize(img_crop,dsize=size,interpolation=inter_flag)
            assert rsz_crop.shape == size
            if is_mask:
                assert np.all(np.unique(img_crop) == np.unique(rsz_crop)), "Interpolation added segmentation labels"
            out_name = f"{in_name}_crop{str(i).zfill(2)}{ext}"
            out_path = f"{output_dir}/{out_name}"
            if os.path.exists(out_path) and overwrite is False:
                logger.info(f"Overwrite disabled, so not saving {out_path} beacause it already exists")
                break
            imageio.imwrite(out_path, rsz_crop)
    inter_dict = {cv2.INTER_AREA:"AREA", cv2.INTER_NEAREST:"NEAREST", cv2.INTER_LINEAR:"LINEAR"}
    logger.info(f"Used interpolation:{inter_dict[inter_flag]}")


if __name__ == '__main__':
    crop_and_resize_dir(input_dir="data/interim/OCT_scans/split/test",output_dir="data/interim/OCT_scans/demo",overwrite=True,keep_original=True, ncrops=1)


