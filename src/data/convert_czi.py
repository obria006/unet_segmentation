''' Converts czi raw images to more accessible formats'''
import os
import czifile as czi
import imageio
from src.utils.logging import StandardLogger as SL

def czi_to_ext(basename:str, input_dir:str, output_dir:str, overwrite:bool=False, ext:str=".tif"):
    ''' 
    Converts czi file to a new image file with the passed extension.
    Assumes czi image has shape of (video frames, ?, width, height, channels).

    For our purposes, the raw data has a channels=1, and ?=1.

    Arguments:
        basename: filename of the czi file
        input_dir: location of the czi file
        output_dir: desired location of converted image
        overwrite: whether to overwrite the converted image if it already exists
        ext: Desired extension of converted files. 
    '''
    logger = SL(__name__)
    
    # Validate desired converted extension
    assert ext in [".tif", ".jpg", ".jpeg", ".tiff", ".png", ".bmp"]
    # Read in image
    in_name = basename.replace(".czi","")
    czi_path = f"{input_dir}/{basename}"
    img_czi = czi.imread(czi_path)
    shape_dim = len(img_czi.shape)
    if shape_dim != 5:
        return None
    (n, _, width, height, channels) = img_czi.shape
    logger.info(f"Converting image: {czi_path}")
    # Save each stack in czi to directory dictated by ouptut_dir and the czi basename
    for k in range(n):
        np_img = img_czi[k, 0, :, :, 0]
        out_basename = f"{in_name}_{str(k).zfill(6)}{ext}"
        modified_output_dir = f"{output_dir}/{in_name}"
        out_path = f"{modified_output_dir}/{out_basename}"
        if not os.path.isdir(modified_output_dir):
            os.makedirs(modified_output_dir)
            logger.info(f"Created directory: {modified_output_dir}")
        if os.path.exists(out_path) and overwrite is False:
            logger.info(f"Overwrite disabled, so not saving {out_path} beacause it already exists")
            break
        imageio.imwrite(out_path, np_img)
        

def czi_convert_directories(input_dir:str, output_dir:str, overwrite:bool=False, ext:str=".tif"):
    '''
    Walk down directories starting at input_dir and all czi files to the
    desired extension. Mirrors directory structure in output directory

    Arguments:
        input_dir: location of the czi file
        output_dir: desired location of converted image
        overwrite: whether to overwrite the converted image if it already exists
        ext: Desired extension of converted files. 
    '''
    for root, dirs, files in os.walk(input_dir):
        # Check if file is czi
        for fname in files:
            if fname.endswith(".czi"):
                inter_dirs = root.replace(input_dir,'') # form of '\\dir1\\dir2'
                mirrored_output_dir = f"{output_dir}{inter_dirs}"
                czi_to_ext(basename=fname,
                           input_dir=root,
                           output_dir=mirrored_output_dir,
                           overwrite=overwrite, 
                           ext=ext)

if __name__ == '__main__':
    czi_convert_directories("data/raw","data/interim/converted",overwrite=False,ext='.tif')
