import os
import glob
import pandas as pd
import SimpleITK as sitk
import numpy as np

from args import readii_parser
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from imgtools.ops import Resize

def find_bbox(mask: sitk.Image) -> np.ndarray:
    """
    Finds the bounding box of a given mask image.

    Parameters:
    mask (sitk.Image): The input mask image.

    Returns:
    np.ndarray: The bounding box coordinates as a numpy array.
    """
    mask_uint = sitk.Cast(mask, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_uint)
    xstart, ystart, zstart, xsize, ysize, zsize = stats.GetBoundingBox(1)
    
    # Prevent the following ITK Error from SmoothingRecursiveGaussianImageFilter: 
    # The number of pixels along dimension 2 is less than 4. This filter requires a minimum of four pixels along the dimension to be processed.
    if xsize < 4:
        xsize = 4
    if ysize < 4:
        ysize = 4
    if zsize < 4:
        zsize = 4

    xend, yend, zend = xstart + xsize, ystart + ysize, zstart + zsize
    return xstart, xend, ystart, yend, zstart, zend

def crop_bbox(image: sitk.Image, bbox_coords: tuple, input_size: tuple) -> sitk.Image:
    """
    Crops a bounding box from the given image and resizes it to the specified input size.
    The if/else statements are used to ensure that the bounding box is not cropped outside the image boundaries.

    Args:
        image (sitk.Image): The input image from which the bounding box will be cropped.
        bbox_coords (tuple): Cordinates of the bounding box.
        input_size (tuple): Desired output size of the cropped image.
    Returns:
        sitk.Image: The cropped and resized image.
    """
    min_x, max_x, min_y, max_y, min_z, max_z = bbox_coords
    img_x, img_y, img_z = image.GetSize()

    if min_x < 0: 
        min_x, max_x = 0, input_size[0]
    elif max_x > img_x: # input_size[0]:
        min_x, max_x = img_x - input_size[0], img_x

    if min_y < 0:
        min_y, max_y = 0, input_size[1]
    elif max_y > img_y: # input_size[1]:
        min_y, max_y = img_y - input_size[1], img_y

    if min_z < 0:
        min_z, max_z = 0, input_size[2]
    elif max_z > img_z: # input_size[2]:
        min_z, max_z = img_z - input_size[2], img_z
    
    img_crop = image[min_x:max_x, min_y:max_y, min_z:max_z]
    img_crop = Resize(input_size)(img_crop)
    return img_crop

def get_row(ct_path, mask_path, output_path):
    """
    Processes a CT image and its corresponding mask to extract a cropped region and save it.

    Args:
        ct_path (str): Path to the CT image file.
        mask_path (str): Path to the mask image file.
        output_path (str): Directory where the cropped image will be saved.

    Returns:
        tuple: Path to the saved cropped image and coordinates (0, 0, 0) if successful, otherwise None.
    """
    if os.path.exists(ct_path) and os.path.exists(mask_path):
        # get GTVp centroid
        img  = sitk.ReadImage(ct_path)
        mask = sitk.ReadImage(mask_path)

        if "PublicDatasets" in ct_path:
            pat_id = ct_path.split("/")[-2]
        else:
            pat_id = ct_path.split("/")[-3]
        
        assert img.GetSize() == mask.GetSize(), (img.GetSize(), mask.GetSize())

        bbox     = find_bbox(mask)
        img_crop = crop_bbox(img, bbox, (50, 50, 50))

        # save crop and row
        crop_path = os.path.join(output_path, f"{pat_id}.nii.gz")
        sitk.WriteImage(img_crop, crop_path)
        return crop_path, 0, 0, 0

def main():
    params = readii_parser().parse_args()
    # reads a Med-ImageTools processed dataset and outputs a csv file of:
    #   - `image_path`: path of image
    #   - `coordX, coordY, coordZ`: coordinates of the center of GTVp

    img_paths  = sorted(glob.glob(os.path.join(params.base_path, "*", params.input_format)))
    mask_paths = sorted(glob.glob(os.path.join(params.base_path, "*", params.mask_format)))
    
    if not os.path.exists(params.output_path):
        os.makedirs(params.output_path)

    # multi process using ALL available cores
    rows = Parallel(n_jobs=-1)(delayed(get_row)(i, j, params.output_path) for i, j in tqdm(zip(img_paths, mask_paths)))

    # convert to dataframe
    df_new  = pd.DataFrame(columns=["image_path", "coordX", "coordY", "coordZ"])
    for row in rows:
        if row is not None:
            df_new = pd.concat([df_new, pd.DataFrame([row], columns=["image_path", "coordX", "coordY", "coordZ"])], ignore_index=True)
    df_new.to_csv(params.save_path, index=False)

if __name__ == '__main__':
    main()
