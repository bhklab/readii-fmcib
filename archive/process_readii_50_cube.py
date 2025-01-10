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
    x_size, y_size, z_size = max_x - min_x, max_y - min_y, max_z - min_z

    # get maximum dimension of bounding box
    max_dim = max(max_x - min_x, max_y - min_y, max_z - min_z)
    mean_x = (max_x + min_x) // 2
    mean_y = (max_y + min_y) // 2
    mean_z = (max_z + min_z) // 2

    # define new bounding boxes based on the maximum dimension of ROI bounding box
    min_x = mean_x - max_dim // 2
    max_x = mean_x + max_dim // 2
    min_y = mean_y - max_dim // 2
    max_y = mean_y + max_dim // 2
    min_z = mean_z - max_dim // 2
    max_z = mean_z + max_dim // 2

    img_x, img_y, img_z = image.GetSize()

    if min_x < 0: 
        min_x, max_x = 0, max_dim
    elif max_x > img_x: 
        min_x, max_x = img_x - max_dim, img_x

    if min_y < 0:
        min_y, max_y = 0, max_dim
    elif max_y > img_y: 
        min_y, max_y = img_y - max_dim, img_y

    if min_z < 0:
        min_z, max_z = 0, max_dim
    elif max_z > img_z: 
        min_z, max_z = img_z - max_dim, img_z
    
    if x_size > img_x:
        print("x_size > img_x")
        min_x, max_x = 0, img_x
    if y_size > img_y:
        print("y_size > img_y")
        min_y, max_y = 0, img_y
    if z_size > img_z:
        print("z_size > img_z")
        min_z, max_z = 0, img_z
    
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
    if "PublicDatasets" in ct_path:
        pat_id = ct_path.split("/")[-2]
    else:
        pat_id = ct_path.split("/")[-3]

    crop_dir  = os.path.join(output_path, pat_id)
    crop_path = os.path.join(crop_dir, ct_path.split("/")[-1])
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    if not os.path.exists(crop_path):
        try:
            if os.path.exists(ct_path) and os.path.exists(mask_path):
                # get GTVp centroid
                img  = sitk.ReadImage(ct_path)
                mask = sitk.ReadImage(mask_path)
                assert img.GetSize() == mask.GetSize(), (img.GetSize(), mask.GetSize())

                bbox     = find_bbox(mask)
                img_crop = crop_bbox(img, bbox, (50, 50, 50))

                # save crop and row
                sitk.WriteImage(img_crop, crop_path)
                return crop_path, 0, 0, 0
        except Exception as e:
            # print(pat_id, "ERROR", e)
            return None, 0, 0, 0
    else:
        return crop_path, 0, 0, 0

def process_one_set(negative_control, params):
    img_paths  = sorted(glob.glob(os.path.join(params.base_path, f"*/*/{negative_control}.nii.gz")))
    mask_paths = sorted(glob.glob(os.path.join(params.base_path, "*/*/GTV.nii.gz")))
    print("num of image and masks:", len(img_paths), len(mask_paths))
     
    if not os.path.exists(params.output_path):
        os.makedirs(params.output_path)

    # multi process using ALL available cores
    rows = Parallel(n_jobs=-1)(delayed(get_row)(i, j, params.output_path) for i, j in tqdm(zip(img_paths, mask_paths), total=len(img_paths)))

    # convert to dataframe
    df_new  = pd.DataFrame(columns=["image_path", "coordX", "coordY", "coordZ"])
    for row in rows:
        if row is not None:
            df_new = pd.concat([df_new, pd.DataFrame([row], columns=["image_path", "coordX", "coordY", "coordZ"])], ignore_index=True)
    df_new.to_csv(params.save_path.replace(".csv", f"_{negative_control}.csv"), index=False)
    return df_new

def main():
    print("START START HELLO HELLO")
    params = readii_parser().parse_args()
    # reads a Med-ImageTools processed dataset and outputs a csv file of:
    #   - `image_path`: path of image
    #   - `coordX, coordY, coordZ`: coordinates of the center of GTVp

    print("PARAMS", params)

    process_one_set("original", params)
    for neg_method in ["randomized", "randomized_sampled", "shuffled"]:
        for neg_area in ["full", "non_roi", "roi"]:
            process_one_set(f"{neg_method}_{neg_area}", params)

if __name__ == '__main__':
    main()
