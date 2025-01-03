import pandas as pd
import SimpleITK as sitk
import numpy as np

from imgtools.ops import Resize
from pathlib import Path
from typing import Literal

def find_bbox(mask: sitk.Image) -> np.ndarray:
    """
    Find the bounding box of a given mask image.

    Parameters:
    mask (sitk.Image): The input mask image.

    Returns:
    np.ndarray: The bounding box coordinates as a numpy array, [xstart, xend, ystart, yend, zstart, zend].
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



def find_centroid(mask: sitk.Image) -> np.ndarray:
    """
    Find the centroid of a binary image in image
    coordinates.

    Parameters
    ----------
    mask
        The bimary mask image.

    Returns
    -------
    np.ndarray
        The (x, y, z) coordinates of the centroid
        in image space.
    """
    mask_uint = sitk.Cast(mask, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_uint)
    centroid_coords = stats.GetCentroid(1)
    centroid_idx = mask.TransformPhysicalPointToIndex(centroid_coords)
    return np.asarray(centroid_idx, dtype=np.float32)



def crop_centroid(image: sitk.Image, centroid: tuple, input_size: tuple) -> sitk.Image:
    """
    Crop an image centered on the centroid with specified input dimension. No resizing.

    Parameters
    ----------
    image (sitk.Image)
        The input image from which the bounding box will be cropped.
    bbox_coords (tuple)
        Cordinates of the bounding box.
    input_size (tuple)
        Desired output size of the cropped image. eg. (50, 50, 50)

    Returns
    -------
        sitk.Image: The cropped and resized image.
    """

    min_x = int(centroid[0] - input_size[0] // 2)
    max_x = int(centroid[0] + input_size[0] // 2)
    min_y = int(centroid[1] - input_size[1] // 2)
    max_y = int(centroid[1] + input_size[1] // 2)
    min_z = int(centroid[2] - input_size[2] // 2)
    max_z = int(centroid[2] + input_size[2] // 2)

    img_x, img_y, img_z = image.GetSize()

    if min_x < 0:
        min_x, max_x = 0, input_size[0]
    elif max_x > img_x:
        min_x, max_x = img_x - input_size[0], img_x

    if min_y < 0:
        min_y, max_y = 0, input_size[1]
    elif max_y > img_y:
        min_y, max_y = img_y - input_size[1], img_y

    if min_z < 0:
        min_z, max_z = 0, input_size[2]
    elif max_z > img_z:
        min_z, max_z = img_z - input_size[2], img_z

    return image[min_x:max_x, min_y:max_y, min_z:max_z]



def crop_bbox(image: sitk.Image, bbox_coords: tuple, input_size: tuple) -> sitk.Image:
    """
    Crops a bounding box from the given image and resizes it to the specified input size.
    The if/else statements are used to ensure that the bounding box is not cropped outside the image boundaries.

    Parameters
    ----------
    image (sitk.Image)
        The input image from which the bounding box will be cropped.
    bbox_coords (tuple)
        Cordinates of the bounding box.
    input_size (tuple)
        Desired output size of the cropped image. eg. (50, 50, 50)

    Returns
    -------
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



def crop_maxdim_cube(image: sitk.Image, bbox_coords: tuple, input_size: tuple) -> sitk.Image:
    """
    Crop given image to a cube based on the max dim from a bounding box and resize to specified input size.
    The if/else statements are used to ensure that the bounding box is not cropped outside the image boundaries.

    Parameters
    ----------
    image (sitk.Image)
        The input image from which the bounding box will be cropped.
    bbox_coords (tuple)
        Cordinates of the bounding box.
    input_size (tuple)
        Desired output size of the cropped image. eg. (50, 50, 50)

    Returns
    -------
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



def crop_fmcib_input(image_path:Path, 
                     mask_path:Path, 
                     crop_method:Literal["bbox", "centroid", "cube"]="bbox",
                     input_size:tuple = (50,50,50),
                    ):

    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    match crop_method:
        case "bbox":
            bbox = find_bbox(mask)
            cropped_image = crop_bbox(image, bbox, input_size)
        
        case "centroid":
            centroid = find_centroid(mask)
            cropped_image = crop_centroid(image, centroid, input_size)
        
        case "cube":
            bbox = find_bbox(mask)
            cropped_image = crop_maxdim_cube(image, bbox, input_size)
    
    return cropped_image


def prep_for_fmcib(images_metadata, image_dir, output_path, negative_controls, crop_method, crop_size):

    pass