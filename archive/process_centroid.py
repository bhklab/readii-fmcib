import os, glob
import pandas as pd
from args import readii_parser
from joblib import Parallel, delayed
import SimpleITK as sitk
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from imgtools.ops import Resize


def find_centroid(mask: sitk.Image) -> np.ndarray:
    """Find the centroid of a binary image in image
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
    blank = np.zeros(input_size)

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

                centroid = find_centroid(mask)
                img_crop = crop_centroid(img, centroid, (50, 50, 50))

                # save crop and row
                sitk.WriteImage(img_crop, crop_path)
                return crop_path, 0, 0, 0
        except:
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


if __name__ == "__main__":
    main()
