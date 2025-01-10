

# %%
import csv
import itertools
import pandas as pd
import SimpleITK as sitk 
import yaml

from imgtools.autopipeline import AutoPipeline
from pathlib import Path
from readii.io.loaders import loadImageDatasetConfig
from readii.negative_controls import applyNegativeControl
from tqdm.notebook import tqdm

import sys; sys.path.append("code")
from process_readii import find_bbox, crop_bbox

# %% [markdown]
# ## Initialize dataset name and negative control settings

# %%
config = loadImageDatasetConfig("RADCURE", Path("config"))

DATASET_NAME = config["dataset_name"]
NEG_CONTROL_REGIONS = config["negative_control_regions"]
NEG_CONTROL_TYPES = config["negative_control_types"]

# %% [markdown]
# ## Set up data directories

# %%
for combo in itertools.product(["rawdata", "procdata"], [DATASET_NAME], ["clinical", "images"]):
    Path(*combo).mkdir(parents=True, exist_ok=True)

raw_images_dir = Path("rawdata", DATASET_NAME , "images")
proc_images_dir = Path("procdata", DATASET_NAME, "images")

# %% [markdown]
# ## Create symlinks for the raw image data for RADCURE test set to the `rawdata/RADCURE/images` directory

# %%
# INPUT THE PATH TO THE RADCURE IMAGE DATA
image_dir_path = Path("/home/bioinf/bhklab/radiomics/radiomics_orcestra/rawdata/RADCURE/images/zipped")


# %%
mit_output_dir = proc_images_dir / "mit_outputs"
modalities = "CT,RTSTRUCT"
roi_yaml_path = raw_images_dir / "mit_roi_names.yaml"
roi_name = "GTV"

# %% [markdown]
# ## Load the CT and RTSTURCT to run through READII to generate negative controls, crop and save the original and negative control images

# %%
# Read in the dataset.csv file made by med-imagetools autopipeline
images_metadata = pd.read_csv(Path(mit_output_dir, "dataset.csv"), index_col=0)

# Set up the output directories for all the READII processed images
# Make main output directory for cropped nifti images
cropped_images_dir = proc_images_dir / "cropped_images"

# Make output directory for the original CT
cropped_original_dir = cropped_images_dir / "original"
Path.mkdir(cropped_original_dir, parents=True, exist_ok=True)

# Make list of negative control types and regions
# Regions refer to what portion of the CT image to apply the negative control to
# Types refer to what will be done with the voxels of the CT image in the region
negative_control_regions = ["full", "roi", "non_roi"]
negative_control_types = ["shuffled", "randomized_sampled"]

# %%
for image_idx in tqdm(images_metadata.index):
    image_idx_metadata = images_metadata.loc[image_idx]

    patient_ID = image_idx_metadata['patient_ID']
    print(f"Processing {patient_ID}...")

    # Load in the CT image output from med-imagetools
    ct_image = sitk.ReadImage(Path(mit_output_dir, image_idx, "CT", "CT.nii.gz"))
    
    # Load in the RTSTRUCT image output from med-imagetools
    roi_image = sitk.ReadImage(Path(mit_output_dir, image_idx, "RTSTRUCT_CT", f"{roi_name}.nii.gz"))

    print("----> CT and RTSTRUCT loaded.")

    # Find the bounding box of the ROI to crop CT image to
    bounding_box = find_bbox(roi_image)

    # Process the original CT image
    # Crop the CT image to the bounding box and resize it to 50x50x50 for input to FMCIB
    cropped_ct_image = crop_bbox(ct_image, bounding_box, (50, 50, 50))
    
    # Save the cropped CT image to the cropped_original_dir
    cropped_output_path = cropped_original_dir / f"{patient_ID}.nii.gz"
    sitk.WriteImage(cropped_ct_image, cropped_output_path)
    print("----> Original CT image cropped to the ROI bounding box, resized, and saved.")

    # Process the negative control CT images
    for negative_control in itertools.product(NEG_CONTROL_TYPES, NEG_CONTROL_REGIONS):
        # Make negative control image using READII
        negative_control_ct_image = applyNegativeControl(ct_image, 
                                                         negativeControlType=negative_control[0], 
                                                         negativeControlRegion=negative_control[1],
                                                         roiMask=roi_image,
                                                         randomSeed=10)
    
        # Crop the negative control CT image to the bounding box and resize it to 50x50x50 for input to FMCIB
        cropped_nc_ct_image = crop_bbox(negative_control_ct_image, bounding_box, (50, 50, 50))

        # Set up the directory to save the cropped negative control CT images
        cropped_nc_dir = cropped_images_dir / f"{negative_control[0]}_{negative_control[1]}"
        Path.mkdir(cropped_nc_dir, parents=True, exist_ok=True)

        # Save the cropped negative control CT image
        cropped_nc_output_path = cropped_nc_dir / f"{patient_ID}.nii.gz"
        sitk.WriteImage(cropped_nc_ct_image, cropped_nc_output_path)
        print(f"----> Negative control {negative_control[0]}_{negative_control[1]} cropped to the ROI bounding box, resized, and saved.")  

# %% [markdown]
# ## Set up expected input file for FMCIB

# %%
for image_type_dir_path in sorted(cropped_images_dir.glob("*")):
    image_type = image_type_dir_path.name
    
    image_type_file_paths = sorted(image_type_dir_path.glob("*.nii.gz"))

    # Create a dataframe with these image paths and all coordinates set to 0
    fmcib_input_df = pd.DataFrame(data = {"image_path": image_type_file_paths})
    fmcib_input_df["coordX"] = 0
    fmcib_input_df["coordY"] = 0
    fmcib_input_df["coordZ"] = 0

    fmcib_input_df.to_csv(Path(proc_images_dir, "fmcib_input", f"fmcib_input_{DATASET_NAME}_{image_type}.csv"), index=False)

# %%



