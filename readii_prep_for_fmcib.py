# %% [markdown]
# # Create READII negative control CTs to run through FMCIB
# 
# This code utilizes the RADCURE dataset from TCIA. This dataset is under the TCIA Restricted License, so users will need to request access prior to running this code.
# 
# We will be using the RADCURE test subset specified in the clinical data sheet. 

# %% [markdown]
# ## Set up pixi environment kernel
# 
# 1. Run the following commands in the terminal:
# 
#     ```bash
#     $ pixi install
# 
#     $ pixi run make_kernel
#     ```
# 
# 2. In the `Select Kernel` menu at the top right of the notebook, select `Jupyter Kernel` as the source. 
# 
# 3. Refresh the options and one called `readii-fmcib` should appear. Select this option.

# %% [markdown]
# ## Imports

# %%
import csv
import itertools
import shutil
import yaml

from imgtools.autopipeline import AutoPipeline
from pathlib import Path
from readii.io.loaders import loadImageDatasetConfig

import sys; sys.path.append("code")
from process_crop import prep_data_for_fmcib

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
# ## Copy the raw image data for RADCURE test set to the `rawdata/RADCURE/images` directory

# %%
# INPUT THE PATH TO THE RADCURE IMAGE DATA
# downloaded_images_dir = Path("/home/bioinf/bhklab/radiomics/radiomics_orcestra/rawdata/RADCURE/images/zipped")

# # %%
# patient_ID_list_file = Path("./rawdata/RADCURE/clinical/col_test_patient_IDs_RADCURE.csv")
# copy_dir_path = Path(raw_images_dir)

# with open(patient_ID_list_file, "r") as f:
#     pat_list = csv.reader(f)
#     for row in pat_list:
#         patient_ID = row[0]

#         existing_patient_image_directory = downloaded_images_dir / patient_ID
#         copy_patient_image_directory = copy_dir_path / patient_ID

#         if copy_patient_image_directory.exists():
#             print(f"Copy of {patient_ID}'s image file already exists.")
#         else:
#             destination = shutil.copytree(existing_patient_image_directory, copy_patient_image_directory, dirs_exist_ok=True)

# %%
# Unzip the files
# find ./rawdata/RADCURE/images -name "*.zip" | while read filename; do unzip -o -d "`dirname "$filename"`" "$filename"; done;

# %% [markdown]
# ## Make ROI yaml file

# %%
roi_name = "GTV"
roi_matches = {roi_name: "GTVp$"}

with open(f"{raw_images_dir}/mit_roi_names.yaml", "w") as outfile:
    yaml.dump(roi_matches, outfile)

# %% [markdown]
# ## Run med-imagetools to get converted nifti files and get dataset summary file

# %%
nifti_dir = proc_images_dir / "converted_niftis"
modalities = "CT,RTSTRUCT"
roi_yaml_path = raw_images_dir / "mit_roi_names.yaml"

# %%
pipeline = AutoPipeline(input_directory=raw_images_dir,
                        output_directory=nifti_dir,
                        modalities=modalities,
                        spacing=(0., 0., 0.),
                        read_yaml_label_names = True,
                        ignore_missing_regex = True,
                        roi_yaml_path = roi_yaml_path,
                        update=True,
                        )

pipeline.run()

# %% [markdown]
# ## Process and crop images and generate expected input file for FMCIB for each image type

# %%
crop_method = "bbox"
fmcib_input_size = (50,50,50)

# Crop and resize the original images
original_image_df = prep_data_for_fmcib(input_image_dir = nifti_dir,
                                        output_dir_path = proc_images_dir,
                                        crop_method = crop_method,
                                        input_size = fmcib_input_size,
                                        roi_name = roi_name,
                                        negative_control_strategy = "original",
                                        )

# Make negative control, then crop and resize images
for negative_control in itertools.product(NEG_CONTROL_TYPES, NEG_CONTROL_REGIONS):
    neg_control_df = prep_data_for_fmcib(input_image_dir = nifti_dir,
                                         output_dir_path = proc_images_dir,
                                         crop_method = crop_method,
                                         input_size = fmcib_input_size,
                                         roi_name = roi_name,
                                         negative_control_strategy = negative_control[0],
                                         negative_control_region = negative_control[1])

# %%



