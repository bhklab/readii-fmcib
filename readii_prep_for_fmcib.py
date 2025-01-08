import itertools
import yaml

from imgtools.autopipeline import AutoPipeline
from pathlib import Path
from readii.io.loaders import loadImageDatasetConfig

import sys; sys.path.append("code")
from process_crop import prep_data_for_fmcib

# Load in dataset specific settings for run to set global variables
config = loadImageDatasetConfig("RADCURE", Path("config"))

DATASET_NAME = config["dataset_name"]
NEG_CONTROL_REGIONS = config["negative_control_regions"]
NEG_CONTROL_TYPES = config["negative_control_types"]
CROP_METHOD = "bbox"
FMCIB_INPUT_SIZE = (50,50,50)

# Set up data directories
for combo in itertools.product(["rawdata", "procdata"], [DATASET_NAME], ["clinical", "images"]):
    Path(*combo).mkdir(parents=True, exist_ok=True)

raw_images_dir = Path("rawdata", DATASET_NAME , "images")
proc_images_dir = Path("procdata", DATASET_NAME, "images")


# Make ROI yaml file
roi_name = "GTV"
roi_matches = {roi_name: "GTVp$"}

with open(f"{raw_images_dir}/mit_roi_names.yaml", "w") as outfile:
    yaml.dump(roi_matches, outfile)


# Run med-imagetools to get converted nifti files and get dataset summary file
nifti_dir = proc_images_dir / "converted_niftis"
modalities = "CT,RTSTRUCT"
roi_yaml_path = raw_images_dir / "mit_roi_names.yaml"

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


# Process and crop images and generate expected input file for FMCIB for each image type
crop_method = CROP_METHOD
fmcib_input_size = FMCIB_INPUT_SIZE

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



