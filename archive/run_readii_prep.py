import itertools
import yaml

from imgtools.autopipeline import AutoPipeline
from pathlib import Path
from readii.io.loaders import loadImageDatasetConfig

import sys; sys.path.append("code")
from process_crop import prep_data_for_fmcib


# Settings for processing
config = loadImageDatasetConfig("RADCURE", Path("config"))

DATASET_NAME = config["dataset_name"]

ROI_PATTERN = config["roi_pattern"]
ROI_NAME = list(ROI_PATTERN.keys())[0]

MODALITIES = config['modalities']

NEG_CONTROL_REGIONS = config["negative_control_regions"]
NEG_CONTROL_TYPES = config["negative_control_types"]


# Set up data directories

for combo in itertools.product(["rawdata", "procdata"], [DATASET_NAME], ["clinical", "images"]):
    Path(*combo).mkdir(parents=True, exist_ok=True)

raw_images_dir = Path("rawdata", DATASET_NAME , "images")
proc_images_dir = Path("procdata", DATASET_NAME, "images")

# Directory for output of MIT run
nifti_dir = proc_images_dir / "converted_niftis"


# Make ROI yaml file to use with Med-ImageTools
roi_yaml_path = raw_images_dir / "mit_roi_names.yaml"

if not roi_yaml_path.exists():
    with open(roi_yaml_path, "w") as outfile:
        yaml.dump(ROI_PATTERN, outfile)

# ## Run Med-ImageTools on downloaded data to index and convert to niftis
# Path to the directory containing the patient ID level directories of the images
# path_to_downloaded_data = Path("INSERT_PATH_HERE")
path_to_downloaded_data = Path("rawdata/RADCURE/UNZIPPED_IMAGES_DO_NOT_DELETE/")

pipeline = AutoPipeline(input_directory=path_to_downloaded_data,
                        output_directory=nifti_dir,
                        modalities=MODALITIES,
                        spacing=(0.,0.,0.),
                        read_yaml_label_names = True,
                        ignore_missing_regex = True,
                        roi_yaml_path = roi_yaml_path
                        )

pipeline.run()


# # FMCIB Input Prep
# 
# ### READII Negative Controls
# 
# Create each type of negative control specified by `NEGATIVE_CONTROL_REGIONS` and `NEGATIVE_CONTROL_TYPES` using `READII`
# 
# ### Crop and resize for FMCIB expected input size
# 
# Crop and resize the images to `FMCIB_INPUT_SIZE` (can be set in next cell) with one of three `CROP_METHOD` options:
# 
# 1. `bbox` - Find bounding box based on dimensions of the region of interest (ROI), crop image to these coordinates, resize/resample to `FMCIB_INPUT_SIZE`.
# 2. `cube` - Create cube based on largest ROI bounding box dimension, crop image to these coordinates, resize/resample to `FMCIB_INPUT_SIZE`.
# 3. `centroid` - Create `FMCIB_INPUT_SIZE` cube centered on the ROI centroid, crop to these coordinates.

CROP_METHOD = "bbox" # Options are bbox, cube, centroid
FMCIB_INPUT_SIZE = (50,50,50)


# Crop and resize the original images
print(f"Processing original images...")
original_image_df = prep_data_for_fmcib(input_image_dir = nifti_dir,
                                        output_dir_path = proc_images_dir,
                                        crop_method = CROP_METHOD,
                                        input_size = FMCIB_INPUT_SIZE,
                                        roi_name = ROI_NAME,
                                        negative_control_strategy = "original",
                                        )

# Make negative control, then crop and resize images

for negative_control in itertools.product(NEG_CONTROL_TYPES, NEG_CONTROL_REGIONS):
    print(f"Creating and processing {negative_control} negative control...")
    neg_control_df = prep_data_for_fmcib(input_image_dir = nifti_dir,
                                         output_dir_path = proc_images_dir,
                                         crop_method = CROP_METHOD,
                                         input_size = FMCIB_INPUT_SIZE,
                                         roi_name = ROI_NAME,
                                         negative_control_strategy = negative_control[0],
                                         negative_control_region = negative_control[1],
                                         parallel=False)



