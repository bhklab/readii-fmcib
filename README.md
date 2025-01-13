This repository is a collection of lightweight scripts to easily extract [fmcib](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/tree/master/fmcib) features. 

Most of the lifting we do is for processing the data, as the extraction has been abstracted out with the easy to use `get_features` function. 
* These scripts will interface BEST with [med-imagetools](https://pypi.org/project/med-imagetools/) outputs.

## Pipeline Setup
Install [Pixi](https://pixi.sh/latest/#installation)


Clone the `aerts-example` branch of the `readii-fmcib` repo

```bash 
git clone https://github.com/bhklab/readii-fmcib.git -b aerts-example
```

## Data Setup
For this example, we have processed a subset of the [RADCURE](https://www.cancerimagingarchive.net/collection/radcure/) dataset.

Download the [RADCURE](https://www.cancerimagingarchive.net/collection/radcure/) dataset from The Cancer Imaging Archive. This dataset is under the TCIA Restricted License, so you will need to submit a request form to access the data. Additionally, the dataset is quite large, so this pipeline can be run with just the testing cohort of RADCURE, defined in the clinical data under the `RADCURE-challenge` column as test and in `rawdata/RADCURE/clinical/col_test_patient_IDs_RADCURE.csv`


## Configuring the Pipeline
In the configuration file under `config/RADCURE.yaml`, set up the following for processing:
+ `negative_control_regions`: What regions to make [READII negative controls](https://github.com/bhklab/readii?tab=readme-ov-file#negative-control-options) for. Can be any of `["full", "roi", "non_roi"]`.
+ `negative_control_types`: What method to use to create [READII negative controls](https://github.com/bhklab/readii?tab=readme-ov-file#negative-control-options). Can be any of `["shuffled", "randomized_sampled"]`. `randomized` was not included for this test as it fails on some of the data.
+ `crop_method`: Method to use to crop images to expected input size for FMCIB inference
+ `crop_size`: Size to crop/resize images to for FMCIB inference 

> [!WARNING]
> Do not change any of the other settings in the configuration file for RADCURE. This will impact the pre-processing steps. If you wish to process a different dataset, then `dataset_name`, `patient_id_pattern`, `roi_pattern` and `modalities` should be updated accordingly, but in the same style.

## Running the Pipeline
1. Open the `run_readii_prep.ipynb` notebook and follow the instructions to run the steps. Make sure to change `path_to_downloaded_data` to the directory that you downloaded the RADCURE dataset to in Data Setup.

2. Open the `run_fmcib.ipynb` notebook and follow the instructions to run the steps.

3. Open the `run_analysis.ipynb` notebook and follow the instructions to run the steps.

The final results will appear in a results directory. 

> [!NOTE]
> Each notebook will need to be run for each `crop_method` setting you select.

