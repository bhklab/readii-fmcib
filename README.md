This repository is a collection of lightweight scripts to easily extract [fmcib](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/tree/master/fmcib) features. 

Most of the lifting we do is for processing the data, as the extraction has been abstracted out with the easy to use `get_features` function. 
* These scripts will interface BEST with [med-imagetools](https://pypi.org/project/med-imagetools/) outputs.


To run full pipeline:

1. Download the [RADCURE](https://www.cancerimagingarchive.net/collection/radcure/) dataset from The Cancer Imaging Archive. This dataset is under the TCIA Restricted License, so you will need to submit a request form to access the data. Additionally, the dataset is quite large, so this pipeline can be run with just the testing cohort of RADCURE, defined in the clinical data under the `RADCURE-challenge` column as test.

2. Clone the `aerts-example` branch of the `readii-fmcib` repo
    ```bash 
    git clone https://github.com/bhklab/readii-fmcib.git -b aerts-example
    ```

3. In the configuration file under `config/RADCURE.yaml`, set the READII negative control region and types, crop method, and crop size you wish to process for inference with FMCIB.


    > [!WARNING]
    > Do not change any of the other settings in the configuration file for RADCURE. This will impact the pre-processing steps. If you wish to process a different dataset, then `dataset_name`, `patient_id_pattern`, `roi_pattern` and `modalities` should be updated accordingly, but in the same style.


4. Open the `run_readii_prep.ipynb` notebook and follow the instructions to run the steps. Make sure to change `path_to_downloaded_data` to the directory that you downloaded the RADCURE dataset to in step 1.

5. Open the `run_fmcib.ipynb` notebook and follow the instructions to run the steps.

6. Open the `run_analysis.ipynb` notebook and follow the instructions to run the steps.


The final results will appear in a results directory. 


