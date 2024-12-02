from fmcib.run import get_features
from args import parser
import os
from typing import Optional

def infer(feature_file_save_path:str, 
          csv_path:str, 
          weights_path:Optional[str]=None, 
          precropped:Optional[bool]=False):
    """ Function to run FMCIB feature extraction and save out the results to a csv file. 

    Args:
        feature_file_save_path (str): Path to where to save the output file containing the FMCIB features
        csv_path (str): Path to the csv file containing the paths to the input images and coordinates
        weights_path (str): Path to the model weights
        precropped (bool, optional): Whether to use the precropped data. Defaults to False.
    """

    # make sure the output directory exists
    feature_dir_path = os.path.dirname(feature_file_save_path)
    if not os.path.exists(feature_dir_path):
        os.makedirs(feature_dir_path)

    # run FMCIB feature extraction
    feature_df = get_features(csv_path=csv_path, 
                              weights_path=weights_path, 
                              precropped=precropped,
                              spatial_size=(50,50,50))
    
    # clean up the dataframe
    feature_df.drop(columns=['coordX', 'coordY', 'coordZ'], inplace=True)
    
    # save out the FMCIB features to a csv
    feature_df.to_csv(feature_file_save_path, index=False)
    
    # sanity check to STDOUT
    # print(feature_df)
    return feature_df


if __name__ == "__main__":
    params = parser().parse_args()

    infer(feature_file_save_path=params.save_path, 
          csv_path=params.input_path, 
          weights_path=params.weights_path, 
          precropped=params.precropped)