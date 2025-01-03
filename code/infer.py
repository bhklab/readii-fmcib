from fmcib.run import get_features
from args import parser
import os

if __name__ == "__main__":
    params = parser().parse_args()

    # make sure the output directory exists
    dir_path = os.path.dirname(params.save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    neg_controls = [f"{neg_method}_{neg_area}" for neg_method in ["shuffled"] for neg_area in ["full", "non_roi", "roi"]]
    neg_controls += ["original"]
    # run FMCIB feature extraction
    for neg_control in neg_controls:
        feature_df = get_features(params.input_path.replace(".csv", f"_{neg_control}.csv"), 
                                  precropped=params.precropped)
    
        # clean up the dataframe
        feature_df.drop(columns=['coordX', 'coordY', 'coordZ'], inplace=True)
        
        # save the FMCIB features to `params.save_path`
        feature_df.to_csv(params.save_path.replace(".csv", f"_{neg_control}.csv"), index=False)
    
    # sanity check to STDOUT
    print(feature_df)