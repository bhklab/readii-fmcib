import itertools
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

from readii.data.label import setPatientIdAsIndex
from readii.analyze.plot_correlation import plotSelfCorrHeatmap, plotCrossCorrHeatmap, plotSelfCorrHistogram, plotCrossCorrHistogram

def prepPatientIndex(feature_df:pd.DataFrame, file_path_column:str, pat_id_pattern:str) -> pd.DataFrame:
    """Extract patient ID from a DataFrame column of file paths based on a provided regex pattern."""
    # Get patient ID from file path name and make a column for this
    pat_ids = feature_df[file_path_column].str.findall(pat_id_pattern)

    feature_df['patient_ID'] = list(itertools.chain.from_iterable(pat_ids))
    
    # Set the patient ID column as the index for the dataframe
    feature_df = setPatientIdAsIndex(feature_df, 'patient_ID')

    # Remove the image_path column
    feature_df.drop(labels="image_path", axis=1, inplace=True)

    return feature_df



def makeAllHeatmapPlots(correlation_matrix:pd.DataFrame, 
                        vertical_feature_type:str, 
                        horizontal_feature_type:str, 
                        save_dir_path:Path,
                        correlation_method:str="pearson", 
                        heatmap_cmap:str="nipy_spectral",
                        overwrite:bool=False,)-> tuple[Path, Path, Path]:
    """"Plot and save correlation heatmaps for the vertical, horizontal, and cross correlation feature sections of a full correlation matrix."""

    print("Plotting vertical feature correlations heatmap...")
    _, vert_heatmap_path = plotSelfCorrHeatmap(correlation_matrix,
                                               vertical_feature_type,
                                               correlation_method,
                                               heatmap_cmap,
                                               save_dir_path,
                                               overwrite)
    print("Plotting horizontal feature correlations heatmap...")
    _, horiz_heatmap_path = plotSelfCorrHeatmap(correlation_matrix,
                                                horizontal_feature_type,
                                                correlation_method,
                                                heatmap_cmap,
                                                save_dir_path,
                                                overwrite)
    print("Plotting cross feature correlations heatmap...")
    _, cross_heatmap_path = plotCrossCorrHeatmap(correlation_matrix,
                                                 vertical_feature_type,
                                                 horizontal_feature_type,
                                                 correlation_method,
                                                 heatmap_cmap,
                                                 save_dir_path,
                                                 overwrite)
    plt.close('all')
    return vert_heatmap_path, horiz_heatmap_path, cross_heatmap_path


def makeAllHistogramPlots(correlation_matrix:pd.DataFrame, 
                        vertical_feature_type:str, 
                        horizontal_feature_type:str, 
                        save_dir_path:Path,
                        correlation_method:str="pearson", 
                        num_bins:int = 450,
                        self_corr_y_max = 250000,
                        cross_corr_y_max = 950000,
                        overwrite:bool=False)-> tuple[Path, Path, Path]:
    """"Plot and save correlation histograms for the vertical, horizontal, and cross correlation feature sections of a full correlation matrix."""

    print("Plotting vertical feature correlations histogram...")
    _, vert_histogram_path = plotSelfCorrHistogram(correlation_matrix,
                                               vertical_feature_type,
                                               correlation_method,
                                               num_bins,
                                               y_upper_bound = self_corr_y_max,
                                               save_dir_path=save_dir_path,
                                               overwrite=overwrite)
    print("Plotting horizontal feature correlations histogram...")
    _, horiz_histogram_path = plotSelfCorrHistogram(correlation_matrix,
                                                horizontal_feature_type,
                                                correlation_method,
                                                y_upper_bound = self_corr_y_max,
                                                save_dir_path=save_dir_path,
                                                overwrite=overwrite)
    print("Plotting cross feature correlations histogram...")
    _, cross_histogram_path = plotCrossCorrHistogram(correlation_matrix,
                                                 vertical_feature_type,
                                                 horizontal_feature_type,
                                                 correlation_method,
                                                 y_upper_bound = cross_corr_y_max,
                                                 save_dir_path=save_dir_path,
                                                 overwrite=overwrite)
    plt.close('all')
    return vert_histogram_path, horiz_histogram_path, cross_histogram_path

