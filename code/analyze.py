import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from readii.process.label import setPatientIdAsIndex
from readii.analyze.correlation import getSelfAndCrossCorrelations
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



def sortCorrelationsByClustering(correlation_matrix:pd.DataFrame,
                                 cluster_values:np.array,
                                 cross_correlation:bool = False,
                                 vertical_feature_type:str = "vertical", 
                                 horizontal_feature_type:str = "horizontal",
                                 ) -> pd.DataFrame:
    """Sort a correlation matrix by cluster values indicating which cluster each feature should be in. Both columns and rows will be sorted by this index."""
    
    # Make a copy of the correlation matrix to perform clustering on so we don't alter the original
    clustered_matrix = correlation_matrix.copy(deep=True)

    # Add cluster values as a column of the correlation matrix
    clustered_matrix["cluster_ids"] = cluster_values

    # Sort the rows of the matrix by the cluster values
    clustered_matrix.sort_values(by="cluster_ids", axis=0, inplace=True)

    # Remove the cluster values column
    clustered_matrix.drop(labels="cluster_ids", axis=1, inplace=True)

    # To sort the columns, use the order of the index
    if cross_correlation:
        # Drop the image feature type suffixes from the index and columns so sorting can be done
        clustered_matrix.rename(index = lambda x: x.removesuffix(vertical_feature_type),
                                columns = lambda x: x.removesuffix(horizontal_feature_type),
                                inplace = True)
        
        # Sort the columns using the now resorted index values 
        clustered_matrix = clustered_matrix[clustered_matrix.index.to_list()]
        
        # Add the image type suffixes back to the index and columns
        clustered_matrix = clustered_matrix.add_suffix(vertical_feature_type, axis=0)
        clustered_matrix = clustered_matrix.add_suffix(horizontal_feature_type, axis=1)

    else:
        # Sort the columns using the now resorted index values 
        clustered_matrix = clustered_matrix[clustered_matrix.index.to_list()]

    return clustered_matrix



def makeAllClusterHeatmapPlots(correlation_matrix:pd.DataFrame, 
                               vertical_feature_type:str, 
                               horizontal_feature_type:str,
                               cluster_values:np.array, 
                               save_dir_path:Path,
                               correlation_method:str="pearson", 
                               heatmap_cmap:str="nipy_spectral",
                               overwrite:bool=False
                              ):
    """Sort each of the self and cross-correlation matrix subsections by cluster id valuesm then plot and save the heatmaps for each."""
    # split into self vs self vs other 
    vertical_self_corr, horizontal_self_corr, cross_corr = getSelfAndCrossCorrelations(correlation_matrix,
                                                                                       vertical_feature_name=f"_{vertical_feature_type}",
                                                                                       horizontal_feature_name=f"_{horizontal_feature_type}",
                                                                                      )
    
    clustered_vertical = sortCorrelationsByClustering(vertical_self_corr, cluster_values)
    clustered_horizontal = sortCorrelationsByClustering(horizontal_self_corr, cluster_values)
    clustered_cross_corr = sortCorrelationsByClustering(cross_corr, cluster_values, cross_correlation=True, vertical_feature_type=vertical_feature_type, horizontal_feature_type=horizontal_feature_type)

    print("Plotting vertical feature correlations heatmap...")
    _, vert_heatmap_path = plotSelfCorrHeatmap(clustered_vertical,
                                               vertical_feature_type,
                                               correlation_method,
                                               heatmap_cmap,
                                               save_dir_path,
                                               overwrite)
    print("Plotting horizontal feature correlations heatmap...")
    _, horiz_heatmap_path = plotSelfCorrHeatmap(clustered_horizontal,
                                                horizontal_feature_type,
                                                correlation_method,
                                                heatmap_cmap,
                                                save_dir_path,
                                                overwrite)
    print("Plotting cross feature correlations heatmap...")
    _, cross_heatmap_path = plotCrossCorrHeatmap(clustered_cross_corr,
                                                 vertical_feature_type,
                                                 horizontal_feature_type,
                                                 correlation_method,
                                                 heatmap_cmap,
                                                 save_dir_path,
                                                 overwrite)
    plt.close('all')
    return vert_heatmap_path, horiz_heatmap_path, cross_heatmap_path