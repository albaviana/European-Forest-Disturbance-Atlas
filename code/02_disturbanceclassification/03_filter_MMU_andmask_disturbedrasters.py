#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Script to filter isolated disturbed pixels and mask according to forest land use layer (see code in section 01)

import os
import rasterio
import numpy as np
import time
from multiprocessing import Pool
from skimage.measure import label
from concurrent.futures import ThreadPoolExecutor

start_time = time.time()
def eliminate_small_patches(input_path, output_path, min_patch_area=3, nodata_value=np.nan):
    # Open the input raster file
    with rasterio.open(input_path) as src:
        # Read the raster data into a numpy array
        raster_data = src.read(1)
        nodata = src.nodata
        # Create a binary mask where pixels equal to 1 are considered valid
        mask = (raster_data == 1).astype(bool)

        # Use skimage label function to identify connected components
        labels, num_labels = label(mask, connectivity=1, return_num=True)

        # Get the size of each connected component
        component_sizes = np.bincount(labels.ravel())

        # Create a mask to identify patches smaller than min_patch_area pixels
        small_patches_mask = np.isin(labels, np.where(component_sizes < min_patch_area)[0])

        # Convert small patches to value 0
        raster_data[np.logical_and(small_patches_mask, raster_data == 1)] = 0

        # Set nodata values to specified nodata_value
        raster_data[raster_data == nodata] = nodata_value

        # Update the metadata to handle the new nodata value and dtype
        kwargs = src.meta.copy()
        kwargs.update(nodata=nodata_value, dtype='uint8', driver='GTiff')

        # Create a new raster file for the modified data
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(raster_data, 1)

def process_subfolder(subdir):
    tilepath_pred = os.path.join(dir_path_pred, subdir)
    print(tilepath_pred)
    listraster = os.listdir(tilepath_pred)

    tilepath_mask = os.path.join(dir_path_mask, subdir)
    print(tilepath_mask)
    listmask = os.listdir(tilepath_mask)

    for year in range(1985, 2024):
        if os.path.isdir(tilepath_pred):
            current_file_pred = os.path.join(tilepath_pred, f"{year}{common_int}.tif")
            with rasterio.open(current_file_pred) as src:
                pred_raster = src.read(1)

                mask_file = os.path.join(tilepath_mask, "forest_nonforest_multitemp.tif")

                with rasterio.open(mask_file) as mask_src:
                    mask_array = mask_src.read(1)

                    # Mask pred_raster where mask_array is 0
                    masked_pred_array = np.where(mask_array == 0, np.nan, pred_raster)

                    output_subfolder = os.path.join(output_dir, subdir)
                    os.makedirs(output_subfolder, exist_ok=True)

                    profile = src.profile.copy()
                    profile.update(dtype=rasterio.uint8, nodata=255)

                    output_pred_file = os.path.join(output_subfolder, f"{year}_disturbed_undisturbed_pred_v211.tif")
                    with rasterio.open(output_pred_file, 'w', **profile) as dst:
                        dst.write(masked_pred_array.astype(rasterio.uint8), 1)
                    print("**********prediction exported************")
                    print(f"Masked {year} for {subdir}")

                    # Apply eliminate_small_patches function
                    input_path = output_pred_file
                    output_iso_path = os.path.join(output_subfolder, f"{year}__disturbed_undisturbed_pred_v211_masked.tif")
                    eliminate_small_patches(input_path, output_iso_path, min_patch_area=3, nodata_value=255)
                    print("**********small patches eliminated************")

    
if __name__ == '__main__':
    dir_path_pred = '/path/to/europe/RF_outputs/europe/disturbed_undisturbed/'
    dir_path_mask = '/path/to/europe/RF_outputs/europe/forest_nonforest/'
    common_int = "_disturbed_undisturbed_predth_v211"
    output_dir = '/path/to/europe/RF_outputs/europe/disturbed_undisturbed/'

    os.chdir(dir_path_pred)
    os.chdir(dir_path_mask)
    #lista_pred = os.listdir(dir_path_pred)
    lista_mask = os.listdir(dir_path_mask)
    
    # Read subfolder names from the text file
    subfolder_file_path = '/path/to/subfolders_to_process_tiles_datacube.txt'
    with open(subfolder_file_path, 'r') as file:
        subfolders_list = file.read().splitlines()

    start_time = time.time()

    num_threads = 10
    with Pool(processes=num_threads) as pool:
        pool.map(process_subfolder, subfolders_list)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken: ", elapsed_time)