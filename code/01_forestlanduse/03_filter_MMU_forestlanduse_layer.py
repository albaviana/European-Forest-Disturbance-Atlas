#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Script to filter small forest patches to match FAO definition of forest (see preprint section 2.3)

import os
import rasterio
import numpy as np
from scipy.ndimage import label, binary_dilation
from concurrent.futures import ThreadPoolExecutor
import time

start_time = time.time()

# Function to eliminate small patches of forest
def eliminate_patches(raster, profile, output_path, min_patch_size=5):
    # Find connected patches of pixels with value = 1
    labeled_array, num_features = label(raster)

    # Eliminate small patches
    eliminated = raster.copy()
    for label_value in range(1, num_features + 1):
        patch_size = (labeled_array == label_value).sum()
        if patch_size < min_patch_size:
            eliminated[labeled_array == label_value] = 0
            
    return eliminated, profile

# Function to eliminate small holes in forest patches
def convert_small_patches(classified_raster, profile, output_path, min_patch_size=5):
    # Create a structuring element for dilation
    structuring_element = np.ones((3, 3), dtype=np.uint8)

    # Perform binary dilation to expand forest patches
    dilated = binary_dilation(classified_raster, structure=structuring_element)

    # Identify small non-forest patches within dilated forest patches
    small_nonforest_patches = (classified_raster == 0) & (dilated == 1)
    labeled_array, num_features = label(small_nonforest_patches)

    # Convert small non-forest patches to forest
    converted = classified_raster.copy()
    for label_value in range(1, num_features + 1):
        patch_size = (labeled_array == label_value).sum()
        if patch_size < min_patch_size:
            converted[labeled_array == label_value] = 1

    return converted, profile

# Specify the number of threads/processes to use
num_threads = 10

# Create a thread pool executor
executor = ThreadPoolExecutor(max_workers=num_threads)

# Define the root directory path where the subfolders are located
dir_path = '/path/to/europe/RF_outputs/europe/forest_nonforest/forest_nonforest_multitemp/'

# Read the list of subdirectories from the text file
with open('/path/to/subfolders_to_process_tiles_datacube.txt', 'r') as subfolders_file:
    subdirectories = subfolders_file.read().splitlines()

# Loop over the raster predictions for each subdirectory
for subdir in subdirectories:
    tilepath = os.path.join(dir_path, subdir)
    input_path = os.path.join(tilepath, "forest_nonforest_multitemp.tif")
    output_path1 = os.path.join(tilepath, "forest_nonforest_multitemp_MMU.tif")
    output_path2 = os.path.join(tilepath, "forest_nonforest_multitemp_MMUfill.tif")

    # Open the raster directly without classification
    with rasterio.open(input_path) as src:
        # Read the raster data
        raster = src.read(1)
        print(f"reading.....{subdir}")
        
        # Get the profile information before exiting the context
        raster_profile = src.profile

    # Submit the task to eliminate small patches
    task2 = executor.submit(eliminate_patches, raster, raster_profile, output_path1, min_patch_size=5)
    eliminated_raster1, _ = task2.result()
    print(f"elim patches done for {subdir}")

    # Write the eliminated raster to disk
    with rasterio.open(output_path1, 'w', **raster_profile) as dst:
        dst.write(eliminated_raster1, 1)

    # Submit the task to eliminate small holes in forest patches
    task3 = executor.submit(convert_small_patches, eliminated_raster1, raster_profile, output_path2, min_patch_size=5)
    eliminated_raster2, _ = task3.result()
    print(f"elim small holes done for {subdir}")

    # Write the second eliminated raster to disk
    with rasterio.open(output_path2, 'w', **raster_profile) as dst:
        dst.write(eliminated_raster2, 1)

    print(f"Processing raster: {subdir}")

# Shutdown the executor to wait for all tasks to complete
executor.shutdown(wait=True)

end_time = time.time()
elapsed_time = end_time - start_time
print("Time taken: ", elapsed_time)