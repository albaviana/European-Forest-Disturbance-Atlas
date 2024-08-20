#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Summarises latest disturbance per pixel (1985-2023)

import os
import rasterio
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def extract_year_from_filename(filename):
    # Use regular expression to extract the year from the filename
    match = re.match(r'^(\d{4})_', filename)
    if match:
        return int(match.group(1))
    return None

def process_raster(raster_path):
    with rasterio.open(raster_path) as raster:
        # Get the indices of pixels with a value of 1 (disturbed)
        disturbed_indices = np.where(raster.read(1) == 1)

        # Get the year from the filename
        year = extract_year_from_filename(os.path.basename(raster_path))
        print("got the year")
       
        # Create an array with the same shape as the raster, initialized with a special value to represent undisturbed pixels
        latest_disturbance_year = np.full(raster.shape, -9999, dtype=np.int16)

        # Assign the year to the disturbed pixels
        latest_disturbance_year[disturbed_indices] = year
        print("asigned")
        
        return latest_disturbance_year

def combine_results(results):
    # Combine the results to get the latest disturbance per pixel
    return np.stack(results).max(axis=0)

parent_folder = "/path/to/europe/RF_outputs/europe/disturbed_undisturbed/"
output_parent_folder = "/path/to/europe/RF_outputs/europe/disturbed_undisturbed/summary/"

# Define the number of threads to use for parallel processing
num_threads = 10  # Change this to the desired number of threads

# Read subfolder names from a text file
subfolders_file = "/path/to/subfolders_to_process_tiles_datacube.txt"
with open(subfolders_file, 'r') as file:
    subfolders_to_process = [line.strip() for line in file]

# Initialize a list to store the raster file paths
raster_paths = {}

# Traverse through subfolders and find raster files
for root, _, files in os.walk(parent_folder):
    subfolder = os.path.basename(root)
    if subfolder in subfolders_to_process:
        for file in files:
            if file.endswith("_disturbed_undisturbed_pred_v211_masked_filt.tif"):
                if subfolder not in raster_paths:
                    raster_paths[subfolder] = []
                raster_paths[subfolder].append(os.path.join(root, file))

# Create output folders for each subfolder if they don't exist
for subfolder in raster_paths.keys():
    output_folder = os.path.join(output_parent_folder, subfolder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# Use ThreadPoolExecutor to parallelize the raster processing for each subfolder
for subfolder, paths in raster_paths.items():
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_raster, paths))

    # Combine the results to get the latest disturbance per pixel
    latest_disturbance_per_pixel = combine_results(results)

    # Get the reference to the first raster to obtain georeferenced information
    first_raster = rasterio.open(paths[0])

    # Define the output file path for the current subfolder
    output_file = os.path.join(output_parent_folder, subfolder, "latest_disturbance_v211.tif")

    # Optionally, you can save the output as a new raster file using rasterio
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=first_raster.height,
        width=first_raster.width,
        count=1,
        dtype=np.int16,
        crs=first_raster.crs,
        transform=first_raster.transform
    ) as output_raster:
        output_raster.write(latest_disturbance_per_pixel, 1)

    print(f"Latest disturbance exported for subfolder: {subfolder}")