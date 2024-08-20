#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Summarises greatest disturbance per pixel (1985-2023)

import os
import rasterio
import numpy as np
import concurrent.futures

# Input folder path
input_folder = "/path/to/europe/RF_outputs/europe/disturbed_undisturbed/"
output_parent_folder = "/path/to/europe/RF_outputs/europe/disturbed_undisturbed/summary/"

# Get a list of all subfolders in the input folder
subfolders = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))]

# Read the list of subdirectories from the text file
with open("/path/to/subfolders_to_process_tiles_datacube.txt", 'r') as subfolders_file:
    subdirectories = subfolders_file.read().splitlines()
    
# Iterate through each subfolder
def process_subfolder(subfolder):
    for subfolder in subdirectories:
        # Construct the full path to the current subfolder
        subfolder_path = os.path.join(input_folder, subfolder)
    
        # Output file paths for the current subfolder
        output_min_magnitude_path = os.path.join(output_folder, subfolder, "greatest_disturbance_magnitudev211.tif")
        output_corresponding_year_path = os.path.join(output_folder, subfolder, "greatest_disturbance_v211.tif")
        
        # Skip processing if the output file already exists
        if os.path.exists(output_corresponding_year_path):
            print(f"Output file 'greatest_disturbance_v3.tif' already exists for {subfolder}. Skipping...")
            continue
    
        # List all files in the current subfolder
        input_files = os.listdir(subfolder_path)
    
        # List all files in the input subfolders
        magnitude_files = [f for f in input_files if f.endswith("_spectralbands_indices_diff1.tif")]
        year_files = [f for f in input_files if f.endswith("_disturbed_undisturbed_pred_v211_masked_filt.tif")]
    
        # Initialize the results arrays with large initial values
        min_magnitude_data = None
        corresponding_year_data = None
    
        # Initialize geospatial variables
        transform = None
        nodata_value = -9999  # Adjust as needed
    
        # Iterate through the magnitude files and process each one
        for magnitude_file in magnitude_files:
            # Extract the year from the magnitude file name
            year = int(magnitude_file.split("_")[0])
    
            # Construct the full file path for magnitude
            magnitude_file = os.path.join(subfolder_path, magnitude_file)
    
            # Open the magnitude file
            with rasterio.open(magnitude_file) as ds:
                # Read band 7 (magnitude data)
                magnitude_data = ds.read(7)
    
                # Get the geospatial information from the first file
                if transform is None:
                    transform = ds.transform
    
                # Set nodata values where data is missing (adjust this based on your data)
                magnitude_data[magnitude_data <= -2] = nodata_value
    
                # Initialize the result arrays if not done already
                if min_magnitude_data is None:
                    min_magnitude_data = np.full(magnitude_data.shape, np.inf, dtype=np.float32)
                    corresponding_year_data = np.zeros(magnitude_data.shape, dtype=np.int16)
    
            # Construct the full file path for the corresponding year file
            year_file = os.path.join(subfolder_path, f"{year}_disturbed_undisturbed_pred_v211_masked_filt.tif")
    
            # Open the year file
            with rasterio.open(year_file) as ds:
                # Read the year data
                year_data = ds.read(1)
    
            # Identify pixels where the year value is 1
            year_mask = year_data == 1
    
            # Identify pixels where the current magnitude is lower and the year value is 1
            lower_magnitude_mask = (magnitude_data < min_magnitude_data) & year_mask
    
            # Update the results arrays with the minimum magnitude and corresponding year
            min_magnitude_data[lower_magnitude_mask] = magnitude_data[lower_magnitude_mask]
            corresponding_year_data[lower_magnitude_mask] = year
            
            print(f"greatest disturbance extracting for {subfolder}")
    
        # Create the output files for the minimum disturbance magnitude and corresponding year
        output_subfolder_path = os.path.join(output_folder, subfolder)
        os.makedirs(output_subfolder_path, exist_ok=True)
    
        with rasterio.open(output_corresponding_year_path, 'w', driver='GTiff', height=corresponding_year_data.shape[0], width=corresponding_year_data.shape[1], count=1, dtype=np.int16, crs=ds.crs, transform=transform) as corresponding_year_ds:
            corresponding_year_ds.write(corresponding_year_data, 1)
            corresponding_year_ds.nodata = nodata_value  # Set the nodata value
            print(f"*** DONE greatest disturbance for {subfolder}")
        
# Use a ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(process_subfolder, subfolders)