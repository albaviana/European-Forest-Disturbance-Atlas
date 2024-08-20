#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Script to collapse/filter consecutive disturbances or alternative disturbances in short time-frames that lead
# to illogical changes i.e. disturbance- no disturbance -disturbance.

import os
import re
import numpy as np
import rasterio
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor, as_completed

def stack_rasters(input_folder, output_path):
    # Get a list of all raster files in the input folder
    raster_files = [f for f in os.listdir(input_folder) if f.endswith('predth_v211_masked.tif')]

    # Sort the raster files based on the year in the filename
    sorted_raster_files = sorted(raster_files, key=lambda x: int(re.search(r'\d{4}', x).group()))

    # Read the first raster to get metadata
    with rasterio.open(os.path.join(input_folder, sorted_raster_files[0])) as src:
        metadata = src.meta.copy()

    # Update the count to the number of years
    metadata['count'] = len(sorted_raster_files)

    # Create an empty array to store the stacked data
    stacked_data = np.zeros((metadata['count'], metadata['height'], metadata['width']), dtype=np.uint8)

    # Loop through each sorted raster file and stack the data
    for i, raster_file in enumerate(sorted_raster_files):
        input_path = os.path.join(input_folder, raster_file)
        with rasterio.open(input_path) as src:
            stacked_data[i, :, :] = src.read(1)

    # Write the stacked data to a new raster file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=metadata['height'],
        width=metadata['width'],
        count=metadata['count'],
        dtype=np.uint8,
        crs=metadata['crs'],
        transform=metadata['transform'],
        nodata=255,  # Set nodata value to 255
    ) as dst:
        dst.write(stacked_data)


def identify_disturbed_pixels(stacked_data):
    # Count the number of disturbances for each pixel
    disturbed_count = np.sum(stacked_data == 1, axis=0)

    # Identify pixels with more than one disturbance
    multiple_disturbance_pixels = disturbed_count > 1

    return multiple_disturbance_pixels

def reclassify_consecutive_disturbances(stacked_data, multiple_disturbance_pixels):
    # Loop through each pixel and band
    for i in range(stacked_data.shape[1]):
        for j in range(stacked_data.shape[2]):
            if multiple_disturbance_pixels[i, j]:
                # Find three consecutive disturbances and reclassify the "second" and "third" disturbances to 0
                for k in range(stacked_data.shape[0] - 2):
                    if (
                        stacked_data[k, i, j] == 1
                        and stacked_data[k + 1, i, j] == 1
                        and stacked_data[k + 2, i, j] == 1
                    ):
                        stacked_data[k + 1, i, j] = 0
                        stacked_data[k + 2, i, j] = 0
                
                # Find consecutive disturbances and reclassify the "second" disturbance to 0
                for k in range(stacked_data.shape[0] - 1):
                    if stacked_data[k, i, j] == 1 and stacked_data[k + 1, i, j] == 1:
                        stacked_data[k + 1, i, j] = 0
                        
                # Find alternate disturbances and reclassify the second disturbance to 0
                for k in range(stacked_data.shape[0] - 3):
                    if (
                        (stacked_data[k, i, j] == 1 and stacked_data[k + 1, i, j] == 0 and stacked_data[k + 2, i, j] == 1)
                        or (stacked_data[k, i, j] == 1 and stacked_data[k + 1, i, j] == 0 and stacked_data[k + 2, i, j] == 0 and stacked_data[k + 3, i, j] == 1)
                    ):
                        stacked_data[k + 2, i, j] = 0
                        stacked_data[k + 3, i, j] = 0
    return stacked_data


def process_subfolder(subfolder, input_folder, output_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    output_path = os.path.join(output_folder, subfolder, "stacked_disturbed_undisturbed.tif")
    
    # Step 1: Stack the rasters
    stack_rasters(subfolder_path, output_path)
    print(f"Stacked rasters for {subfolder}")

    # Step 2: Read the stacked raster data
    with rasterio.open(output_path) as src:
        stacked_data = src.read()
    print(f"Read stacked raster for {subfolder}")

    # Step 3: Identify and reclassify consecutive disturbances
    multiple_disturbance_pixels = identify_disturbed_pixels(stacked_data)
    stacked_data = reclassify_consecutive_disturbances(stacked_data, multiple_disturbance_pixels)
    print(f"Reclassified disturbances for {subfolder}")

    # Step 4: Export each band as a separate raster file
    for year in range(1985, 2024):
        output_band_path = os.path.join(output_folder, subfolder, f"{year}_disturbed_undisturbed_predth_v211_masked_filt.tif")
        band_index = year - 1985  # Adjust band index based on the starting year

        with rasterio.open(
            output_band_path,
            'w',
            driver='GTiff',
            height=stacked_data.shape[1],
            width=stacked_data.shape[2],
            count=1,
            dtype=np.uint8,
            crs=src.crs,
            transform=src.transform,
            nodata=255,  # Set nodata value to 255
        ) as dst:
            dst.write(stacked_data[band_index, :, :], 1)
            print(f"Exported reclassified raster for {subfolder} and year {year}")

def export_filtered_output_processes(input_folder, output_folder):
    # Iterate over subfolders in the input folder
    #subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    
    subfolder_file_path = '/path/to/subfolders_to_process_tiles_datacube.txt'
    with open(subfolder_file_path, 'r') as file:
        subfolders = file.read().splitlines()

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_subfolder, subfolder, input_folder, output_folder): subfolder for subfolder in subfolders}

        for future in as_completed(futures):
            subfolder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {subfolder}: {e}")


# Specify the input and output folders
input_folder = '/path/to/europe/RF_outputs/europe/disturbed_undisturbed/'
output_folder = '/path/to/europe/RF_outputs/europe/disturbed_undisturbed/'

# Call the function to export the filtered output for all subfolders
export_filtered_output_processes(input_folder, output_folder)