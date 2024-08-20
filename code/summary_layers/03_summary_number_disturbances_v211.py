#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Summarises number of disturbance events per pixel (1985-2023)

import os
import numpy as np
import rasterio
from concurrent.futures import ThreadPoolExecutor

def process_subfolder(args):
    root, output_folder, file_pattern, custom_nodata_value = args
    relative_path = os.path.relpath(root, parent_folder)
    subfolder_name = os.path.basename(root)

    output_subfolder = os.path.join(output_folder, relative_path)
    os.makedirs(output_subfolder, exist_ok=True)

    output_filename = os.path.join(output_subfolder, "summary_number_disturbances_v211.tif")
    summary = None
    nodata_value = None

    for file in os.listdir(root):
        if file.endswith(file_pattern):
            filepath = os.path.join(root, file)
            with rasterio.open(filepath) as dataset:
                data = dataset.read(1, masked=True)
                print(f"reading {subfolder_name}")

                if summary is None:
                    summary = np.ma.masked_array(np.zeros(data.shape, dtype='uint8'), mask=data.mask)
                    nodata_value = dataset.nodata

                is_disturbance = (data == 1) & ~data.mask
                summary += is_disturbance.astype('uint8')

    if summary is not None:
        with rasterio.open(os.path.join(root, file)) as template_dataset:
            profile = template_dataset.profile
            profile['dtype'] = 'uint8'
            profile['nodata'] = None

            with rasterio.open(output_filename, 'w', **profile) as dst:
                dst.write(summary.filled(custom_nodata_value).astype('uint8'), 1)
                dst.update_tags(1, name=subfolder_name)
                print(f"**DONE {subfolder_name}")

def create_summary_raster(input_folder, output_folder, file_pattern, subfolders_to_process):
    args_list = [(root, output_folder, file_pattern, custom_nodata_value) for root, _, files in os.walk(input_folder) if
                 os.path.basename(root) in subfolders_to_process]

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_subfolder, args_list)

    print("Summary number raster files created successfully.")

if __name__ == "__main__":
    parent_folder = "/path/to/europe/RF_outputs/europe/disturbed_undisturbed/"
    output_parent_folder = "/path/to/europe/RF_outputs/europe/disturbed_undisturbed/summary/"
    file_pattern = "_disturbed_undisturbed_pred_v211_masked_filt.tif"
    custom_nodata_value = 255

    # Read subfolder names from a text file
    subfolders_file = "/path/to/subfolders_to_process_tiles_datacube.txt"
    with open(subfolders_file, 'r') as file:
        subfolders_to_process = [line.strip() for line in file]

    create_summary_raster(parent_folder, output_folder, file_pattern, subfolders_to_process)

