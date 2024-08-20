#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Summarises disturbance agent per pixel (1985-2023)

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from multiprocessing import Pool

input_folder = "/path/to/europe/RF_outputs/europe/disturbed_undisturbed/"
output_parent_folder = "/path/to/europe/RF_outputs/europe/disturbed_undisturbed/summary/"
   
def process_subfolder(subfolder_path, output_dir):
    folder = os.path.basename(subfolder_path)
    print(f"Processing subfolder: {folder}")
      
    first_raster = True
    final_result = None

    for year in range(1985, 2024):
        raster_path = os.path.join(subfolder_path, f"{year}_disturbed_undisturbed_attribution_barkbeetle_fire_wind.tif")
        print(f"  Processing year: {year}, {folder}")

        with rasterio.open(raster_path) as src:
            data = src.read(1, masked=True)
            
            if year == 1985:
                final_result = data
            else:
                non_nan_mask = ~np.isnan(data)
                final_result[non_nan_mask] = np.where(np.isnan(final_result[non_nan_mask]), data[non_nan_mask], final_result[non_nan_mask])
                mixed_mask = (final_result != data) & ~np.isnan(final_result) & ~np.isnan(data)
                final_result[mixed_mask] = 4

    output_path = os.path.join(output_folder, folder, f"{folder}_summary.tif")
    profile = src.profile
    profile.update(dtype=final_result.dtype, count=1)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(final_result, 1)

    print(f"  Finished processing subfolder: {folder}")

def main():
    subfolders = [os.path.join(input_folder, folder) for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

    num_processes = 10  # Specify the number of processes/threads to use
    print(f"Starting parallel processing with {num_processes} processes...")
    
    output_paths = []
    with Pool(processes=num_processes) as pool:
        for subfolder in subfolders:
            output_path = os.path.join(output_folder, os.path.basename(subfolder), f"{os.path.basename(subfolder)}_summary.tif")
            output_paths.append(output_path)
            pool.apply_async(process_subfolder, args=(subfolder, output_folder))
        
        pool.close()
        pool.join()

    print("All subfolders processed.")
    print("Output paths:")
    for path in output_paths:
        print(path)

if __name__ == "__main__":
    main()

