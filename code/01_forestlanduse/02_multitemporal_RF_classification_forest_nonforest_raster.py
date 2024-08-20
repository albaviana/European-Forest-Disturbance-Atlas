#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Script to make raster predictions on forest land use based on pre-trained model in step 01.
# Predictions are made based on all variables (bands+indices) and years (multitemporal classification)
# Designed to make predictions per tile in the datacube and chuncks to avoid overloading

import os
import rasterio
import numpy as np
from joblib import load

# Function to predict a chunk of the raster data using the random forest model with a specified threshold
def predict_chunk_with_threshold(chunk_data, rf_model, ordered_indices, threshold=0.5):
    chunk_data_reordered = chunk_data[:, :, ordered_indices, :]
    chunk_reshaped = chunk_data_reordered.reshape(-1, num_years * num_variables)
    chunk_reshaped[np.isnan(chunk_reshaped)] = -9999
    
    # Predict probabilities for the positive class
    y_prob_chunk = rf_model.predict_proba(chunk_reshaped)[:, 1]
    
    # Convert probabilities to binary predictions based on the threshold
    y_pred_chunk = (y_prob_chunk > threshold).astype(int)
    
    return y_pred_chunk

# Load the pre-trained Random Forest model
rf_model = load('/path/to/model/level1_aux/best_rf_forest_nonforest.joblib')

# Read the subfolder names from the text file
subfolder_list_file = "/path/to/subfolders_to_process_tiles_datacube.txt"
with open(subfolder_list_file, "r") as f:
    subfolders_to_process = [line.strip() for line in f.readlines()]

# Define the root directory path where the subfolders are located
dir_path = '/path/to/level3/europe/'
common_int = "0801_LEVEL3_LNDLG"
suffixes = ["_NBR.tif", "_NDVI.tif", "_TCB.tif", "_TCG.tif", "_TCW.tif", "_TC_DIn.tif"]

# Define the output directory for the difference rasters
output_dir = '/path/to/europe/RF_outputs/europe/forest_nonforest/'

os.chdir(dir_path)
lista = os.listdir(dir_path)

# Get a list of all subfolders in the parent folder
# Iterate over all subfolders within the parent folder
for subdir in lista:
    if subdir not in subfolders_to_process:
        continue

    tilepath = os.path.join(dir_path, subdir)
    print(tilepath)

    # Initialize an array to store all data for each year
    num_years = 2019 - 1985
    num_variables = 12  # 6 IBAP variables + 6 index variables
    
    # Set the chunk size
    chunk_size = 1250
    
    # Initialize predictions array
    predictions = np.zeros((5000, 5000))

    # Iterate through chunks and make predictions
    for i in range(0, 5000, chunk_size):
        for j in range(0, 5000, chunk_size):
            # Initialize chunk_data for IBAP and indices
            chunk_data = np.zeros((chunk_size, chunk_size, num_variables, num_years))
    
            # Iterate over the years
            for year in range(1985, 2019):
                # Open the raster for IBAP variables
                if os.path.isdir(tilepath):
                    current_fileIBAP = os.path.join(tilepath, f"{year}{common_int}_IBAP.tif")
                    with rasterio.open(current_fileIBAP) as src:
                        ibap_data = np.moveaxis(src.read(window=((i, i + chunk_size), (j, j + chunk_size))), 0, -1)
                        chunk_data[:, :, :6, year - 1985] = ibap_data
                        print(f"Added IBAP data for year {year}")
    
                # Open the raster for other variables (indices)
                if os.path.isdir(tilepath):
                    for suffix_index, suffix in enumerate(suffixes):
                        current_fileIND = os.path.join(tilepath, f"{year}{common_int}{suffix}")
                        with rasterio.open(current_fileIND) as src:
                            index_data = src.read(window=((i, i + chunk_size), (j, j + chunk_size)))
                            chunk_data[:, :, 6 + suffix_index, year - 1985] = index_data
                            print(f"Added indices data for year {year}")
                    # Print the order of variables in chunk_data for the current year
                    print(f"Order of variables for year {year}: {chunk_data[0, 0, :, year - 1990]}")

            # Rearrange the order of predictors based on your RF model
            ordered_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            chunk_data_reordered = chunk_data[:, :, ordered_indices, :]
            
            # Predict using the trained model with the specified threshold
            predictions_chunk = predict_chunk_with_threshold(chunk_data, rf_model, ordered_indices, threshold=0.5)
    
            # Update the predictions array with chunk predictions
            predictions[i:i + chunk_size, j:j + chunk_size] = predictions_chunk.reshape(chunk_size, chunk_size)

    # Export the predictions to a new raster file
    subdir_name = str(tilepath[74:87])
    os.makedirs(os.path.join(output_dir, subdir_name), exist_ok=True)
    output_pred_file = os.path.join(output_dir, subdir_name, "forest_nonforest_multitemp.tif")
    with rasterio.open(output_pred_file, 'w', driver='GTiff', height=src.height, width=src.width,
                       count=1, dtype=np.uint8, crs=src.crs, transform=src.transform) as dst:
        dst.write(predictions.astype(np.uint8), 1)

    print(f"******prediction exported******for {output_pred_file}****")