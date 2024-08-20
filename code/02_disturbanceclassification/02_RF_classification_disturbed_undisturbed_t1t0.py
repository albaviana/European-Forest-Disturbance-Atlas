#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""
# Script to make predictions on raster for disturbed and undisturbed pixels based on
# pretained model in step 01.


import os
import time
import multiprocessing as mp
import rasterio
import numpy as np
from joblib import load

def load_subfolders(txt_file_path):
    with open(txt_file_path, 'r') as txt_file:
        return [line.strip() for line in txt_file.readlines()]

def load_raster(file_path, band):
    with rasterio.open(file_path) as src:
        return src.read(band), src.width, src.height, src.crs, src.transform

def calculate_differences(current_rasters, prev_rasters, suffixes):
    diff_rasters = {}
    for suffix in suffixes:
        diff_rasters[suffix] = current_rasters[suffix] - prev_rasters[suffix]
    return diff_rasters

def stack_raster_data(current_rasters, diff_rasters, year_suffixes):
    raster_data = [
        current_rasters[suffix] for suffix in year_suffixes
    ] + [
        diff_rasters[suffix] for suffix in year_suffixes
    ]
    return np.dstack(raster_data)

def predict_chunk(chunk, model):
    return model.predict(chunk)

def predict_raster_data(num_threads, raster_data, model):
    chunk_size = raster_data.shape[0] // num_threads
    chunks = [raster_data[i:i + chunk_size] for i in range(0, raster_data.shape[0], chunk_size)]
    
    with mp.Pool(num_threads) as pool:
        results = pool.starmap(predict_chunk, [(chunk, model) for chunk in chunks])

    return np.concatenate(results).reshape(5000, 5000)

def save_predictions(output_pred_file, predictions, width, height, crs, transform):
    with rasterio.open(output_pred_file, 'w', driver='GTiff',
                       height=height, width=width, count=1, dtype=np.uint8,
                       crs=crs, transform=transform, compress='lzw') as dst:
        dst.write(predictions.astype(np.uint8), 1)

def process_subfolder(subdir, model, dir_path, common_int, suffixes, output_dir, num_threads):
    tilepath = os.path.join(dir_path, subdir)
    for year in range(1985, 2024):
        current_rasters, prev_rasters, prev_exists = {}, {}, False
        prev_year = year - 1

        for suffix in suffixes:
            current_fileIND = os.path.join(tilepath, f"{year}{common_int}{suffix}")
            current_rasters[suffix], width, height, crs, transform = load_raster(current_fileIND, 1)

            prev_fileIND = os.path.join(tilepath, f"{prev_year}{common_int}{suffix}")
            if os.path.exists(prev_fileIND):
                prev_rasters[suffix], _, _, _, _ = load_raster(prev_fileIND, 1)
                prev_exists = True

        if prev_exists:
            diff_rasters = calculate_differences(current_rasters, prev_rasters, suffixes)
        else:
            diff_rasters = {suffix: np.zeros_like(current_rasters[suffix]) for suffix in suffixes}

        raster_data_stack = stack_raster_data(current_rasters, diff_rasters, suffixes)
        raster_data = raster_data_stack.reshape(-1, raster_data_stack.shape[-1])
        raster_data[np.isnan(raster_data)] = 0

        predictions = predict_raster_data(num_threads, raster_data, model)
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        output_pred_file = os.path.join(output_dir, subdir, f"{year}_disturbed_undisturbed_pred_v211.tif")
        save_predictions(output_pred_file, predictions, width, height, crs, transform)

def main():
    start_time = time.time()
    num_threads = 10
    rf_model = load('/path/to/model/level2_aux/best_rf_disturbed_undisturbed.joblib')
    dir_path = '/path/to/level3/europe/'
    txt_file_path = '/path/to/subfolders_to_process_tiles_datacube.txt'
    output_dir = '/path/to/europe/RF_outputs/europe/disturbed_undisturbed/'
    common_int = "0801_LEVEL3_LNDLG"
    suffixes = ["_NBR.tif", "_NDVI.tif", "_TCB.tif", "_TCG.tif", "_TCW.tif", "_TC_DIn.tif"]
    
    subfolders_to_process = load_subfolders(txt_file_path)
    os.chdir(dir_path)
    
    for subdir in subfolders_to_process:
        process_subfolder(subdir, rf_model, dir_path, common_int, suffixes, output_dir, num_threads)
    
    elapsed_time = time.time() - start_time
    print("Time taken: ", elapsed_time)

if __name__ == "__main__":
    main()
