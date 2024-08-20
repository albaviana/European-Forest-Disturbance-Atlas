#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:47:12 2023

@author: aviana
"""

# Script to calculate spectral indices from landsat data cube
# see FORCE documentation for further details on the datacube formats and nomenclature: https://force-eo.readthedocs.io/en/latest/index.html
# to save space Int16 to Tasseled Cap and only use float32 for NDVI-NBR (but scale to 10,000 is recomended)


from osgeo import gdal
import os
import numpy as np
import time

def array_to_raster(array, dsimage, output_name):
    """Converts a NumPy array to a raster file."""
    cols, rows = dsimage.RasterXSize, dsimage.RasterYSize
    transform = dsimage.GetGeoTransform()
    projection = dsimage.GetProjection()

    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    dataset_index = driver.Create(output_name, cols, rows, 1, gdal.GDT_Float32) ##Int16

    # Set the projection and geotransform
    dataset_index.SetGeoTransform(transform)
    dataset_index.SetProjection(projection)
    
    # Write array data to raster
    dataset_index.GetRasterBand(1).WriteArray(array)
    dataset_index.FlushCache()  # Write to disk

    return dataset_index

def calculate_indices(dsimage):
    """Calculates various spectral indices from raster bands."""
    array1 = dsimage.GetRasterBand(1).ReadAsArray().astype(float)  # Blue
    array2 = dsimage.GetRasterBand(2).ReadAsArray().astype(float)  # Green
    array3 = dsimage.GetRasterBand(3).ReadAsArray().astype(float)  # Red
    array4 = dsimage.GetRasterBand(4).ReadAsArray().astype(float)  # NIR
    array5 = dsimage.GetRasterBand(5).ReadAsArray().astype(float)  # SWIR1
    array6 = dsimage.GetRasterBand(6).ReadAsArray().astype(float)  # SWIR2
    
    # Calculate indices and tasseled cap 
    # TC coefficients from http://dx.doi.org/10.1080/2150704X.2014.915434
    ndvi = (array4 - array3) / (array4 + array3)
    nbr = (array4 - array6) / (array4 + array6)
    tcb = 0.3029 * array1 + 0.2786 * array2 + 0.4733 * array3 + 0.5599 * array4 + 0.508 * array5 + 0.1872 * array6
    tcg = -0.2941 * array1 - 0.243 * array2 - 0.5424 * array3 + 0.7276 * array4 + 0.0713 * array5 - 0.1608 * array6
    tcw = 0.1511 * array1 + 0.1973 * array2 + 0.3283 * array3 + 0.3407 * array4 - 0.7117 * array5 - 0.4559 * array6
    tc_di = tcb - (tcg + tcw) # Disturbance index: (Healey et al., 2005)
    
    # Normalized indices
    tcb_n = (tcb - np.mean(tcb)) / np.std(tcb)
    tcg_n = (tcg - np.mean(tcg)) / np.std(tcg)
    tcw_n = (tcw - np.mean(tcw)) / np.std(tcw)
    tc_di_n = tcb_n - (tcg_n + tcw_n)               

    return {
        "NDVI": ndvi,
        "NBR": nbr,
        "TCB": tcb,
        "TCG": tcg,
        "TCW": tcw,
        "TC_DI": tc_di,
        "TC_DI_N": tc_di_n
    }

def process_images_in_folder(worksp):
    """Processes all images in the specified workspace."""
    gdal.AllRegister()
    os.chdir(worksp)
    folders = os.listdir(worksp)

    start_time = time.time()
    
    for folder in folders:
        folder_path = os.path.join(worksp, folder)
        print(f"Processing folder: {folder_path}")
        
        files = os.listdir(folder_path)
        for f in files:
            # to open composites data per year
            if f.endswith('801_LEVEL3_LNDLG_IBAP.tif'):  
                image_path = os.path.join(folder_path, f)
                print(f"Processing image: {image_path}")
                
                dsimage = gdal.Open(image_path, gdal.GA_ReadOnly)
                if dsimage is None:
                    print(f"Failed to open image: {image_path}")
                    continue

                # Calculate indices
                indices = calculate_indices(dsimage)

                # Save each index as a separate raster file
                for index_name, array in indices.items():
                    output_name = os.path.join(folder_path, f.replace('_IBAP.tif', f'{index_name}.tif'))
                    array_to_raster(array, dsimage, output_name)
                    print(f"Raster for {index_name} saved as: {output_name}")
                print('******************************')
                
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    workspace = '/path/to/level3/europe/'
    process_images_in_folder(workspace)