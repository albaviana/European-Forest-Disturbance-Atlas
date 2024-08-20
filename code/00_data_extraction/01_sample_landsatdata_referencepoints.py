#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""
import os
import geopandas as gpd
import rasterio as rio
from shapely.geometry import Polygon     
import time
start_time = time.time()  
        

#### Open Landsat data to be sampled at reference points: ####
worksp = '/path/to/datacube/level3/'
os.chdir(worksp)
list = os.listdir(worksp)

# Read the shapefile to extract raster values (for forest and non forest - decomment alternatively)
shapefile_gdf = gpd.read_file('/path/to/forest_reference_data') #shp #gpkg
#shapefile_gdf = gpd.read_file('/path/to/nonforest_reference_data') #shp #gpkg

# Iterate through tiles
for d in list:
    tilepath = os.path.join(worksp, d)
    print (tilepath)
    listraster = os.listdir(tilepath)
    for file in listraster:
        if file.endswith('.tif'):
            raster_path = os.path.join(tilepath, file)
            with rio.open(raster_path) as src:
               # Get the number of bands in the raster
               num_bands = src.count
               transform = src.transform
               crs = src.crs
               
               # Get the bounds of the raster
               bounds = src.bounds
               # Create a polygon from the bounds
               poly = Polygon([(bounds.left, bounds.bottom), (bounds.left, bounds.top), 
                                    (bounds.right, bounds.top), (bounds.right, bounds.bottom)])
               # Extract the points from the shapefile that intersect with the polygon
               intersecting_gdf = shapefile_gdf[shapefile_gdf.geometry.intersects(poly)]
                    
               # Extract the date from the raster file name
               date = file.split("_")[0].split(".")[0] + file.split("_")[-1].split(".")[0]
               # Create an empty column to store the results
               col_name = date
               
               # List of the coordinates in x,y format rather than as the points that are in the geomentry column               
               coord_list = [(x,y) for x,y in zip(shapefile_gdf['geometry'].x , shapefile_gdf['geometry'].y)] 

               samples = [x for x in src.sample(coord_list)]  # f'band_{i+1}' 

               # Create a new column in the DataFrame with the values from this raster
               shapefile_gdf[col_name] = samples
               
               
    print (intersecting_gdf.head())
                    
    df_cleaned = intersecting_gdf.astype(str).replace({"\[":"", "\]":""}, regex=True)
    # arrange extraction in chronological order, from 1984 to 2023
    df_cleaned  = df_cleaned.reindex(sorted(df_cleaned.columns), axis=1)
        
    outpath = '/path/to/ouput_folder/'
    df_cleaned.to_csv(os.path.join(outpath, f"forest_dataset_{d}.csv"))
    #df_cleaned.to_csv(os.path.join(outpath, f"non_forest_dataset_{d}.csv"))
                    
print("**** Reference data extracted *****")  
  
end_time = time.time()
elapsed_time = end_time - start_time
print("Time taken: ", elapsed_time)