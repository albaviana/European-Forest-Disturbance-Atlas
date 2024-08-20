#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

import pandas as pd
import numpy as np

# It calculates differences from year to year in reference data
df = pd.read_csv('/path/to/forest_dataset_alltiles_landsatdata.csv', sep=';')

df_select_cols = df.drop(['plotid', 'merge_id', 'class_level1','class_level2','country', 'coordx', 'coordy', 'uniqueid', 'Tile_ID'], axis=1) # Select all columns except the last one  data.iloc[:, :-1]

# Sort the DataFrame by point and year
df_select_cols = df_select_cols.sort_values(['fid_y', 'year'])

# Group the DataFrame by point
grouped = df_select_cols.groupby('fid_y') # this id has to be unique per plot for the entire dataset (i.e. two plots from two countries cannot have same fid_y)

# Calculate the differences between the current row and the previous row
diff_df = grouped.diff()

# Rename the columns in the diff DataFrame to indicate that they are differences
diff_df = diff_df.rename(columns={'BLU': 'BLU_diff', 'GRN': 'GRN_diff', 'RED': 'RED_diff', 'NIR': 'NIR_diff', 'SW1': 'SW1_diff', 'SW2': 'SW2_diff', 
                                  'NBR': 'NBR_diff', 'NDVI': 'NDVI_diff', 'TCB': 'TCB_diff', 'TCG': 'TCG_diff', 'TCW': 'TCW_diff', 'DIn': 'DIn_diff'})

# Add the diff DataFrame to the original DataFrame
df_concat = pd.concat([df, diff_df], axis=1)

# Print the resulting DataFrame
print(df_concat)

df_concat.to_csv('/path/to/forest_dataset_alltiles_landsatdata_DIFFS.csv')