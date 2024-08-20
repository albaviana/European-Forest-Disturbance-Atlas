#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""
import pandas as pd
import glob

folder = '/path/to/forest_reference_samples/'
#folder = '/path/to/non_forest_reference_samples/

all_files = glob.glob(folder + "forest_dataset_*.csv")
#all_files = glob.glob(folder + "non_forest_dataset_*.csv")

df_list = []

for filename in all_files:
    df = pd.read_csv(filename)
    df_list.append(df)

concatenated_df = pd.concat(df_list)
concatenated_df.to_csv('/path/to/all_refrencedata_eu.csv')