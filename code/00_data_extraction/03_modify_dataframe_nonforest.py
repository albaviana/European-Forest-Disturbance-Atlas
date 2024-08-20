#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Script to reorganise data extracted
# Separate in columns each Landsat ban value per year and flip, each ID plot should have a time series
# i.e. each plot has as many rows as years and bands are in columns

import pandas as pd

df_non_forest = pd.read_csv('/path/to/non_forest_dataset_alltiles.csv', index_col=(0))  

# from original data, first extract IBAP info to split columns and the melt (darle la vuelta para que cada id plot tenga la serie)

# select IBAP columns first to then split in 6 columns -bands
df_IBAP = df_non_forest .loc[:, df_non_forest.columns.str.endswith('IBAP')]
# select column 'plotid' by name
df_IBAP_id = df_non_forest ['POINT_ID']
# select column 'plotid' by name
df_IBAP_class = df_non_forest ['LC1_class']

# Loop through the columns and split the values
for col in df_IBAP.columns:
    df_IBAP[[f'{col}_BLU', f'{col}_GRN', f'{col}_RED', f'{col}_NIR', f'{col}_SW1', f'{col}_SW2']] = df_IBAP[col].str.split(expand=True)

df_selectedIBAP = df_IBAP.iloc[:, 38:]   # select from 38 column onwards

# concatenate the two selected columns into a new DataFrame including the IBAP values and the plot id
df_IBAP_r = pd.concat([df_selectedIBAP, df_IBAP_id, df_IBAP_class], axis=1)
# get all unique values of the 'ID' column
ids_IBAP = df_IBAP_r ['POINT_ID'].unique()

# create a list to store the results
list_BLU = []
list_GRN = []
list_RED = []
list_NIR = []
list_SW1 = []
list_SW2 = []

# iterate over each unique ID value
for id_value in ids_IBAP:
    # select the rows where the value of 'ID' is equal to the current ID value
    selected_rows = df_IBAP_r.loc[df_IBAP_r['POINT_ID'] == id_value, :]
    
    # select only the columns that end with '_col'
    selected_colsBLU = selected_rows.filter(regex='_BLU$', axis=1)
    selected_colsGRN = selected_rows.filter(regex='_GRN$', axis=1)
    selected_colsRED = selected_rows.filter(regex='_RED$', axis=1)
    selected_colsNIR = selected_rows.filter(regex='_NIR$', axis=1)
    selected_colsSW1 = selected_rows.filter(regex='_SW1$', axis=1)
    selected_colsSW2 = selected_rows.filter(regex='_SW2$', axis=1)

    # transpose the values from columns to rows using pd.melt()
    meltedBLU = pd.melt(selected_colsBLU, value_vars = selected_colsBLU.columns, var_name ='Orig_Col_BLU', value_name ='BLU')
    meltedGRN = pd.melt(selected_colsGRN, value_vars = selected_colsGRN.columns, var_name ='Orig_Col_GRN', value_name ='GRN')
    meltedRED = pd.melt(selected_colsRED, value_vars = selected_colsRED.columns, var_name ='Orig_Col_RED', value_name ='RED')
    meltedNIR = pd.melt(selected_colsNIR, value_vars = selected_colsNIR.columns, var_name ='Orig_Col_NIR', value_name ='NIR')
    meltedSW1 = pd.melt(selected_colsSW1, value_vars = selected_colsSW1.columns, var_name ='Orig_Col_SW1', value_name ='SW1')
    meltedSW2 = pd.melt(selected_colsSW2, value_vars = selected_colsSW2.columns, var_name ='Orig_Col_SW2', value_name ='SW2')

    # add the 'ID' column to the melted DataFrame
    meltedBLU['POINT_ID'] = id_value
    meltedGRN['POINT_ID'] = id_value
    meltedRED['POINT_ID'] = id_value
    meltedNIR['POINT_ID'] = id_value
    meltedSW1['POINT_ID'] = id_value
    meltedSW2['POINT_ID'] = id_value
    
    # add the result to the list of results
    list_BLU.append(meltedBLU)
    list_GRN.append(meltedGRN)
    list_RED.append(meltedRED)
    list_NIR.append(meltedNIR)
    list_SW1.append(meltedSW1)
    list_SW2.append(meltedSW2)

# concatenate the list of results into a single DataFrame
result_BLU = pd.concat(list_BLU, axis=0, ignore_index=True)
result_GRN = pd.concat(list_GRN, axis=0, ignore_index=True)
result_RED = pd.concat(list_RED, axis=0, ignore_index=True)
result_NIR = pd.concat(list_NIR, axis=0, ignore_index=True)
result_SW1 = pd.concat(list_SW1, axis=0, ignore_index=True)
result_SW2 = pd.concat(list_SW2, axis=0, ignore_index=True)

all_landsat_bands = pd.concat([result_BLU, result_GRN, result_RED, result_NIR, result_SW1, result_SW2], axis=1)
all_landsat_bands.to_csv('/path/to/non_forest_dataset_alltiles_landsatbands.csv')

print ("Landsat Bands added to dataframe in order year+ ID plot and exported")

#### DO THE SAME FOR INDICES ####

# select indices columns first to then split in 6 columns -bands
df_NBR = df_non_forest .loc[:, df_non_forest .columns.str.endswith('NBR')]
df_NDVI = df_non_forest .loc[:, df_non_forest .columns.str.endswith('NDVI')]
df_TCB = df_non_forest .loc[:, df_non_forest .columns.str.endswith('TCB')]
df_TCG = df_non_forest .loc[:, df_non_forest .columns.str.endswith('TCG')]
df_TCW = df_non_forest .loc[:, df_non_forest .columns.str.endswith('TCW')]
df_DIn = df_non_forest .loc[:, df_non_forest .columns.str.endswith('DIn')]

# select column 'plotid' by name
df_Index_id = df_non_forest ['POINT_ID']

df_indices = pd.concat([df_NBR, df_NDVI, df_TCB, df_TCG, df_TCW, df_DIn, df_Index_id], axis=1)

# select column 'plotid' by name
ids_index = df_non_forest ['POINT_ID'].unique()

# create a list to store the results
list_NBR = []
list_NDVI = []
list_TCB = []
list_TCG = []
list_TCW = []
list_DIn = []

# iterate over each unique ID value
for id_value in ids_index:
    # select the rows where the value of 'ID' is equal to the current ID value
    selected_rowsIND  = df_indices.loc[df_NBR['POINT_ID'] == id_value, :]
  
    # select only the columns that end with '_col'
    selected_colsNBR = selected_rowsIND.filter(regex='_NBR$', axis=1)
    selected_colsNDVI = selected_rowsIND.filter(regex='_NDVI$', axis=1)
    selected_colsTCB = selected_rowsIND.filter(regex='_TCB$', axis=1)
    selected_colsTCG = selected_rowsIND.filter(regex='_TCG$', axis=1)
    selected_colsTCW = selected_rowsIND.filter(regex='_TCW$', axis=1)
    selected_colsDIn = selected_rowsIND.filter(regex='_DIn$', axis=1)

    # transpose the values from columns to rows using pd.melt()
    meltedNBR  = pd.melt(selected_colsNBR , value_vars = selected_colsNBR.columns, var_name ='Orig_Col_NBR', value_name ='NBR')
    meltedNDVI  = pd.melt(selected_colsNDVI , value_vars = selected_colsNDVI.columns, var_name ='Orig_Col_NDVI', value_name ='NDVI')
    meltedTCB  = pd.melt(selected_colsTCB , value_vars = selected_colsTCB.columns, var_name ='Orig_Col_TCB', value_name ='TCB')
    meltedTCG  = pd.melt(selected_colsTCG , value_vars = selected_colsTCG.columns, var_name ='Orig_Col_TCG', value_name ='TCG')
    meltedTCW  = pd.melt(selected_colsTCW , value_vars = selected_colsTCW.columns, var_name ='Orig_Col_TCW', value_name ='TCW')
    meltedDIn  = pd.melt(selected_colsDIn , value_vars = selected_colsDIn.columns, var_name ='Orig_Col_DIn', value_name ='DIn')

    # add the 'ID' column to the melted DataFrame
    meltedNBR ['POINT_ID'] = id_value
    meltedNDVI ['POINT_ID'] = id_value
    meltedTCB ['POINT_ID'] = id_value
    meltedTCG ['POINT_ID'] = id_value
    meltedTCW ['POINT_ID'] = id_value
    meltedDIn ['POINT_ID'] = id_value
    
    # add the result to the list of results
    list_NBR.append(meltedNBR)
    list_NDVI.append(meltedNDVI)
    list_TCB.append(meltedTCB)
    list_TCG.append(meltedTCG)
    list_TCW.append(meltedTCW)
    list_DIn.append(meltedDIn)

# concatenate the list of results into a single DataFrame
result_NBR = pd.concat(list_NBR, axis=0, ignore_index=True)
result_NDVI = pd.concat(list_NDVI, axis=0, ignore_index=True)
result_TCB = pd.concat(list_TCB, axis=0, ignore_index=True)
result_TCG = pd.concat(list_TCG, axis=0, ignore_index=True)
result_TCW = pd.concat(list_TCW, axis=0, ignore_index=True)
result_DIn = pd.concat(list_DIn, axis=0, ignore_index=True)
                          
all_landsat_Indices = pd.concat([result_NBR, result_NDVI, result_TCB, result_TCG, result_TCW, result_DIn], axis=1)
all_landsat_Indices.to_csv('/path/to/non_forest_dataset_alltiles_landsatIndices.csv')