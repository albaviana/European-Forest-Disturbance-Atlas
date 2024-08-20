#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:47:07 2024

@author: aviana
"""

import pandas as pd
import random

# Read the CSV file into a DataFrame
data = pd.read_csv('/path/to/forest_dataset_alltiles_landsatdata_DIFFS.csv', sep=';', low_memory=False)

## new to take 2500 plots in forest areas
plots_per_country = {
    'albania': 9, 'austria': 47, 'belarus': 96,'belgium': 8, 'bosniaherzegovina': 31,'bulgaria': 43,
    'croatia': 30, 'czechia': 31,'denmark': 7,'estonia': 27, 'finland': 278, 'france': 294, 'germany': 136, 'greece': 45,
    'hungary': 25,'ireland': 9,'italy': 127,'latvia': 34,'lithuania': 25,'moldova': 4,'montenegro': 7,
    'netherlands': 4,'norway': 145,'poland': 107,'portugal': 38,'romania': 83,'serbia': 32,'slovakia': 24,
    'slovenia': 15,'spain': 219,'sweden': 335,'switzerland': 15,'macedonia': 12,'ukraine': 124,'unitedkingdom': 34,
    # Add more countries and corresponding plot counts here
}

# Create an empty DataFrame to store the selected data
selected_data = pd.DataFrame()
# Create an empty DataFrame to store the unselected data
unselected_data = pd.DataFrame()

# Group the data by the "country" column
grouped = data.groupby('country')

# Loop through each country and select the specified number of random plots
for country, num_plots in plots_per_country.items():
    if country in grouped.groups:
        country_data = grouped.get_group(country)
        
        # Randomly select plots from eligible plots
        selected_plotids = random.sample(eligible_plots, min(num_plots, len(eligible_plots)))
        
        # Select the rows for the selected plotids
        selected_plots = country_data[country_data['plotid'].isin(selected_plotids)]
        
        selected_data = pd.concat([selected_data, selected_plots])
        
        unselected_plotids = list(set(country_data['plotid']) - set(selected_plotids))
        unselected_plots = country_data[country_data['plotid'].isin(unselected_plotids)]
        unselected_data = pd.concat([unselected_data, unselected_plots])
        
# Save the selected data to a new CSV file
selected_data.to_csv('/path/to/forest_dataset_landsatdata_samples_disturbance_validation.csv', index=False)
# Save the unselected data to a new CSV file
unselected_data.to_csv('/path/to/forest_dataset_landsatdata_samples_disturbance_calibration.csv', index=False)