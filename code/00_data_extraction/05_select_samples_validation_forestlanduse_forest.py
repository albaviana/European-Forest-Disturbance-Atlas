#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Script to select validation samples for independent forestlanduse accuracy assessment
# Unselected samples will be used to model calibration
# see Table 1 in preprint and section 2.2 for detailed information

import pandas as pd
import random

data = pd.read_csv('/path/to/forest_dataset_alltiles_landsatdata.csv', sep=';', low_memory=False)

## new to take 2000 plots in forest areas
plots_per_country = {
    'albania': 3, 'austria': 17, 'belarus': 35,'belgium': 3, 'bosniaherzegovina': 11,'bulgaria': 16,
    'croatia': 11, 'czechia': 11,'denmark': 3,'estonia': 10, 'finland': 101, 'france': 107, 'germany': 50, 'greece': 16,
    'hungary': 9,'ireland': 3,'italy': 46,'latvia': 12,'lithuania': 9,'moldova': 1,'montenegro': 3,
    'netherlands': 2,'norway': 53,'poland': 39,'portugal': 14,'romania': 30,'serbia': 12,'slovakia': 9,
    'slovenia': 6,'spain': 80,'sweden': 122,'switzerland': 5,'macedonia': 4,'ukraine': 46,'unitedkingdom': 13,
    # Add more countries and corresponding plot counts here
}

# Create an empty DataFrame to store the selected data
selected_data = pd.DataFrame()
# Create an empty DataFrame to store the unselected data that will be used to model calibration
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
selected_data.to_csv('/path/to/forest_dataset_alltiles_landsatdata_samples_landusevalidation.csv', index=False)
# Save the unselected data to a new CSV file
unselected_data.to_csv('/path/to/forest_dataset_alltiles_landsatdata_samples_landusecalibration.csv', index=False)