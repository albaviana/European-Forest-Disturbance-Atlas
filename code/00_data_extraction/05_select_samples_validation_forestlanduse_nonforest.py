#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aviana
"""

# Script to select validation samples (within non-forest LUCAS dataset) for independent forestlanduse accuracy assessment
# Unselected samples will be used to model calibration
# see Table 1 in preprint and section 2.2 for detailed information

import pandas as pd
import random

# Read the CSV file into a DataFrame
data = pd.read_csv('/path/to/non_forest_dataset_alltiles_landsatdata.csv', sep=';')

# Define the number of plots per country
plots_per_country = {
    'AT': 20, 'BE':10 ,'BG': 32,'HR': 14,'CZ': 23,'DK':16, 'EE': 10, 'FI': 46,'FR': 132,'DE': 105, 'GR': 38,'HU': 31,
    'IE': 28,'IT': 85, 'LV': 16,'LT': 19,'NL':13,'PL':97,'PT':25, 'RO':74, 'SK':12,'SI':3,'ES':137,'SE':74,'UK': 94
    # Add more countries and corresponding plot counts here
}

# Create an empty DataFrame to store the selected data for independent validation
selected_data = pd.DataFrame()
# Create an empty DataFrame to store the unselected data that will be used to model calibration
unselected_data = pd.DataFrame()

# Group the data by the "country" column from lucas - nonforest dataset
grouped = data.groupby('NUTS0')

# Loop through each country and select the specified number of random plots
for country, num_plots in plots_per_country.items():
    if country in grouped.groups:
        country_data = grouped.get_group(country)
        
        # Randomly select plots from eligible plots
        selected_plotids = random.sample(eligible_plots, min(num_plots, len(eligible_plots)))
        
        # Select the rows for the selected plotids
        selected_plots = country_data[country_data['POINT_ID'].isin(selected_plotids)]
        selected_data = pd.concat([selected_data, selected_plots])
        
        unselected_plotids = list(set(country_data['POINT_ID']) - set(selected_plotids))
        unselected_plots = country_data[country_data['POINT_ID'].isin(unselected_plotids)]
        unselected_data = pd.concat([unselected_data, unselected_plots])
     
# Save the selected data to a new CSV file
selected_data.to_csv('/path/to/non_forest_dataset_alltiles_landsatdata_samples_flandusevalidation.csv', index=False)
# Save the unselected data to a new CSV file
unselected_data.to_csv('/path/to/non_forest_dataset_alltiles_landsatdata_samples_flandusecalibration.csv', index=False)