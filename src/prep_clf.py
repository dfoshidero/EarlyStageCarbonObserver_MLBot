"""
1. Import Libraries and Load Data   
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../data/raw')
export_dir = os.path.join(current_dir, '../data/processed')

os.makedirs(export_dir, exist_ok=True)

CLF_EMBODIED_CARBON_PATH = os.path.join(data_dir, 'model/CLF Embodied Carbon_Cleaned.csv')

clf_df = pd.read_csv(CLF_EMBODIED_CARBON_PATH)



"""
2. Clean the Datasets
"""
#### CLF
# Convert relevant columns to numeric, forcing errors to NaN
cols_to_convert = [
    'Minimum Building Area in Square Meters', 'Maximum Building Area in Square Meters',
    'Minimum Building Storeys', 'Maximum Building Storeys'
]

for col in cols_to_convert:
    clf_df[col] = pd.to_numeric(clf_df[col], errors='coerce')

# Replace np.inf and NaN with the maximum observed value in the respective columns
for col in cols_to_convert:
    clf_df[col].replace(np.inf, np.nan, inplace=True)
    max_value = clf_df[col].max(skipna=True)
    clf_df[col].fillna(max_value, inplace=True)

# Calculate average area and storeys
clf_df['Average Building Area in Square Meters'] = (clf_df['Minimum Building Area in Square Meters'] + clf_df['Maximum Building Area in Square Meters']) / 2
clf_df['Average Building Storeys'] = (clf_df['Minimum Building Storeys'] + clf_df['Maximum Building Storeys']) / 2

# Drop the minimum and maximum columns if they are no longer needed
clf_df = clf_df.drop(columns=[
    'Minimum Building Area in Square Meters', 'Maximum Building Area in Square Meters',
    'Minimum Building Storeys', 'Maximum Building Storeys'
])
# Select relevant columns
clf_df = clf_df[[
    "Building Type", "Building Use", "Building Location Region", "Building New or Renovation", 
    "Average Building Area in Square Meters", "Average Building Storeys", "Embodied Carbon Whole Building Excluding Operational" 
]]

clf_df = clf_df.dropna()

clf_df = clf_df.rename(columns={
    'Building Type': 'Building_Project_Type',
    'Building Use': 'Building_Use_Type',
    'Building Location Region': 'Continent',
    'Building New or Renovation': 'Building_Project_Type',
    'Average Building Area in Square Meters': 'Gross_Floor_Area_m2',
    'Average Building Storeys': 'Floors_Above_Ground',
    'Embodied Carbon Whole Building Excluding Operational': 'Total_Embodied_Carbon'
})

"""
3. Save dataframe to CSV for modelimg
"""
clf_df_PATH = os.path.join(export_dir, 'cleaned_clf.csv')
clf_df.to_csv(clf_df_PATH, index=False)
clf_df.info()

