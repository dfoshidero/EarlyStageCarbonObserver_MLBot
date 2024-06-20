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

CARB_EN_MATS_PATH = os.path.join(data_dir, 'model/CarbEnMats_dataset.xlsx')

df = pd.read_excel(CARB_EN_MATS_PATH)

"""
2. Clean the Dataset
"""
# Define a list of values to be replaced with NaN
na_values = ["n/a", "N/a", "N/A", "No data", "no data"]
# Replace these values with NaN
df.replace(na_values, np.nan, inplace=True)

# Save the merged dataframe to a CSV file
DF_PATH = os.path.join(export_dir, 'BUILDING_DATA.csv')
df.to_csv(DF_PATH, index=False)
df.info()

"""
3. Select and Rename relevant columns
"""

df = df[['bldg_project_type',
         'bldg_use_type',
         'bldg_use_subtype',
         'site_region_world',
         'site_country',
         'site_region_local',
         'bldg_area_gfa',
         'bldg_area_interval',
         'bldg_users_total',
         'bldg_floors_ag',
         'bldg_floors_ag_interval',
         'bldg_floors_bg',
         'bldg_struct_type',
         'bldg_roof_type',

         ]]