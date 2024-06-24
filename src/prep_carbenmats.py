import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Enable the IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../data/raw')
export_dir = os.path.join(current_dir, '../data/processed')

os.makedirs(export_dir, exist_ok=True)

CARB_EN_MATS_PATH = os.path.join(data_dir, 'model/CarbEnMats_dataset.xlsx')

carbenmats_df = pd.read_excel(CARB_EN_MATS_PATH)

"""
2. Clean the Dataset
"""

# Define a list of values to be replaced with NaN
na_values = ["n/a", "N/a", "N/A", "No data", "no data"]
# Replace these values with NaN
carbenmats_df.replace(na_values, np.nan, inplace=True)

# Define a function to calculate the median from an interval
def median_from_interval(interval):
    if pd.isna(interval):
        return np.nan
    try:
        start, end = map(float, interval.split('-'))
        return (start + end) / 2
    except:
        return np.nan

# Fill missing values
carbenmats_df['bldg_area_gfa'] = carbenmats_df['bldg_area_gfa'].fillna(carbenmats_df['bldg_area_interval'].apply(median_from_interval)) # Fill missing areas with median from interval
carbenmats_df['bldg_floors_ag'] = carbenmats_df['bldg_floors_ag'].fillna(carbenmats_df['bldg_floors_ag_interval'].apply(median_from_interval)) # Fill missing storeys with median from interval
carbenmats_df['bldg_users_total'] = carbenmats_df['bldg_users_total'].fillna(np.nan) # Fill missing with nan
carbenmats_df['bldg_floors_bg'] = carbenmats_df['bldg_floors_bg'].fillna(np.nan) # Fill missing with nan

"""
3. Fill and impute missing values.
"""
# Define categorical and numerical columns
categorical_cols = ['bldg_project_type', 'bldg_use_type', 'bldg_use_subtype', 'site_region_world', 'site_country', 'site_region_local', 'bldg_struct_type', 'bldg_roof_type']
numerical_cols = ['bldg_area_gfa', 'bldg_users_total', 'bldg_floors_ag', 'bldg_floors_bg']

# Fill missing values in categorical columns with 'missing'
imp_categorical = SimpleImputer(strategy='constant', fill_value='missing')
carbenmats_df[categorical_cols] = imp_categorical.fit_transform(carbenmats_df[categorical_cols])

# Convert categorical columns to numerical values using LabelEncoder
encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    carbenmats_df[col] = encoders[col].fit_transform(carbenmats_df[col].astype(str))

# Use Iterative Imputer for numerical columns
imp_numerical = IterativeImputer(max_iter=10, random_state=0)
carbenmats_df[numerical_cols] = imp_numerical.fit_transform(carbenmats_df[numerical_cols])

# Convert categorical columns back to their original data types
for col in categorical_cols:
    carbenmats_df[col] = encoders[col].inverse_transform(carbenmats_df[col].astype(int))

mass_columns = [
    'mass_wood', 'mass_straw_hemp', 'mass_fungi', 'mass_brass_copper', 'mass_earth',
    'mass_bamboo', 'mass_glass', 'mass_stone', 'mass_stone_wool', 'mass_ceramics',
    'mass_metals', 'mass_plastics', 'mass_steel_reinforcement', 'mass_EPS_XPS', 'mass_aluminium', 
    'mass_concrete_wo_reinforcement', 'mass_other', 'mass_concrete_reinforced', 'mass_cement_mortar',
]

# Fill missing mass values with zero
carbenmats_df[mass_columns] = carbenmats_df[mass_columns].fillna(0)

# Identify rows where all mass columns are zero
all_mass_zero = carbenmats_df[mass_columns].sum(axis=1) == 0

# Set mass values to NaN for rows where all mass columns are zero
carbenmats_df.loc[all_mass_zero, mass_columns] = np.nan

"""
4. Select and Rename relevant columns
"""
# Calculate total carbon
ghg_columns = ['GHG_A123_m2a', 'GHG_A45_m2a', 'GHG_B1234_m2a', 'GHG_B5_m2a', 'GHG_B67_m2a', 'GHG_C12_m2a', 'GHG_C34_m2a', 'GHG_D_m2a']
carbenmats_df['Total_Embodied_Carbon'] = carbenmats_df[ghg_columns].sum(axis=1)

# Calculate kg CO2e per square meter - need to multiply by reference study period, as values are stored as per m2 per year
carbenmats_df['Total_Embodied_Carbon_PER_m2'] = carbenmats_df['Total_Embodied_Carbon'] * carbenmats_df['lca_RSP']

carbenmats_df = carbenmats_df[['bldg_project_type', 'bldg_use_type', 'bldg_use_subtype', 'site_region_world', 'site_country',
     'site_region_local', 'bldg_area_gfa', 'bldg_users_total', 'bldg_floors_ag', 'bldg_floors_bg',
     'bldg_struct_type', 'bldg_roof_type',

     'mass_wood', 'mass_straw_hemp', 'mass_fungi', 'mass_brass_copper', 'mass_earth',
     'mass_bamboo', 'mass_glass', 'mass_stone', 'mass_stone_wool', 'mass_ceramics',
     'mass_metals', 'mass_plastics', 'mass_steel_reinforcement', 'mass_EPS_XPS', 'mass_aluminium', 
     'mass_concrete_wo_reinforcement', 'mass_concrete_reinforced', 'mass_cement_mortar', 'mass_other',

     'Total_Embodied_Carbon_PER_m2'
     ]]

# Rename columns for better inspection
carbenmats_df.rename(columns={
'bldg_project_type': 'Building_Project_Type',
'bldg_use_type': 'Building_Use_Type',
'bldg_use_subtype': 'Building_Use_Subtype',
'site_region_world': 'Continent',
'site_country': 'Country',
'site_region_local': 'City',
'bldg_area_gfa': 'Gross_Floor_Area_m2',
'bldg_users_total': 'Total_Users',
'bldg_floors_ag': 'Floors_Above_Ground',
'bldg_floors_bg': 'Floors_Below_Ground',
'bldg_struct_type': 'Structure_Type',
'bldg_roof_type': 'Roof_Type',
'mass_wood': 'Mass_Wood',
'mass_straw_hemp': 'Mass_Straw_Hemp',
'mass_fungi': 'Mass_Fungi',
'mass_brass_copper': 'Mass_Brass_Copper',
'mass_earth': 'Mass_Earth',
'mass_bamboo': 'Mass_Bamboo',
'mass_glass': 'Mass_Glass',
'mass_stone': 'Mass_Stone',
'mass_stone_wool': 'Mass_Stone_Wool',
'mass_ceramics': 'Mass_Ceramics',
'mass_metals': 'Mass_Metals',
'mass_plastics': 'Mass_Plastics',
'mass_steel_reinforcement': 'Mass_Steel_Reinforcement',
'mass_EPS_XPS': 'Mass_EPS_XPS',
'mass_aluminium': 'Mass_Aluminium',
'mass_concrete_wo_reinforcement': 'Mass_Concrete_Without_Reinforcement',
'mass_other': 'Mass_Other',
'mass_concrete_reinforced': 'Mass_Concrete_With_Reinforcement',
'mass_cement_mortar': 'Mass_Cement_Mortar',
}, inplace=True)

"""
5. Drop rows with any remaining NaN values, anddrop rows with "0" embodied carbon.
"""
carbenmats_df = carbenmats_df.dropna()

"""
6. Remove Outliers
"""
# Calculate Q1, Q3, and IQR for Total_Embodied_Carbon
Q1 = carbenmats_df['Total_Embodied_Carbon_PER_m2'].quantile(0.25)
Q3 = carbenmats_df['Total_Embodied_Carbon_PER_m2'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
carbenmats_df = carbenmats_df[(carbenmats_df['Total_Embodied_Carbon_PER_m2'] >= lower_bound) & (carbenmats_df['Total_Embodied_Carbon_PER_m2'] <= upper_bound)]

"""
7. Save Cleaned DataFrame to CSV
"""
# Save the cleaned dataframe to a CSV file
carbenmats_df_PATH = os.path.join(export_dir, 'cleaned_carbenmats.csv')
carbenmats_df.to_csv(carbenmats_df_PATH, index=False)
carbenmats_df.info()
