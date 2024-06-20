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
CARB_EN_MATS_PATH = os.path.join(data_dir, 'model/CarbEnMats_dataset.xlsx')
BECD_PATH = os.path.join(data_dir, 'model/BECD_2024-06-17 18.41.17.csv')

clf_df = pd.read_csv(CLF_EMBODIED_CARBON_PATH)
carbenmats_df = pd.read_excel(CARB_EN_MATS_PATH)
becd_df = pd.read_csv(BECD_PATH)


"""
2. Align and Merge Datasets
"""

# Calculate the total embodied carbon for BECD
becd_df['Total_Embodied_Carbon'] = becd_df[
    ['Total_Normalised_A1ToA3', 'Total_Normalised_A4', 'Total_Normalised_A5',
     'Total_Normalised_B1', 'Total_Normalised_B2', 'Total_Normalised_B3',
     'Total_Normalised_B4', 'Total_Normalised_B5', 'Total_Normalised_C1',
     'Total_Normalised_C2', 'Total_Normalised_C3', 'Total_Normalised_C4',
     'Total_Normalised_D']
].sum(axis=1)

# Calculate the total embodied carbon for CarbEnMats
carbenmats_df['Total_Embodied_Carbon'] = carbenmats_df[
    ['GHG_A123_m2a', 'GHG_A45_m2a', 'GHG_B1234_m2a', 'GHG_B5_m2a',
     'GHG_B67_m2a', 'GHG_C12_m2a', 'GHG_C34_m2a', 'GHG_D_m2a']
].sum(axis=1)

# Function to split the interval
def split_interval(interval):
    if interval == 'No data':
        return pd.Series([None, None])
    try:
        min_val, max_val = map(int, interval.split('-'))
        return pd.Series([min_val, max_val])
    except ValueError:
        return pd.Series([None, None])

# Rename columns to align with clf_df
becd_df.rename(columns={'EntityCode': 'Building Public ID', 'ProjectType': 'Building New or Renovation',
                        'Location': 'Building Location Region', 'SizePrimary': 'BuildingAreaExact SquareMeters', 
                        'Total_Normalised_TotalBiogenicCarbon': 'Total_Biogenic_Carbon'}, inplace=True) 
        # need to also change all new/reno names correctly.

carbenmats_df.rename(columns={'site_country': 'Country', 'bldg_use_type': 'Building Type', 'bldg_use_subtype': 'Building Use', 
                              'site_region_world': 'Building Location Region', 'bldg_project_type': 'Building New or Renovation' }, inplace=True)
carbenmats_df[['Minimum Building Area in Square Meters', 'Maximum Building Area in Square Meters']] = carbenmats_df['bldg_area_interval'].apply(split_interval)
carbenmats_df[['Minimum Building Storeys', 'Maximum Building Storeys']] = carbenmats_df['bldg_floors_ag_interval'].apply(split_interval)
        # need to change all non-residential in building use type to 'commercial'
        # need to change all "multi-family house...." to just "multi-family" in building use, do the same for single-family also
        #remove any "No data" etc

clf_df.rename(columns={'Embodied Carbon Whole Building Excluding Operational': 'Total_Embodied_Carbon'}, inplace=True)
# change building location region to continents

# Define a function to combine columns
def combine_columns(df, column_map):
    for new_col, old_cols in column_map.items():
        if old_cols:  # Ensure there's something to sum
            df[new_col] = df[old_cols].sum(axis=1)
        else:  # If no columns to sum, create the column with NaNs
            df[new_col] = pd.Series([float('nan')] * len(df), index=df.index)
    return df

# Define the column mappings
becd_column_map = {
    'A123': ['Total_Normalised_A1ToA3'],
    'A45': ['Total_Normalised_A4', 'Total_Normalised_A5'],
    'B1234': ['Total_Normalised_B1', 'Total_Normalised_B2', 'Total_Normalised_B3', 'Total_Normalised_B4'],
    'B5': ['Total_Normalised_B5'],
    'B67': [],  # No equivalent in BECD
    'C12': ['Total_Normalised_C1', 'Total_Normalised_C2'],
    'C34': ['Total_Normalised_C3', 'Total_Normalised_C4'],
    'D': ['Total_Normalised_D']
}

carbenmats_column_map = {
    'A123': ['GHG_A123_m2a'],
    'A45': ['GHG_A45_m2a'],
    'B1234': ['GHG_B1234_m2a'],
    'B5': ['GHG_B5_m2a'],
    'B67': ['GHG_B67_m2a'],
    'C12': ['GHG_C12_m2a'],
    'C34': ['GHG_C34_m2a'],
    'D': ['GHG_D_m2a']
}

# Apply the column mappings to create combined columns
becd_df_combined = combine_columns(becd_df.copy(), becd_column_map)
carbenmats_df_combined = combine_columns(carbenmats_df.copy(), carbenmats_column_map)

# Align columns
all_columns = set(becd_df_combined.columns).union(set(carbenmats_df_combined.columns)).union(set(clf_df.columns))
becd_df_combined = becd_df_combined.reindex(columns=all_columns)
carbenmats_df_combined = carbenmats_df_combined.reindex(columns=all_columns)
clf_df = clf_df.reindex(columns=all_columns)

# Add a column to indicate the dataset source
becd_df_combined['Dataset'] = 'BECD'
carbenmats_df_combined['Dataset'] = 'CarbEnMats'
clf_df['Dataset'] = 'CLF'

# Concatenate dataframes
merged_df = pd.concat([becd_df_combined, carbenmats_df_combined, clf_df], ignore_index=True)


"""
3. Select and Rename relevant columns
"""

# Create main dataframe
df = merged_df[['Country', 'Building Location Region', 'Building Type', 'Building Use', 'Building New or Renovation', 'bldg_users_total', 
                       'Minimum Building Area in Square Meters', 'Maximum Building Area in Square Meters', 'BuildingAreaExact SquareMeters',
                       'Minimum Building Storeys', 'Maximum Building Storeys', 'AboveGroundStorey', 'UndergroundStorey', 'Total_Biogenic_Carbon',

                       'mass_wood', 'mass_straw_hemp', 'mass_fungi', 'mass_brass_copper', 'mass_earth',
                       'mass_bamboo', 'mass_glass', 'mass_stone', 'mass_stone_wool', 'mass_ceramics',
                       'mass_metals', 'mass_plastics', 'mass_steel_reinforcement', 'mass_EPS_XPS', 'mass_aluminium', 
                       'mass_concrete_wo_reinforcement', 'mass_other', 'mass_concrete_reinforced', 'mass_cement_mortar', 'mass_total_mats'
,
                       'PSCHorizontalElementTypePrimary', 'PSCVerticalElementStructureTypePrimary', 'PSCFinishesTypePrimary', 
                       'PSCCladdingTypePrimary', 'PSCFoundationTypePrimary', 'PSCGroundFloorTypePrimary', 
                       'PSCHeatingTypePrimary', 'PSCCoolingTypePrimary', 'PSCVentilationTypePrimary',
                       
                       'Total_Embodied_Carbon']]

# Rename columns to be clearer
df = df.rename(columns={
    'Country': 'Country_Name',
    'Building Location Region': 'Country_Region',
    'Building Type': 'Bldg_Type',
    'Building Use': 'Bldg_Use',
    'Building New or Renovation': 'Construction_Type',
    'bldg_users_total': 'Total_Bldg_Users',
    'AboveGroundStorey': 'Exact_Storeys',
    'UndergroundStorey': 'Underground_Storeys',
    'Minimum Building Area in Square Meters': 'Min_Area_SqMeters',
    'Maximum Building Area in Square Meters': 'Max_Area_SqMeters',
    'BuildingAreaExact SquareMeters': 'Exact_Area_SqMeters',
    'Minimum Building Storeys': 'Min_Storeys',
    'Maximum Building Storeys': 'Max_Storeys',
    'Total_Biogenic_Carbon': 'Total_Sequestered_Carbon',
    'mass_wood': 'Mass_Wood',
    'mass_straw_hemp': 'Mass_Straw_Hemp',
    'mass_fungi': 'Mass_Fungi',
    'mass_brass_copper': 'Mass_Brass_Copper',
    'mass_earth': 'Mass_Earth',
    'mass_bamboo': 'Mass_Bamboo',
    'mass_glass': 'Mass_Glass',
    'mass_stone': 'Mass_Stone',
    'mass_stone_wool': 'Mass_Stone_Wool',
    'mass_total_mats': 'Total_Materials_Mass',
    'mass_ceramics': 'Mass_Ceramics',
    'mass_metals': 'Mass_Metals',
    'mass_plastics': 'Mass_Plastics',
    'mass_steel_reinforcement': 'Mass_Steel_Reinforcement',
    'mass_EPS_XPS': 'Mass_EPS_XPS',
    'mass_aluminium': 'Mass_Aluminium',
    'mass_concrete_wo_reinforcement': 'Mass_Concrete_Without_Reinforcement',
    'mass_other': 'Mass_Other',
    'mass_concrete_reinforced': 'Mass_Reinforced_Concrete',
    'mass_cement_mortar': 'Mass_Cement_Mortar',
    'PSCHorizontalElementTypePrimary': 'Primary_Horizontal_Element_Type',
    'PSCVerticalElementStructureTypePrimary': 'Primary_Vertical_Element_Type',
    'PSCFinishesTypePrimary': 'Primary_Finishes_Type',
    'PSCCladdingTypePrimary': 'Primary_Cladding_Type',
    'PSCFoundationTypePrimary': 'Primary_Foundation_Type',
    'PSCGroundFloorTypePrimary': 'Primary_Ground_Floor_Type',
    'PSCHeatingTypePrimary': 'Primary_Heating_Type',
    'PSCCoolingTypePrimary': 'Primary_Cooling_Type',
    'PSCVentilationTypePrimary': 'Primary_Ventilation_Type',
    'Total_Embodied_Carbon': 'Total_Embodied_Carbon'
})


"""
4. Clean Datasets
"""

# Cleaning:


######
# 1. Impute countries, taking region into account.
# 2. Find region from countries.

COUNTRIES = os.path.join(data_dir, 'preprocess/countries_continents.csv')
country_df = pd.read_csv(COUNTRIES)
country_df = country_df.rename(columns={'Country': 'Country_Name'})

df['Country_Name'] = df['Country_Name'].str.lower()
country_df['Country_Name'] = country_df['Country_Name'].str.lower()

# Function to capitalize the first letter of each word
def capitalize_words(name):
    if pd.isnull(name):
        return None
    return ' '.join(word.capitalize() for word in name.split())

df['Country_Name'] = df['Country_Name'].apply(capitalize_words)                     # Apply the function to the 'Country_Name' columns
country_df['Country_Name'] = country_df['Country_Name'].apply(capitalize_words)
df['Country_Name'] = df['Country_Name'].replace('Usa', 'United States')             # Rename "Usa" to "United States"

regions = country_df['Continent'].unique()                                          # Identify rows where 'Country_Name' matches any 'Country_Region'

def move_region_to_country(row):
    if row['Country_Name'] in regions:
        if pd.isnull(row['Country_Region']):
            row['Country_Region'] = row['Country_Name']
        row['Country_Name'] = None
    return row

df = df.apply(move_region_to_country, axis=1)                                       # Apply the function to move region names
region_mapping = {                                                                  # Dictionary to standardize country regions
    'Asia-Pacific': 'Asia',
    'Middle East, North Africa, and Greater Arabia': 'Middle East',
    'Europe': 'Europe',
    'North America': 'North America',
    'South America': 'South America',
    'Oceania': 'Oceania',
    'Africa': 'Africa',
    'Middle East': 'Middle East'
}
df['Country_Region'] = df['Country_Region'].replace(region_mapping)                 # Standardize the 'Country_Region' column

df = df.merge(country_df, on='Country_Name', how='left')                            # Merge the DataFrames
df['Country_Region'] = df['Country_Region'].combine_first(df['Continent'])          # Replace NaNs in 'Country_Region' with values from 'Country_Region_x' and 'Country_Region_y'
df = df.drop(columns=['Continent'])                                                 # Drop unnecessary columns


######
# 3. Change all "Non-residential" to "Commercial"
df['Bldg_Type'] = df['Bldg_Type'].replace('Non-residential', 'Commercial')
def update_bldg_type(row):                                                          # Function to update Bldg_Type based on Bldg_Use
    commercial_uses = [
        'Park', 'Parking', 'Public Assembly', 'Public Order and Safety', 
        'Industrial', 'Office', 'Retail', 'Service', 'Mixed Use'
    ]
    if row['Bldg_Use'] in commercial_uses:
        row['Bldg_Type'] = 'Commercial'
    elif row['Bldg_Use'] == 'Residential':
        row['Bldg_Type'] = 'Residential'
    elif row['Bldg_Type'] in ['NonCommercial', 'Generic', 'No data']:
        row['Bldg_Type'] = None
    return row
df = df.apply(update_bldg_type, axis=1)                                             # Apply the function to update Bldg_Type

renovation_values = [                                                               # Mapping for Construction_Type column
    'Existing building', 'Extension and retrofit', 
    'Refurbishment', 'Renovation', 'Retrofit'
]
new_values = ['New', 'New built', 'New construction', 'Demolish and new build']
def update_construction_type(value):
    if value in renovation_values:
        return 'Renovation'
    elif value in new_values:
        return 'New'
    else:
        return None
df['Construction_Type'] = df['Construction_Type'].apply(update_construction_type)    # Apply the function to update Construction_Type


#######
# 4. Add higher band for inf in max storeys and area, use avg of minmax for missing exact vals

# Convert relevant columns to numeric
df['Min_Storeys'] = pd.to_numeric(df['Min_Storeys'], errors='coerce')
df['Max_Storeys'] = pd.to_numeric(df['Max_Storeys'], errors='coerce')
df['Exact_Storeys'] = pd.to_numeric(df['Exact_Storeys'], errors='coerce')
df['Min_Area_SqMeters'] = pd.to_numeric(df['Min_Area_SqMeters'], errors='coerce')
df['Max_Area_SqMeters'] = pd.to_numeric(df['Max_Area_SqMeters'], errors='coerce')
df['Exact_Area_SqMeters'] = pd.to_numeric(df['Exact_Area_SqMeters'], errors='coerce')
df['Underground_Storeys'] = pd.to_numeric(df['Underground_Storeys'], errors='coerce')

# Replace 'inf' in Max_Storeys column with 100
df['Max_Storeys'] = df['Max_Storeys'].replace([np.inf, 'inf'], 100)

# Replace 'inf' in Max_Area_SqMeters column with 100
df['Max_Area_SqMeters'] = df['Max_Area_SqMeters'].replace([np.inf, 'inf'], 1000000)

# Replace values in Exact_Storeys greater than 200 with NaN
df.loc[df['Exact_Storeys'] > 200, 'Exact_Storeys'] = np.nan
# Replace values in Underground_Storeys greater than 20 with NaN
df.loc[df['Underground_Storeys'] > 20, 'Underground_Storeys'] = np.nan

# Fill missing Exact_Storeys with the median of Min_Storeys and Max_Storeys
df['Exact_Storeys'] = df.apply(
    lambda row: (row['Min_Storeys'] + row['Max_Storeys']) / 2 
    if np.isnan(row['Exact_Storeys']) else row['Exact_Storeys'], axis=1
    )
# Fill missing Exact_Area_SqMeters with the median of Min_Area_SqMeters and Max_Area_SqMeters
df['Exact_Area_SqMeters'] = df.apply(
    lambda row: (row['Min_Area_SqMeters'] + row['Max_Area_SqMeters']) / 2 
    if np.isnan(row['Exact_Area_SqMeters']) else row['Exact_Area_SqMeters'], axis=1
    )

# Replace 0s in Max_Storeys column with nan
df['Exact_Storeys'] = df['Exact_Storeys'].replace(0, np.nan)

# Remove the value in Underground_Storeys if Exact_Storeys is still NaN
df.loc[df['Exact_Storeys'].isna(), 'Underground_Storeys'] = np.nan

# Drop Min and Max columns
df.drop(columns=['Min_Storeys', 'Max_Storeys', 'Min_Area_SqMeters', 'Max_Area_SqMeters'], inplace=True)


#######
# 5. Remove all entries that are "N/A" or "No data"

# List of values to be replaced with empty strings
values_to_replace = ["No data", "no data", "Not applicable", "Not Applicable", "NA", "N/A", "Unknown"]
# Replace the specified values with empty strings
df.replace(values_to_replace, '', inplace=True)                                       


######
# 6. For all masses that don't have a value, input 0.
mass_columns = ['Mass_Wood', 'Mass_Straw_Hemp', 'Mass_Fungi', 'Mass_Brass_Copper', 'Mass_Earth', 
                'Mass_Bamboo', 'Mass_Glass', 'Mass_Stone', 'Mass_Stone_Wool', 'Mass_Ceramics',
                'Mass_Metals', 'Mass_Plastics', 'Mass_Steel_Reinforcement', 'Mass_EPS_XPS', 'Mass_Aluminium',
                'Mass_Concrete_Without_Reinforcement', 'Mass_Other', 'Mass_Reinforced_Concrete', 'Mass_Cement_Mortar', 
                'Total_Materials_Mass']
# Fill missing mass values with zero
df[mass_columns] = df[mass_columns].fillna(0)

# Identify rows where all mass columns are zero
all_mass_zero = df[mass_columns].sum(axis=1) == 0

# Set mass values to NaN for rows where all mass columns are zero
df.loc[all_mass_zero, mass_columns] = np.nan



######
# 7.  Any biogenic with no data should be 0. Any carbon stages with no data should be 0.
biogenic_columns = ['Total_Sequestered_Carbon']

# Function to replace "no data" values with 0
def replace_no_data_with_zero(column):                                                
    df[column] = df[column].replace(['No data', 'no data', 'Not applicable', 'Not Applicable', 'NA', 'N/A', None, np.nan], 0).astype(float)
# Replace "no data" values with 0 for biogenic columns
for column in biogenic_columns:
    replace_no_data_with_zero(column)

######
# 8. Total carbon should include total sequestered (total accrued - total sequestered = actual total)
# Make all sequestered carbon values negative
df['Total_Sequestered_Carbon'] = df['Total_Sequestered_Carbon'].apply(lambda x: -abs(x))
# Calculate the actual total carbon
df['Actual_Total_Carbon'] = df['Total_Embodied_Carbon'] + df['Total_Sequestered_Carbon']
# Drop sequestered carbon column
df.drop(columns=['Total_Sequestered_Carbon'], inplace=True)


######
# 9. Remove all entries with total carbon as 0.
df = df[df['Actual_Total_Carbon'] != 0]



######
# 10. Remove outliers from dataset

# Define a function to identify outliers using Z-score
def identify_outliers_zscore(data, threshold=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    return np.where(np.abs(z_scores) > threshold)

# Identify outliers
outlier_indices = identify_outliers_zscore(df['Actual_Total_Carbon'], threshold=3)

# Print the outliers
outliers = df.iloc[outlier_indices]
#print("Outliers identified using Z-score method:")
#print(outliers)

# Remove outliers using a boolean mask
mask = np.ones(len(df), dtype=bool)
mask[outlier_indices] = False
df = df[mask]


# Drop the 'Total_Embodied_Carbon' column if no longer needed
df = df.drop(columns=['Total_Embodied_Carbon'])


"""
5. Export data to CSV
"""

# Save the merged dataframe to a CSV file
DF_PATH = os.path.join(export_dir, 'BUILDING_DATA.csv')
df.to_csv(DF_PATH, index=False)
df.info()