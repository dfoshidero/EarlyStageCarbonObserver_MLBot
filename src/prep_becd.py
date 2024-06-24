import pandas as pd
import os
import numpy as np

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../data/raw')
export_dir = os.path.join(current_dir, '../data/processed')

os.makedirs(export_dir, exist_ok=True)

BECD_PATH = os.path.join(data_dir, 'model/BECD_2024-06-17 18.41.17.csv')

becd_df = pd.read_csv(BECD_PATH)

"""
2. Calculate Total Embodied Carbon
"""
# Calculate the total embodied carbon for BECD
becd_df['Total_Embodied_Carbon_PER_m2'] = becd_df[
    ['Total_Normalised_A1ToA3', 'Total_Normalised_A4', 'Total_Normalised_A5',
     'Total_Normalised_B1', 'Total_Normalised_B2', 'Total_Normalised_B3',
     'Total_Normalised_B4', 'Total_Normalised_B5', 'Total_Normalised_C1',
     'Total_Normalised_C2', 'Total_Normalised_C3', 'Total_Normalised_C4',
     'Total_Normalised_D']
].sum(axis=1)

"""
3. Drop Rows Where Total Embodied Carbon is Zero
"""
# Drop rows where Total_Embodied_Carbon is zero
becd_df = becd_df[becd_df['Total_Embodied_Carbon_PER_m2'] != 0]

"""
4. Select Relevant Columns
"""
# Select relevant columns
relevant_columns = [
    'ProjectType', 
    'PSCFoundationTypePrimary', 'PSCGroundFloorTypePrimary', 'PSCVerticalElementStructureTypePrimary', 'PSCHorizontalElementTypePrimary',
    'ProjectStageComponentsSlabTypePrimary', 'PSCCladdingTypePrimary', 'PSCHeatingTypePrimary', 'PSCCoolingTypePrimary',
    'PSCFinishesTypePrimary', 'PSCVentilationTypePrimary',
    'Total_Embodied_Carbon_PER_m2'
] # Only using primary material types.

becd_df = becd_df[relevant_columns]

"""
5. Rename Columns
"""
# Rename columns to be clearer
becd_df = becd_df.rename(columns={
    'ProjectType': 'Building_Project_Type',
    'PSCFoundationTypePrimary': 'Primary_Foundation_Type',
    'PSCGroundFloorTypePrimary': 'Primary_Ground_Floor_Type',
    'PSCVerticalElementStructureTypePrimary': 'Primary_Vertical_Element_Type',
    'PSCHorizontalElementTypePrimary': 'Primary_Horizontal_Element_Type',
    'ProjectStageComponentsSlabTypePrimary': 'Primary_Slab_Type',
    'PSCCladdingTypePrimary': 'Primary_Cladding_Type',
    'PSCHeatingTypePrimary': 'Primary_Heating_Type',
    'PSCCoolingTypePrimary': 'Primary_Cooling_Type',
    'PSCFinishesTypePrimary': 'Primary_Finishes_Type',
    'PSCVentilationTypePrimary': 'Primary_Ventilation_Type'
})

"""
6. Impute Missing Values
"""
# Impute missing values for Building_Project_Type and Country
becd_df['Building_Project_Type'].fillna('missing', inplace=True)

"""
7. Drop Rows with All 'Not Applicable' or 'Unknown' in Primary Columns
"""
# Drop rows where any of the Primary_ columns have all values as "Not applicable" or "Unknown"
values_to_check = ["Not applicable", "Unknown"]
primary_columns = [col for col in becd_df.columns if col.startswith('Primary_')]

# Create a mask to identify rows to drop
mask = becd_df[primary_columns].apply(lambda row: all(val in values_to_check for val in row), axis=1)
becd_df = becd_df[~mask]

"""
8. Drop Rows with Empty Cells
"""
# Drop columns with empty cells
becd_df = becd_df.dropna()

"""
9. Remove Outliers
"""
# Calculate Q1, Q3, and IQR for Total_Embodied_Carbon
Q1 = becd_df['Total_Embodied_Carbon_PER_m2'].quantile(0.25)
Q3 = becd_df['Total_Embodied_Carbon_PER_m2'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
becd_df = becd_df[(becd_df['Total_Embodied_Carbon_PER_m2'] >= lower_bound) & (becd_df['Total_Embodied_Carbon_PER_m2'] <= upper_bound)]

"""
10. Save Cleaned DataFrame to CSV
"""
# Save dataframe to CSV for modeling
becd_df_PATH = os.path.join(export_dir, 'cleaned_becd.csv')
becd_df.to_csv(becd_df_PATH, index=False)
becd_df.info()
