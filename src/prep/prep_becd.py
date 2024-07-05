import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data/raw')
export_dir = os.path.join(current_dir, '../../data/processed')

os.makedirs(export_dir, exist_ok=True)

BECD_PATH = os.path.join(data_dir, 'model/BECD_2024-06-17 18.41.17.csv')

becd_df = pd.read_csv(BECD_PATH)

"""
2. Calculate Total Embodied Carbon
"""
# Calculate the total embodied carbon for BECD
becd_df['Total Embodied Carbon PER m2'] = becd_df[
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
becd_df = becd_df[becd_df['Total Embodied Carbon PER m2'] != 0]

"""
4. Select Relevant Columns
"""
# Select relevant columns
relevant_columns = [
    'ProjectType', 
    'PSCFoundationTypePrimary', 'PSCGroundFloorTypePrimary', 'PSCVerticalElementStructureTypePrimary', 'PSCHorizontalElementTypePrimary',
    'ProjectStageComponentsSlabTypePrimary', 'PSCCladdingTypePrimary', 'PSCHeatingTypePrimary', 'PSCCoolingTypePrimary',
    'PSCFinishesTypePrimary', 'PSCVentilationTypePrimary',
    'Total Embodied Carbon PER m2'
] # Only using primary material types.

becd_df = becd_df[relevant_columns]

"""
5. Rename Columns
"""
# Rename columns to be clearer
becd_df = becd_df.rename(columns={
    'ProjectType': 'Building Project Type',
    'PSCFoundationTypePrimary': 'Primary Foundation Type',
    'PSCGroundFloorTypePrimary': 'Primary Ground_Floor Type',
    'PSCVerticalElementStructureTypePrimary': 'Primary Vertical Element Type',
    'PSCHorizontalElementTypePrimary': 'Primary Horizontal Element Type',
    'ProjectStageComponentsSlabTypePrimary': 'Primary Slab Type',
    'PSCCladdingTypePrimary': 'Primary Cladding Type',
    'PSCHeatingTypePrimary': 'Primary Heating Type',
    'PSCCoolingTypePrimary': 'Primary Cooling Type',
    'PSCFinishesTypePrimary': 'Primary Finishes Type',
    'PSCVentilationTypePrimary': 'Primary Ventilation Type'
})

"""
6. Impute Missing Values
"""
# Impute missing values for Building_Project_Type and Country
becd_df['Building Project Type'].fillna('missing', inplace=True)

"""
7. Drop Rows with 'Not Applicable' or 'Unknown' in ALL Primary Columns
"""
# Drop rows where any of the Primary_ columns have all values as "Not applicable" or "Unknown"
values_to_check = ["Not applicable", "Unknown"]
primary_columns = [col for col in becd_df.columns if col.startswith('Primary')]

# Create a mask to identify rows to drop
mask = becd_df[primary_columns].apply(lambda row: all(val in values_to_check for val in row), axis=1)
becd_df = becd_df[~mask]

"""
8. Drop Rows with Empty Cells
"""
# Drop columns with empty cells
becd_df = becd_df.dropna()

"""
9. Replace 'Not applicable' or 'Unknown' with NaN and Impute
"""
# Replace 'Not applicable' with NaN and 'Unknown' with 'Other'
becd_df.replace({'Not applicable': np.nan, 'Unknown': 'Other'}, inplace=True)

# Impute missing values in categorical columns with 'missing'
becd_df = becd_df.fillna('Other')

"""
10. Remove Outliers
"""
# Calculate Q1, Q3, and IQR for Total_Embodied_Carbon
Q1 = becd_df['Total Embodied Carbon PER m2'].quantile(0.25)
Q3 = becd_df['Total Embodied Carbon PER m2'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
becd_df = becd_df[(becd_df['Total Embodied Carbon PER m2'] >= lower_bound) & (becd_df['Total Embodied Carbon PER m2'] <= upper_bound)]

"""
11. Save Cleaned DataFrame to CSV for inspection.
"""
# Save dataframe to CSV for modeling
becd_df_PATH = os.path.join(export_dir, 'inspect/cleaned_becd.csv')
becd_df.to_csv(becd_df_PATH, index=False)
becd_df.info()

"""
12. Label encode categorical data for ML use.
"""
# Label encode categorical columns
label_encoder = LabelEncoder()
categorical_columns = becd_df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    becd_df[col] = label_encoder.fit_transform(becd_df[col])


"""
13. Save dataframe to CSV for modeling.
"""
becd_df_PATH = os.path.join(export_dir, 'encoded/encoded_becd.csv')
becd_df.to_csv(becd_df_PATH, index=False)
becd_df.info()