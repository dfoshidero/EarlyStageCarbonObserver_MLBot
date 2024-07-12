import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

"""
1. Import synthetic datasets.
"""

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data/raw')
export_dir = os.path.join(current_dir, '../../data/processed')

os.makedirs(export_dir, exist_ok=True)

fcbsData_30000A_path = os.path.join(data_dir, 'model/synthetic/220712FCBS_30000.csv')

fcbsData_30000A = pd.read_csv(fcbsData_30000A_path)

"""
2. Combine synthetic datasets.
"""

syntheticData = pd.concat([fcbsData_30000A])

"""
3. Define and save assumptions to separate dataframe.
"""
def Assume(materials, data):
    # Create an empty DataFrame for assumptions
    assumptionsData = pd.DataFrame(columns=['Material', 'Building Element', 'Assumption'])

    # Function to clean and add rows to the DataFrame
    def add_to_assumptions(material_type, assumption, column):
        clean_assumption = assumption.replace(material_type, '').replace('+', '').strip()
        new_row = {'Material': material_type, 'Building Element': column, 'Assumption': clean_assumption}
        return pd.DataFrame([new_row])

    # Iterate through each column to find and extract columns with the specified materials
    for col in data.columns:
        for material_type in materials:
            values = data[col][data[col].astype(str).str.contains(material_type)]
            if not values.empty:
                unique_assumptions = values.unique()
                for assumption in unique_assumptions:
                    assumptionsData = pd.concat([assumptionsData, add_to_assumptions(material_type, assumption, col)], ignore_index=True)

    # Sort the DataFrame by 'Material' and 'Building Element'
    assumptionsData = assumptionsData.sort_values(by=['Material', 'Building Element'])
    
    # Reset the index for better readability
    assumptionsData.reset_index(drop=True, inplace=True)

    return assumptionsData

# Define the materials to search for
materials = ['Precast RC', 'RC', 'screed', 'Timber Joists', 'JJI Engineered Joists', "Steel tile with"]

# Generate the assumptions data
assumptionsData = Assume(materials, syntheticData)

"""
4. Clean datasets and re-name variables to.
"""

# Replace nan values with "Other"
syntheticData.replace(np.nan, "Other", inplace=True)

# Replace all "Foamglass (domestic only)" cell with Foamglass
for col in syntheticData.columns:
    syntheticData[col] = syntheticData[col].apply(
        lambda x: 'Foamglass' 
        if "Foamglass (domestic only)" == str(x)
        else x )
    
# Replace all "Precast RC" cell with Precast Concrete 
for col in syntheticData.columns:
    syntheticData[col] = syntheticData[col].apply(
        lambda x: 'Precast Concrete' 
        if "Precast RC" in str(x)
        else x )
    
# Replace all "RC" cell with Reinforced Concrete 
for col in syntheticData.columns:
    syntheticData[col] = syntheticData[col].apply(
        lambda x: 'Reinforced Concrete' 
        if "RC" in str(x)
        else x )
    
# Remove all "+ OSB Topper" in joists
for col in syntheticData.columns:
    syntheticData[col] = syntheticData[col].apply(
        lambda x: str(x).replace(' + OSB topper', '').strip()
        if " + OSB topper" in str(x)
        else x )

# Replace all "70mm screed" cell with Screed
for col in syntheticData.columns:
    syntheticData[col] = syntheticData[col].apply(
        lambda x: 'Screed' 
        if "70mm screed" in str(x)
        else x )


# Replace all "Steel tile with 18mm acoustic pad" cell with Steel tile with acoustic pad
for col in syntheticData.columns:
    syntheticData[col] = syntheticData[col].apply(
        lambda x: 'Steel tile with acoustic pad' 
        if "Steel tile with 18mm acoustic pad" == str(x)
        else x )
    
# Replace all "+" cell with //
for col in syntheticData.columns:
    syntheticData[col] = syntheticData[col].apply(
        lambda x: '//' 
        if "+" in str(x)
        else x )

    
# Update values in the "Sector" column with clearer names
subsector_mapping = {
    'Single family house': 'Single Family House',
    'Flat/maisonette': 'Small Flat/Maisonette',
    'Multi-family (< 6 storeys)': 'Low-Rise Apartments',
    'Multi-family (6 - 15 storeys)': 'Mid-Rise Apartments',
    'Multi-family (> 15 storeys)': 'High-Rise Apartments/Hotels',
    'Office': 'Commercial'
}
syntheticData['Sub-Sector'] = syntheticData['Sub-Sector'].replace(subsector_mapping)

# Update values in the "Sector" column with clearer names
sector_mapping = {
    'Office': 'Commercial'
}
syntheticData['Sector'] = syntheticData['Sector'].replace(subsector_mapping)

"""
5. Save dataframe(s) to CSV for inspection.
"""

def printUniqueCols():
    unique_vals ={}
    for column in syntheticData:
        unique_vals[column] = syntheticData[column].unique()

        print(unique_vals)

syntheticData_PATH = os.path.join(export_dir, 'inspect/cleaned_synthetic.csv')
syntheticData.to_csv(syntheticData_PATH, index=False)
syntheticData.info()

assumptionsData_PATH = os.path.join(export_dir, 'inspect/assumptions.csv')
assumptionsData.to_csv(assumptionsData_PATH, index=False)
assumptionsData.info()