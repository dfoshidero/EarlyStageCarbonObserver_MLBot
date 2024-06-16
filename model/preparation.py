import pandas as pd
import os

import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from sklearn.preprocessing import StandardScaler
from itertools import product

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../data')
export_dir = os.path.join(current_dir, 'export')

# Ensure the export directory exists
os.makedirs(export_dir, exist_ok=True)

# Construct the full paths to each dataset
ASPECTS_PATH = os.path.join(data_dir, 'FCBS_Aspects-Elements-Materials_MachineReadable.xlsx')
BUILDUPS_PATH = os.path.join(data_dir, 'FCBS_Build Ups-Details_MachineReadable.xlsx')
SECTORS_PATH = os.path.join(data_dir, 'FCBS_Sectors-Subsectors_MachineReadable.xlsx')
ICE_DB_PATH = os.path.join(data_dir, 'ICE DB_Cleaned.csv')
RIBA_TARGETS_PATH = os.path.join(data_dir, 'RIBA 2030-Targets_MachineReadable.xlsx')
CLF_EMBODIED_CARBON_PATH = os.path.join(data_dir, 'CLF Embodied Carbon_Cleaned.csv')

# Load datasets
fcbs_aspects_elements = pd.read_excel(ASPECTS_PATH)
fcbs_build_ups_details = pd.read_excel(BUILDUPS_PATH)
fcbs_sectors_subsectors = pd.read_excel(SECTORS_PATH)
ice_db = pd.read_csv(ICE_DB_PATH)
riba_targets = pd.read_excel(RIBA_TARGETS_PATH) # to be used for comparisons when user asks questions regarding relative sustainability of building.
clf_embodied_carbon = pd.read_csv(CLF_EMBODIED_CARBON_PATH)

fcbs_aspects_elements = fcbs_aspects_elements.drop(columns=['Declared Unit', 'Metric multiplier'])
fcbs_build_ups_details = fcbs_build_ups_details.drop(columns=['Functional Unit', 'Volume/FU (m3/FU)', 'ICE V3 Reference', 'ICE V3 kgCO2e/kg value',
                                                              'ICE V3 biogenic kgCO2e/kg value', 'Density used', 'Buildup kgCO2e/FU',
                                                              'Buildup biogenic kgCO2e/FU', 'Mass kg/FU'])
fcbs_sectors_subsectors = fcbs_sectors_subsectors.drop(columns=['BS EN 1991-1 specific use classification', 'Imposed floor load (kN/m2)', 'Partitions factor:',
                                                                'RIBA 2030 Targets classification', 'DEC A', 'A4', 'A5', 'B1', 'B2', 'B3',
                                                                'B5', 'C1', 'C2', 'C3', 'C4'])
ice_db = ice_db.drop(columns=['Unique ID', 'Quantity of declared unit', 'Units of declared unit',
                              'Weight per declared unit - kg', 'Density of material - kg per m3', 'Density range - low - kg per m3',
                               'Density Range - high - kg per m3', 'Carbon sequestration included?'])
clf_embodied_carbon = clf_embodied_carbon.drop(columns=['Building Public ID', 'Building Area in Square Meters', 'Building Area in Square Feet', 
                                                        'Building Storeys', 'Life Cycle Assessment Year', 'Life Cycle Assessment Reference Period', 
                                                        'Life Cycle Assessment Source Code', 'Life Cycle Assessment Stages', 'Life Cycle Assessment Building Scope',
                                                        'Life Cycle Assessment Material Quantity'])


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

buildups_descriptions = fcbs_build_ups_details['Description'].tolist()
tokens = tokenizer(buildups_descriptions, padding=True, truncation=True, return_tensors='pt')

# Define tokenized columns to process
tokenized_columns = {
    'fcbs_aspects_elements': ['Building Aspect', 'Element', 'Material', 'Assumptions'],
    'fcbs_build_ups_details': ['Build-up Reference', 'Description', 'Materials included'],
    'fcbs_sectors_subsectors': ['Sector', 'Building Typology', 'Sub-sector'],
    'ice_db': ['Material', 'Sub-material'],
    'clf_embodied_carbon': ['Building Type', 'Building Use', 'Building Location Region', 'Building New or Renovation']
}

# Define numerical columns to process
numerical_columns = {
    'fcbs_aspects_elements': ['Mass (kg/unit)', 'Weight (kN/unit)', 'Embodied carbon (kgCO2e/unit)', 'Sequestered Carbon (kgCO2e/unit)'],
    'fcbs_build_ups_details': ['TOTAL kgCO2e/FU', 'TOTAL BIOGENIC kgCO2e/FU'],
    'fcbs_sectors_subsectors': ['Grid size:', 'Typical (Electric)', 'Best Practice (Electric)', 'Innovative (Electric)',
                                'Typical (Non-electric)', 'Best Practice (Non-electric)', 'Innovative (Non-electric)',
                                'Typical (Total)', 'Best Practice (Total)', 'Innovative (Total)', 'Max (Total)'],
    'ice_db': ['Carbon Storage Amount (Kg CO2 per unit)', 'Embodied Carbon (kg CO2e per declared unit)', 'Embodied Carbon per kg (kg CO2e per kg)'],
    'clf_embodied_carbon': ['Embodied Carbon Whole Building Excluding Operational', 'Embodied Carbon Life Cycle Assessment Area Per Square Meter',
                            'Minimum Building Area in Square Meters', 'Maximum Building Area in Square Meters', 
                            'Minimum Building Area in Square Feet', 'Maximum Building Area in Square Feet', 
                            'Minimum Building Storeys', 'Maximum Building Storeys']
}

# Function to tokenize all relevant columns in a dataframe
def tokenize_dataframe(df, columns):
    tokenized_data = {}
    for column in columns:
        tokenized_data[column] = df[column].apply(lambda x: tokenizer.encode(str(x), add_special_tokens=True) if pd.notnull(x) else [])
    return tokenized_data

# Function to standardize numerical columns in a dataframe
def standardize_numerical_columns(df, numerical_columns):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_columns].fillna(0))
    return pd.DataFrame(scaled_data, columns=numerical_columns)

# Function to clean numerical columns in a dataframe
def clean_numerical_columns(df, columns, value=None):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert non-numeric values to NaN
        if value is None:
            finite_values = df[column].replace([np.inf, -np.inf], np.nan).dropna()
            value = finite_values.median()  # You can choose mean, median, or any other strategy
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        df[column] = df[column].fillna(value)
    return df

# Clean numerical columns in each dataset
fcbs_aspects_elements = clean_numerical_columns(fcbs_aspects_elements, numerical_columns['fcbs_aspects_elements'])
fcbs_build_ups_details = clean_numerical_columns(fcbs_build_ups_details, numerical_columns['fcbs_build_ups_details'])
fcbs_sectors_subsectors = clean_numerical_columns(fcbs_sectors_subsectors, numerical_columns['fcbs_sectors_subsectors'])
ice_db = clean_numerical_columns(ice_db, numerical_columns['ice_db'])
clf_embodied_carbon = clean_numerical_columns(clf_embodied_carbon, numerical_columns['clf_embodied_carbon'])

# Tokenize relevant columns from each dataset
fcbs_aspects_elements_tokens = tokenize_dataframe(fcbs_aspects_elements, tokenized_columns['fcbs_aspects_elements'])
fcbs_build_ups_details_tokens = tokenize_dataframe(fcbs_build_ups_details, tokenized_columns['fcbs_build_ups_details'])
fcbs_sectors_subsectors_tokens = tokenize_dataframe(fcbs_sectors_subsectors, tokenized_columns['fcbs_sectors_subsectors'])
ice_db_tokens = tokenize_dataframe(ice_db, tokenized_columns['ice_db'])
clf_embodied_carbon_tokens = tokenize_dataframe(clf_embodied_carbon, tokenized_columns['clf_embodied_carbon'])

# Standardize numerical data
fcbs_aspects_elements_numerical = standardize_numerical_columns(fcbs_aspects_elements, numerical_columns['fcbs_aspects_elements'])
fcbs_build_ups_details_numerical = standardize_numerical_columns(fcbs_build_ups_details, numerical_columns['fcbs_build_ups_details'])
fcbs_sectors_subsectors_numerical = standardize_numerical_columns(fcbs_sectors_subsectors, numerical_columns['fcbs_sectors_subsectors'])
ice_db_numerical = standardize_numerical_columns(ice_db, numerical_columns['ice_db'])
clf_embodied_carbon_numerical = standardize_numerical_columns(clf_embodied_carbon, numerical_columns['clf_embodied_carbon'])

"""
# Convert tokenized data back to DataFrames
fcbs_aspects_elements_tokens_df = pd.DataFrame(fcbs_aspects_elements_tokens)
fcbs_build_ups_details_tokens_df = pd.DataFrame(fcbs_build_ups_details_tokens)
fcbs_sectors_subsectors_tokens_df = pd.DataFrame(fcbs_sectors_subsectors_tokens)
ice_db_tokens_df = pd.DataFrame(ice_db_tokens)
clf_embodied_carbon_tokens_df = pd.DataFrame(clf_embodied_carbon_tokens)

# Export cleaned and standardized datasets
fcbs_aspects_elements_tokens_df.to_csv(os.path.join(export_dir, 'cleaned_fcbs_aspects_elements.csv'), index=False)
fcbs_build_ups_details_tokens_df.to_csv(os.path.join(export_dir, 'cleaned_fcbs_build_ups_details.csv'), index=False)
fcbs_sectors_subsectors_tokens_df.to_csv(os.path.join(export_dir, 'cleaned_fcbs_sectors_subsectors.csv'), index=False)
ice_db_tokens_df.to_csv(os.path.join(export_dir, 'cleaned_ice_db.csv'), index=False)
clf_embodied_carbon_tokens_df.to_csv(os.path.join(export_dir, 'cleaned_clf_embodied_carbon.csv'), index=False)

# Print the first few rows of each dataset to understand their structure
print(fcbs_aspects_elements.head())
print(fcbs_build_ups_details.head())
print(fcbs_sectors_subsectors.head())
print(ice_db.head())
print(riba_targets.head())
print(clf_embodied_carbon.head())

"""

# Combine tokenized data from all datasets into one list of lists
combined_tokens = []
numerical_features_list = []

# Helper function to combine tokens and numerical features
def combine_data(tokenized_data, numerical_data, data_length):
    for i in range(data_length):
        tokens = []
        for key in tokenized_data.keys():
            tokens.extend(tokenized_data[key][i])
        combined_tokens.append(tokens)
        numerical_features_list.append(numerical_data.iloc[i].values)

# Combine data from each dataset
combine_data(fcbs_aspects_elements_tokens, fcbs_aspects_elements_numerical, len(fcbs_aspects_elements))
combine_data(fcbs_build_ups_details_tokens, fcbs_build_ups_details_numerical, len(fcbs_build_ups_details))
combine_data(fcbs_sectors_subsectors_tokens, pd.DataFrame(np.zeros((len(fcbs_sectors_subsectors), len(numerical_columns['fcbs_sectors_subsectors'])))), len(fcbs_sectors_subsectors))  # Assuming no numerical data for sectors/subsectors
combine_data(ice_db_tokens, ice_db_numerical, len(ice_db))
combine_data(clf_embodied_carbon_tokens, clf_embodied_carbon_numerical, len(clf_embodied_carbon))

# Pad sequences to ensure uniform length for tokenized data
features = pad_sequence([torch.tensor(f) for f in combined_tokens], batch_first=True, padding_value=tokenizer.pad_token_id)

# Ensure numerical features have a uniform length (padding if necessary)
max_length = max(len(features[i]) for i in range(len(features)))
for i in range(len(numerical_features_list)):
    numerical_features_list[i] = np.pad(numerical_features_list[i], (0, max_length - len(numerical_features_list[i])), 'constant')

# Convert numerical features to tensor
numerical_features = torch.tensor(np.array(numerical_features_list), dtype=torch.float32)

# Export tokenized and numerical data
np.save(os.path.join(export_dir, 'tokenized_features.npy'), features.numpy())
np.save(os.path.join(export_dir, 'numerical_features.npy'), numerical_features.numpy())

# Print confirmation
print(f"Tokenized and numerical data have been exported to {export_dir}")

# Prepare target values (total embodied carbon) from clf_embodied_carbon
targets = torch.tensor(clf_embodied_carbon['Embodied Carbon Whole Building Excluding Operational'].values, dtype=torch.float32)

# Export target values
np.save(os.path.join(export_dir, 'targets.npy'), targets.numpy())

# Inspect the shape of the features and targets
print("Features shape:", features.shape)
print("Numerical features shape:", numerical_features.shape)
print("Targets shape:", targets.shape)