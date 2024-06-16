# prep.py
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

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
riba_targets = pd.read_excel(RIBA_TARGETS_PATH)  # to be used for comparisons when user asks questions regarding relative sustainability of building.
clf_embodied_carbon = pd.read_csv(CLF_EMBODIED_CARBON_PATH)

# Drop unnecessary columns
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

# Function to clean numerical columns
def clean_numerical_columns(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert non-numeric values to NaN
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        df[column] = df[column].fillna(df[column].median())
    return df

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

# Clean numerical columns in each dataset
fcbs_aspects_elements = clean_numerical_columns(fcbs_aspects_elements, numerical_columns['fcbs_aspects_elements'])
fcbs_build_ups_details = clean_numerical_columns(fcbs_build_ups_details, numerical_columns['fcbs_build_ups_details'])
fcbs_sectors_subsectors = clean_numerical_columns(fcbs_sectors_subsectors, numerical_columns['fcbs_sectors_subsectors'])
ice_db = clean_numerical_columns(ice_db, numerical_columns['ice_db'])
clf_embodied_carbon = clean_numerical_columns(clf_embodied_carbon, numerical_columns['clf_embodied_carbon'])

# Tokenizer and model for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to concatenate text columns and tokenize using BERT
def tokenize_and_embed(df, text_columns):
    concatenated_texts = df[text_columns].astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
    inputs = tokenizer(concatenated_texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Tokenize and embed text columns
fcbs_aspects_elements_text_embeddings = tokenize_and_embed(fcbs_aspects_elements, ['Building Aspect', 'Element', 'Material', 'Assumptions'])
fcbs_build_ups_details_text_embeddings = tokenize_and_embed(fcbs_build_ups_details, ['Build-up Reference', 'Description', 'Materials included'])
fcbs_sectors_subsectors_text_embeddings = tokenize_and_embed(fcbs_sectors_subsectors, ['Sector', 'Building Typology', 'Sub-sector'])
ice_db_text_embeddings = tokenize_and_embed(ice_db, ['Material', 'Sub-material'])
clf_embodied_carbon_text_embeddings = tokenize_and_embed(clf_embodied_carbon, ['Building Type', 'Building Use', 'Building Location Region', 'Building New or Renovation'])

# Combine numerical data and text embeddings
def combine_numerical_and_text(numerical_data, text_embeddings):
    return np.hstack((numerical_data, text_embeddings))

fcbs_aspects_elements_combined = combine_numerical_and_text(fcbs_aspects_elements[numerical_columns['fcbs_aspects_elements']].values, fcbs_aspects_elements_text_embeddings)
fcbs_build_ups_details_combined = combine_numerical_and_text(fcbs_build_ups_details[numerical_columns['fcbs_build_ups_details']].values, fcbs_build_ups_details_text_embeddings)
fcbs_sectors_subsectors_combined = combine_numerical_and_text(fcbs_sectors_subsectors[numerical_columns['fcbs_sectors_subsectors']].values, fcbs_sectors_subsectors_text_embeddings)
ice_db_combined = combine_numerical_and_text(ice_db[numerical_columns['ice_db']].values, ice_db_text_embeddings)
clf_embodied_carbon_combined = combine_numerical_and_text(clf_embodied_carbon[numerical_columns['clf_embodied_carbon']].values, clf_embodied_carbon_text_embeddings)

# Save the prepared data
np.save(os.path.join(export_dir, 'fcbs_aspects_elements_combined.npy'), fcbs_aspects_elements_combined)
np.save(os.path.join(export_dir, 'fcbs_build_ups_details_combined.npy'), fcbs_build_ups_details_combined)
np.save(os.path.join(export_dir, 'fcbs_sectors_subsectors_combined.npy'), fcbs_sectors_subsectors_combined)
np.save(os.path.join(export_dir, 'ice_db_combined.npy'), ice_db_combined)
np.save(os.path.join(export_dir, 'clf_embodied_carbon_combined.npy'), clf_embodied_carbon_combined)

# Save prepared data to CSV for inspection
pd.DataFrame(fcbs_aspects_elements_combined).to_csv(os.path.join(export_dir, 'fcbs_aspects_elements_combined.csv'), index=False)
pd.DataFrame(fcbs_build_ups_details_combined).to_csv(os.path.join(export_dir, 'fcbs_build_ups_details_combined.csv'), index=False)
pd.DataFrame(fcbs_sectors_subsectors_combined).to_csv(os.path.join(export_dir, 'fcbs_sectors_subsectors_combined.csv'), index=False)
pd.DataFrame(ice_db_combined).to_csv(os.path.join(export_dir, 'ice_db_combined.csv'), index=False)
pd.DataFrame(clf_embodied_carbon_combined).to_csv(os.path.join(export_dir, 'clf_embodied_carbon_combined.csv'), index=False)

# Save target values
np.save(os.path.join(export_dir, 'ice_db_targets.npy'), ice_db['Embodied Carbon (kg CO2e per declared unit)'].values)
np.save(os.path.join(export_dir, 'fcbs_aspects_elements_targets.npy'), fcbs_aspects_elements['Embodied carbon (kgCO2e/unit)'].values)
np.save(os.path.join(export_dir, 'fcbs_build_ups_details_targets.npy'), fcbs_build_ups_details['TOTAL kgCO2e/FU'].values)
np.save(os.path.join(export_dir, 'fcbs_sectors_subsectors_targets.npy'), fcbs_sectors_subsectors['Typical (Total)'].values)
np.save(os.path.join(export_dir, 'clf_embodied_carbon_targets.npy'), clf_embodied_carbon['Embodied Carbon Whole Building Excluding Operational'].values)

# Save target values to CSV for inspection
pd.DataFrame({'target': ice_db['Embodied Carbon (kg CO2e per declared unit)'].values}).to_csv(os.path.join(export_dir, 'ice_db_targets.csv'), index=False)
pd.DataFrame({'target': fcbs_aspects_elements['Embodied carbon (kgCO2e/unit)'].values}).to_csv(os.path.join(export_dir, 'fcbs_aspects_elements_targets.csv'), index=False)
pd.DataFrame({'target': fcbs_build_ups_details['TOTAL kgCO2e/FU'].values}).to_csv(os.path.join(export_dir, 'fcbs_build_ups_details_targets.csv'), index=False)
pd.DataFrame({'target': fcbs_sectors_subsectors['Typical (Total)'].values}).to_csv(os.path.join(export_dir, 'fcbs_sectors_subsectors_targets.csv'), index=False)
pd.DataFrame({'target': clf_embodied_carbon['Embodied Carbon Whole Building Excluding Operational'].values}).to_csv(os.path.join(export_dir, 'clf_embodied_carbon_targets.csv'), index=False)

print("Data preparation complete and files saved.")