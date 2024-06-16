import torch
import numpy as np
import os
from transformers import BertTokenizer
from sklearn.linear_model import LinearRegression
import joblib

# Define the export directory
current_dir = os.path.dirname(os.path.abspath(__file__))
export_dir = os.path.join(current_dir, 'export')

# Load the trained models and data
fcbs_aspects_elements = torch.load(os.path.join(export_dir, 'fcbs_aspects_elements.pt'))
fcbs_build_ups_details = torch.load(os.path.join(export_dir, 'fcbs_build_ups_details.pt'))
fcbs_sectors_subsectors = torch.load(os.path.join(export_dir, 'fcbs_sectors_subsectors.pt'))
ice_db = torch.load(os.path.join(export_dir, 'ice_db.pt'))
clf_embodied_carbon = torch.load(os.path.join(export_dir, 'clf_embodied_carbon.pt'))

ice_model = joblib.load(os.path.join(export_dir, 'ice_model.joblib'))
aspect_model = joblib.load(os.path.join(export_dir, 'aspect_model.joblib'))
buildup_model = joblib.load(os.path.join(export_dir, 'buildup_model.joblib'))
sector_model = joblib.load(os.path.join(export_dir, 'sector_model.joblib'))
clf_model = joblib.load(os.path.join(export_dir, 'clf_model.joblib'))

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Extract features from user input
def extract_features_from_input(material, location):
    # Tokenize material and location
    material_tokens = tokenizer.encode(material, add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
    location_tokens = tokenizer.encode(location, add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
    
    # Use mean values for numerical features not provided
    aspect_mean = np.mean(fcbs_aspects_elements[['Mass (kg/unit)', 'Weight (kN/unit)', 'Sequestered Carbon (kgCO2e/unit)']].values, axis=0)
    buildup_mean = np.mean(fcbs_build_ups_details[['TOTAL BIOGENIC kgCO2e/FU']].values, axis=0)
    sector_mean = np.mean(fcbs_sectors_subsectors[['Grid size:', 'Typical (Electric)', 'Best Practice (Electric)', 'Innovative (Electric)', 
                                                   'Typical (Non-electric)', 'Best Practice (Non-electric)', 'Innovative (Non-electric)',
                                                   'Typical (Total)', 'Best Practice (Total)', 'Innovative (Total)', 'Max (Total)']].values, axis=0)
    clf_mean = np.mean(clf_embodied_carbon[['Embodied Carbon Life Cycle Assessment Area Per Square Meter', 'Minimum Building Area in Square Meters', 
                                            'Maximum Building Area in Square Meters', 'Minimum Building Area in Square Feet', 
                                            'Maximum Building Area in Square Feet', 'Minimum Building Storeys', 'Maximum Building Storeys']].values, axis=0)
    
    return material_tokens, location_tokens, aspect_mean, buildup_mean, sector_mean, clf_mean

# Make predictions based on the processed input
def predict_embodied_carbon(material, location):
    material_tokens, location_tokens, aspect_mean, buildup_mean, sector_mean, clf_mean = extract_features_from_input(material, location)
    
    # First-Stage Prediction: ICE DB
    X_ice = np.hstack([material_tokens, ice_db[['Carbon Storage Amount (Kg CO2 per unit)', 'Embodied Carbon per kg (kg CO2e per kg)']].mean(axis=0)])
    ice_effect = ice_model.predict([X_ice])[0]

    # Second-Stage Prediction: Aspects and Build-ups
    X_aspect = np.hstack([material_tokens, aspect_mean, [ice_effect]])
    aspect_effect = aspect_model.predict([X_aspect])[0]
    
    X_buildup = np.hstack([material_tokens, buildup_mean, [ice_effect]])
    buildup_effect = buildup_model.predict([X_buildup])[0]
    
    # Third-Stage Prediction: Sectors/Subsectors
    X_sector = np.hstack([location_tokens, sector_mean, [aspect_effect], [buildup_effect]])
    sector_effect = sector_model.predict([X_sector])[0]
    
    # Final-Stage Prediction: CLF Embodied Carbon
    X_clf = np.hstack([location_tokens, clf_mean, [sector_effect]])
    final_prediction = clf_model.predict([X_clf])[0]
    
    return final_prediction

# Example usage
user_input_material = "concrete foundations"
predicted_carbon = predict_embodied_carbon(user_input_material, user_input_location)
print(f"The predicted embodied carbon for the building is: {predicted_carbon} kg CO2e")
