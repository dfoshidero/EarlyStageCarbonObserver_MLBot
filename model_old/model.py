# model.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
export_dir = os.path.join(current_dir, 'export')

# Load prepared data
ice_db_combined = np.load(os.path.join(export_dir, 'ice_db_combined.npy'))
ice_db_targets = np.load(os.path.join(export_dir, 'ice_db_targets.npy'))

fcbs_aspects_elements_combined = np.load(os.path.join(export_dir, 'fcbs_aspects_elements_combined.npy'))
fcbs_aspects_elements_targets = np.load(os.path.join(export_dir, 'fcbs_aspects_elements_targets.npy'))

fcbs_build_ups_details_combined = np.load(os.path.join(export_dir, 'fcbs_build_ups_details_combined.npy'))
fcbs_build_ups_details_targets = np.load(os.path.join(export_dir, 'fcbs_build_ups_details_targets.npy'))

fcbs_sectors_subsectors_combined = np.load(os.path.join(export_dir, 'fcbs_sectors_subsectors_combined.npy'))
fcbs_sectors_subsectors_targets = np.load(os.path.join(export_dir, 'fcbs_sectors_subsectors_targets.npy'))

clf_embodied_carbon_combined = np.load(os.path.join(export_dir, 'clf_embodied_carbon_combined.npy'))
clf_embodied_carbon_targets = np.load(os.path.join(export_dir, 'clf_embodied_carbon_targets.npy'))

# Function to train and save models for each stage
def train_and_save_model(X, y, model_name):
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f'{model_name} Mean Squared Error: {mse}')
    joblib.dump(model, os.path.join(export_dir, f'{model_name}.joblib'))
    return predictions

# First stage: ICE DB
ice_db_predictions = train_and_save_model(ice_db_combined, ice_db_targets, 'ice_db_model')

# Save the mean prediction from ICE DB for later use
ice_db_mean_prediction = np.mean(ice_db_predictions)

# Second stage: FCBS Aspects Elements
aspect_combined_with_ice = np.hstack((fcbs_aspects_elements_combined, np.full((fcbs_aspects_elements_combined.shape[0], 1), ice_db_mean_prediction)))
aspects_predictions = train_and_save_model(aspect_combined_with_ice, fcbs_aspects_elements_targets, 'fcbs_aspects_elements_model')

# Second stage: FCBS Build Ups Details
buildup_combined_with_ice = np.hstack((fcbs_build_ups_details_combined, np.full((fcbs_build_ups_details_combined.shape[0], 1), ice_db_mean_prediction)))
build_ups_predictions = train_and_save_model(buildup_combined_with_ice, fcbs_build_ups_details_targets, 'fcbs_build_ups_details_model')

# Save the mean predictions from FCBS Aspects and Build Ups for later use
aspects_mean_prediction = np.mean(aspects_predictions)
build_ups_mean_prediction = np.mean(build_ups_predictions)

# Third stage: FCBS Sectors Subsectors
sector_combined_with_effects = np.hstack((fcbs_sectors_subsectors_combined, 
                                          np.full((fcbs_sectors_subsectors_combined.shape[0], 1), aspects_mean_prediction), 
                                          np.full((fcbs_sectors_subsectors_combined.shape[0], 1), build_ups_mean_prediction)))
sectors_predictions = train_and_save_model(sector_combined_with_effects, fcbs_sectors_subsectors_targets, 'fcbs_sectors_subsectors_model')

# Save the mean prediction from FCBS Sectors for later use
sectors_mean_prediction = np.mean(sectors_predictions)

# Final stage: CLF Embodied Carbon
clf_combined_with_sectors = np.hstack((clf_embodied_carbon_combined, np.full((clf_embodied_carbon_combined.shape[0], 1), sectors_mean_prediction)))
train_and_save_model(clf_combined_with_sectors, clf_embodied_carbon_targets, 'clf_embodied_carbon_model')

print("Models trained and saved.")
