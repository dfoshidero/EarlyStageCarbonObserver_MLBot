import pandas as pd
import os
import joblib

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')

# Load the saved model
MODEL_PATH = os.path.join(models_dir, 'building_model_best.pkl')
saved_model = joblib.load(MODEL_PATH)

# Load the original data to get column names
BUILDING_DATA_PATH = os.path.join(models_dir, 'BUILDING_DATA.csv')
df = pd.read_csv(BUILDING_DATA_PATH)

# Define features
features = df.drop(columns=['Actual_Total_Carbon'])

# Function to make predictions with missing values using the saved model
def predict(input_data):
    # Get all required columns
    all_columns = features.columns
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data])
    # Add missing columns with None values
    for col in all_columns:
        if col not in input_df.columns:
            input_df[col] = None
    # Ensure the column order matches the training data
    input_df = input_df[all_columns]
    # Make the prediction
    prediction = saved_model.predict(input_df)
    return prediction[0]

# Example usage: Predicting for a hospital in the UK with missing values
input_data = {
    'Country_Name': 'United Kingdom',
    'Bldg_Type': 'Office',
    'Mass_Concrete_Without_Reinforcement': 5000,
    'Mass_Reinforced_Concrete': 3000,
    # Other fields are not specified and will be set to None
}

predicted_embodied_carbon = predict(input_data)
print(f"Predicted Embodied Carbon: {predicted_embodied_carbon}")
