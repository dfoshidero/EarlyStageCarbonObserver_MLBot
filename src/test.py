import pandas as pd
import numpy as np
import joblib
import os

# Define the base directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../data/processed')
model_dir = os.path.join(current_dir, '../models')
model_path = os.path.join(model_dir, 'carbon_prediction_model.pkl')
encoders_path = os.path.join(model_dir, 'label_encoders.pkl')

# Load the saved model and encoders
model = joblib.load(model_path)
label_encoders = joblib.load(encoders_path)

# Load the dataset to calculate averages for missing values
DF_PATH = os.path.join(data_dir, 'BUILDING_DATA.csv')
df = pd.read_csv(DF_PATH)

# Calculate mean for numerical columns
mean_values = df.mean()

# Calculate mode for categorical columns, with a check for empty mode DataFrame
mode_values = {}
for col in df.select_dtypes(include=[object]).columns:
    mode_value = df[col].mode()
    if not mode_value.empty:
        mode_values[col] = mode_value.iloc[0]
    else:
        mode_values[col] = df[col].value_counts().idxmax()  # Use the most frequent value if mode is empty

# Example new data for prediction (with partial information)
new_data = {
    'Project_Type': ['New construction'],
    'Building_Use_Type': ['Office'],
    'Building_Use_Subtype': [None],  # Missing value
    'Continent': ['Europe'],
    'Country': ['Germany'],
    'City': [None],  # Missing value
    'Gross_Floor_Area': [15000],
    'Total_Users': [None],  # Missing value
    'Floors_Above_Ground': [10],
    'Floors_Below_Ground': [2],
    'Structure_Type': ['Steel'],
    'Roof_Type': [None],  # Missing value
    'Mass_Wood': [1000],
    'Mass_Straw_Hemp': [None],  # Missing value
    'Mass_Fungi': [50],
    'Mass_Brass_Copper': [300],
    'Mass_Earth': [400],
    'Mass_Bamboo': [100],
    'Mass_Glass': [500],
    'Mass_Stone': [600],
    'Mass_Stone_Wool': [150],
    'Mass_Ceramics': [200],
    'Mass_Metals': [1200],
    'Mass_Plastics': [300],
    'Mass_Steel_Reinforcement': [800],
    'Mass_EPS_XPS': [250],
    'Mass_Aluminium': [150],
    'Mass_Concrete_Without_Reinforcement': [2000],
    'Mass_Other': [100],
    'Mass_Concrete_With_Reinforcement': [2500],
    'Mass_Cement_Mortar': [600],
    'Total_Mass_Materials': [None]  # Missing value
}

# Convert the new data to a DataFrame
new_data_df = pd.DataFrame(new_data)

# Fill missing numerical values with the mean
for col in new_data_df.select_dtypes(include=[np.number]).columns:
    new_data_df[col].fillna(mean_values[col], inplace=True)

# Fill missing categorical values with the mode
for col in new_data_df.select_dtypes(include=[object]).columns:
    if col in mode_values:
        new_data_df[col].fillna(mode_values[col], inplace=True)

# Ensure no NaN values remain in the DataFrame
if new_data_df.isna().any().any():
    print(new_data_df)
    raise ValueError("There are still NaN values in the input data after imputation.")

# Apply label encoding to categorical features
categorical_cols = ['Project_Type', 'Building_Use_Type', 'Building_Use_Subtype', 'Continent', 'Country', 'City', 'Structure_Type', 'Roof_Type']
for feature in categorical_cols:
    encoder = label_encoders[feature]
    new_data_df[feature] = new_data_df[feature].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

# Ensure no NaN values remain in the DataFrame after label encoding
if new_data_df.isna().any().any():
    print(new_data_df)
    raise ValueError("There are still NaN values in the input data after label encoding.")

# Make predictions using the loaded model
predictions = model.predict(new_data_df)

# Display the predictions
for i, prediction in enumerate(predictions):
    print(f"Prediction for data point {i+1}: {prediction:.2f}")

# Optionally, save the predictions to a CSV file
PREDICTIONS_PATH = os.path.join(data_dir, 'predictions.csv')
new_data_df['Predicted_Total_Carbon'] = predictions
new_data_df.to_csv(PREDICTIONS_PATH, index=False)
print(f"Predictions saved to {PREDICTIONS_PATH}")
