import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../data/processed')
model_dir = os.path.join(current_dir, '../models')

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Load the dataset
DF_PATH = os.path.join(data_dir, 'BUILDING_DATA.csv')
df = pd.read_csv(DF_PATH)

# Define the features and target variable
features = df.drop(columns=['Total_Carbon'])
target = df['Total_Carbon']

# Apply label encoding to categorical features
categorical_features = ['Project_Type', 'Building_Use_Type', 'Building_Use_Subtype', 'Continent', 'Country', 'City', 'Structure_Type', 'Roof_Type']
label_encoders = {}

for feature in categorical_features:
    encoder = LabelEncoder()
    features[feature] = encoder.fit_transform(features[feature])
    label_encoders[feature] = encoder

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Save the model and label encoders for future use
MODEL_PATH = os.path.join(model_dir, 'carbon_prediction_model.pkl')
ENCODERS_PATH = os.path.join(model_dir, 'label_encoders.pkl')
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoders, ENCODERS_PATH)
print(f"Model saved to {MODEL_PATH}")
print(f"Label encoders saved to {ENCODERS_PATH}")
