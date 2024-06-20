import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the base directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'models')
data_dir = os.path.join(current_dir, '../data')

# Load models
clf_model = joblib.load(os.path.join(model_dir, 'clf_model.pkl'))
aspects_model = joblib.load(os.path.join(model_dir, 'aspects_model.pkl'))
buildups_model = joblib.load(os.path.join(model_dir, 'buildups_model.pkl'))
ice_model = joblib.load(os.path.join(model_dir, 'ice_model.pkl'))

# Define feature names for each dataset
clf_numeric_features = ['Building Area in Square Meters', 'Building Storeys']
clf_categorical_features = ['Building Type', 'Building Use', 'Building Location Region']

aspects_categorical_features = ['Building Aspect', 'Element', 'Material']
buildups_categorical_features = ['buildup type', 'type structure', 'Materials included']
ice_categorical_features = ['Material', 'Sub-material']

# Function to preprocess data and make predictions
def preprocess_and_predict(data, model, numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model.named_steps['regressor'])
    ])

    # Fit the pipeline with some dummy data if not fitted
    dummy_data = pd.DataFrame([{**{col: 0 for col in numeric_features}, **{col: '' for col in categorical_features}}])
    pipeline.fit(dummy_data, [0])  # Assuming 0 as a dummy target value for fitting

    # Ensure the provided data has all necessary columns
    for col in numeric_features + categorical_features:
        if col not in data.columns:
            data[col] = np.nan if col in numeric_features else ''

    predictions = pipeline.predict(data)
    return predictions

# Main function to predict CLF target using all models
def predict_clf_target(aspects_inputs, buildups_inputs, ice_inputs, clf_input):
    # Convert input dictionaries to DataFrames
    aspects_X = pd.DataFrame(aspects_inputs)
    buildups_X = pd.DataFrame(buildups_inputs)
    ice_X = pd.DataFrame(ice_inputs)
    clf_X = pd.DataFrame([clf_input])

    # Predict using each model
    aspects_predictions = preprocess_and_predict(aspects_X, aspects_model, [], aspects_categorical_features)
    buildups_predictions = preprocess_and_predict(buildups_X, buildups_model, [], buildups_categorical_features)
    ice_predictions = preprocess_and_predict(ice_X, ice_model, [], ice_categorical_features)

    # Combine predictions into a DataFrame
    combined_predictions = pd.DataFrame({
        'aspects_predictions': [aspects_predictions.mean()],
        'buildups_predictions': [buildups_predictions.mean()],
        'ice_predictions': [ice_predictions.mean()]
    })

    # Merge combined predictions with CLF features
    combined_data = pd.concat([clf_X.reset_index(drop=True), combined_predictions], axis=1)

    # Fill missing values in numeric columns with the mean of predictions if available
    for col in clf_numeric_features:
        if col not in combined_data.columns or combined_data[col].isnull().all():
            combined_data[col] = combined_predictions.mean().values[0]

    # Predict final CLF target
    clf_predictions = preprocess_and_predict(combined_data, clf_model, clf_numeric_features + ['aspects_predictions', 'buildups_predictions', 'ice_predictions'], clf_categorical_features)
    
    return clf_predictions

# Function to get user input for each model's variables
def get_user_input():
    print("Enter values for CLF model (leave blank for no value):")
    clf_input = {
        'Building Type': input("Building Type: ") or None,
        'Building Use': input("Building Use: ") or None,
        'Building Location Region': input("Building Location Region: ") or None,
        'Building Area in Square Meters': input("Building Area in Square Meters: ") or None,
        'Building Storeys': input("Building Storeys: ") or None
    }

    print("\nEnter values for Aspects model (leave blank for no value):")
    aspects_input = {
        'Building Aspect': input("Building Aspect: ") or None,
        'Element': input("Element: ") or None,
        'Material': input("Material: ") or None
    }

    print("\nEnter values for Buildups model (leave blank for no value):")
    buildups_input = {
        'buildup type': input("Buildup Type: ") or None,
        'type structure': input("Type Structure: ") or None,
        'Materials included': input("Materials Included: ") or None
    }

    print("\nEnter values for ICE model (leave blank for no value, separate multiple materials with commas):")
    ice_input_material = input("Material (separate multiple materials with commas): ") or None
    ice_input_sub_material = input("Sub-material (separate multiple sub-materials with commas): ") or None

    # Handle multiple materials and sub-materials
    ice_inputs = []
    if ice_input_material and ice_input_sub_material:
        materials = ice_input_material.split(',')
        sub_materials = ice_input_sub_material.split(',')
        ice_inputs = [{'Material': material.strip(), 'Sub-material': sub_material.strip()} for material in materials for sub_material in sub_materials]
    elif ice_input_material:
        materials = ice_input_material.split(',')
        ice_inputs = [{'Material': material.strip(), 'Sub-material': ''} for material in materials]
    elif ice_input_sub_material:
        sub_materials = ice_input_sub_material.split(',')
        ice_inputs = [{'Material': '', 'Sub-material': sub_material.strip()} for sub_material in sub_materials]
    else:
        ice_inputs = [{}]

    # Convert missing numeric inputs to None
    for key in clf_input.keys():
        if clf_input[key] == '':
            clf_input[key] = None
        elif key in ['Building Area in Square Meters', 'Building Storeys'] and clf_input[key] is not None:
            clf_input[key] = float(clf_input[key])

    # Convert missing inputs for aspects and buildups to None
    for key in aspects_input.keys():
        if aspects_input[key] == '':
            aspects_input[key] = None

    for key in buildups_input.keys():
        if buildups_input[key] == '':
            buildups_input[key] = None

    return [aspects_input], [buildups_input], ice_inputs, clf_input

if __name__ == "__main__":
    aspects_inputs, buildups_inputs, ice_inputs, clf_input = get_user_input()
    clf_predictions = predict_clf_target(aspects_inputs, buildups_inputs, ice_inputs, clf_input)

    print("\nPredicted Embodied Carbon Whole Building Excluding Operational:")
    print(clf_predictions)
