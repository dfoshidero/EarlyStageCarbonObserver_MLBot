import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../data')
export_dir = os.path.join(current_dir, 'models')

# Ensure the export directory exists
os.makedirs(export_dir, exist_ok=True)

# Construct the full paths to each dataset
ASPECTS_PATH = os.path.join(data_dir, 'FCBS_Aspects-Elements-Materials_MachineReadable.xlsx')
BUILDUPS_PATH = os.path.join(data_dir, 'FCBS_Build Ups-Details_MachineReadable.xlsx')
ICE_DB_PATH = os.path.join(data_dir, 'ICE DB_Cleaned.csv')
CLF_EMBODIED_CARBON_PATH = os.path.join(data_dir, 'CLF Embodied Carbon_Cleaned.csv')

# Load datasets
aspects = pd.read_excel(ASPECTS_PATH)
buildups = pd.read_excel(BUILDUPS_PATH)
ice_db = pd.read_csv(ICE_DB_PATH)
clf_carbon = pd.read_csv(CLF_EMBODIED_CARBON_PATH)

# Function to clean numeric columns
def clean_numeric(df, columns):
    for col in columns:
        df[col] = df[col].replace('#NAME?', 0)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].replace({np.inf: 1000000000, -np.inf: 0}, inplace=True)
    df.dropna(subset=columns, inplace=True)
    return df

# Clean CLF dataset
clf_numeric_columns = [
    'Minimum Building Area in Square Meters',
    'Maximum Building Area in Square Meters',
    'Minimum Building Storeys',
    'Maximum Building Storeys'
]
clf_carbon = clean_numeric(clf_carbon, clf_numeric_columns)
clf_carbon['Embodied Carbon Whole Building Excluding Operational'] = pd.to_numeric(clf_carbon['Embodied Carbon Whole Building Excluding Operational'], errors='coerce')
clf_carbon.dropna(subset=['Embodied Carbon Whole Building Excluding Operational'], inplace=True)
clf_X = clf_carbon[['Building Type', 'Building Use', 'Building Location Region'] + clf_numeric_columns]
clf_y = clf_carbon['Embodied Carbon Whole Building Excluding Operational']

# Clean Aspects dataset
aspects['Buildup kgCO2e/kg'] = aspects['Embodied carbon (kgCO2e/unit)'] / aspects['Mass (kg/unit)']
aspects['Buildup kgCO2e/kg'] = pd.to_numeric(aspects['Buildup kgCO2e/kg'], errors='coerce')
aspects.dropna(subset=['Buildup kgCO2e/kg'], inplace=True)
aspects_X = aspects[['Building Aspect', 'Element', 'Material']]
aspects_y = aspects['Buildup kgCO2e/kg']

# Clean Buildups dataset
buildups[['buildup type', 'type structure']] = buildups['Build-up Reference'].str.split(':', expand=True)
buildups['TOTAL kgCO2e/kg'] = buildups['TOTAL kgCO2e/FU'] / buildups['Mass kg/FU']
buildups['TOTAL kgCO2e/kg'] = pd.to_numeric(buildups['TOTAL kgCO2e/kg'], errors='coerce')
buildups.dropna(subset=['TOTAL kgCO2e/kg'], inplace=True)
buildups_X = buildups[['buildup type', 'type structure', 'Materials included']]
buildups_y = buildups['TOTAL kgCO2e/kg']

# Clean ICE dataset
ice_db['Embodied Carbon per kg (kg CO2e per kg)'] = pd.to_numeric(ice_db['Embodied Carbon per kg (kg CO2e per kg)'], errors='coerce')
ice_db.dropna(subset=['Embodied Carbon per kg (kg CO2e per kg)'], inplace=True)
ice_X = ice_db[['Material', 'Sub-material']]
ice_y = ice_db['Embodied Carbon per kg (kg CO2e per kg)']

# Text data handling
def create_pipeline(numeric_features, categorical_features, model):
    # Create a column transformer to handle both numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create a pipeline that includes the preprocessor and the regression model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    return pipeline

# Define numeric and categorical features for each dataset
clf_numeric_features = ['Minimum Building Area in Square Meters', 'Maximum Building Area in Square Meters', 'Minimum Building Storeys', 'Maximum Building Storeys']
clf_categorical_features = ['Building Type', 'Building Use', 'Building Location Region']

aspects_categorical_features = ['Building Aspect', 'Element', 'Material']

buildups_categorical_features = ['buildup type', 'type structure', 'Materials included']

ice_categorical_features = ['Material', 'Sub-material']

# Split data into training and testing sets
clf_X_train, clf_X_test, clf_y_train, clf_y_test = train_test_split(clf_X, clf_y, test_size=0.3, random_state=42)
aspects_X_train, aspects_X_test, aspects_y_train, aspects_y_test = train_test_split(aspects_X, aspects_y, test_size=0.3, random_state=42)
buildups_X_train, buildups_X_test, buildups_y_train, buildups_y_test = train_test_split(buildups_X, buildups_y, test_size=0.3, random_state=42)
ice_X_train, ice_X_test, ice_y_train, ice_y_test = train_test_split(ice_X, ice_y, test_size=0.3, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.1, 0.01],
    'regressor__max_depth': [3, 5]
}

def tune_model(X_train, y_train, numeric_features, categorical_features):
    model = GradientBoostingRegressor()
    pipeline = create_pipeline(numeric_features, categorical_features, model)
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='r2')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    
    r2_train = r2_score(y_train, y_pred_train)
    
    print(f'{model_name} Model R² (Train): {r2_train}')
    print(f'{model_name} Model R² (Test): {r2_test}')
    print(f'{model_name} Model MSE: {mse_test}')
    print(f'{model_name} Model MAPE: {mape_test}%\n')

# Tune and evaluate models
clf_model = tune_model(clf_X_train, clf_y_train, clf_numeric_features, clf_categorical_features)
evaluate_model('CLF', clf_model, clf_X_train, clf_X_test, clf_y_train, clf_y_test)

aspects_model = tune_model(aspects_X_train, aspects_y_train, [], aspects_categorical_features)
evaluate_model('Aspects', aspects_model, aspects_X_train, aspects_X_test, aspects_y_train, aspects_y_test)

buildups_model = tune_model(buildups_X_train, buildups_y_train, [], buildups_categorical_features)
evaluate_model('Buildups', buildups_model, buildups_X_train, buildups_X_test, buildups_y_train, buildups_y_test)

ice_model = tune_model(ice_X_train, ice_y_train, [], ice_categorical_features)
evaluate_model('ICE', ice_model, ice_X_train, ice_X_test, ice_y_train, ice_y_test)

# Save models
joblib.dump(clf_model, os.path.join(export_dir, 'clf_model.pkl'))
joblib.dump(aspects_model, os.path.join(export_dir, 'aspects_model.pkl'))
joblib.dump(buildups_model, os.path.join(export_dir, 'buildups_model.pkl'))
joblib.dump(ice_model, os.path.join(export_dir, 'ice_model.pkl'))

# Export cleaned data
clf_carbon.to_csv(os.path.join(export_dir, 'cleaned_CLF_carbon.csv'), index=False)
aspects.to_csv(os.path.join(export_dir, 'cleaned_aspects.csv'), index=False)
buildups.to_csv(os.path.join(export_dir, 'cleaned_buildups.csv'), index=False)
ice_db.to_csv(os.path.join(export_dir, 'cleaned_ice_db.csv'), index=False)
