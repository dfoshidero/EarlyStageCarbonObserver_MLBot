"""
1. Import necessary libraries and load data.
"""

import pandas as pd
import numpy as np

import os
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, '../data/processed')
models_dir = os.path.join(current_dir, '../models')

os.makedirs(models_dir, exist_ok=True)
# Load the data
BUILDING_DATA_PATH = os.path.join(input_dir, 'BUILDING_DATA.csv')
df = pd.read_csv(BUILDING_DATA_PATH)

"""
2. Create function to train and save model.
"""

def train_model(X_train, y_train, X_test, y_test, features):
    models = {
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor()
    }
    
    best_model = None
    best_score = float('-inf')
    best_model_name = ""
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        results.append({
            'Feature': features,
            'Model': name,
            'R_squared': r2,
            'MSE': mse
        })
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_mse = mse
            best_model_name = name
    
    model_path = os.path.join(models_dir, f'model_{features}.pkl')
    joblib.dump(best_model, model_path)
    print(f"Feature: {features}, Best model: {best_model_name}, R_squared: {best_score:.4f}, MSE: {best_mse:.4f}")
    
    return best_model, model_path, results

"""
3. Initialize model array and results list
"""

model_array = []
results_list = []

"""
4. Create loop to model each feature against Actual Total Carbon
"""
# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Initialize label encoders for categorical columns
label_encoders = {col: LabelEncoder() for col in categorical_columns}

# Encode categorical columns
for col, encoder in label_encoders.items():
    df[col] = encoder.fit_transform(df[col])

for column in df.columns:
    if column == 'Actual_Total_Carbon':
        continue
    
    X = df[[column]].copy()
    y = df['Actual_Total_Carbon']
    
    # For categorical features, convert to category type
    if column in categorical_columns:
        X[column] = X[column].astype('category')
    
    # Drop rows with NaN values
    df_feature = X.join(y).dropna()
    X = df_feature[[column]]
    y = df_feature['Actual_Total_Carbon']
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model, model_path, results = train_model(X_train, y_train, X_test, y_test, column)
    model_array.append((column, model_path))
    results_list.extend(results)

"""
5. Save results to CSV
"""

results_df = pd.DataFrame(results_list)
results_file_path = os.path.join(input_dir, 'train_results.csv')
results_df.to_csv(results_file_path, index=False)

"""
6. Define function to calculate errors in models
"""

def calculate_errors(df, feature_col, target_col='Actual_Total_Carbon'):
    X = df[[feature_col]]
    y_true = df[target_col]
    
    model_name = [name for name in os.listdir(models_dir) if f'model_{feature_col}' in name][0]
    model_path = os.path.join(models_dir, model_name)
    model = joblib.load(model_path)
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return mse, r2

"""
6. Calculate errors for each model.
"""

# Calculate individual errors
errors = {}
r2_scores = {}
for column, model_path in model_array:
    df_feature = df[[column, 'Actual_Total_Carbon']].dropna()
    mse, r2 = calculate_errors(df_feature, column)
    
    errors[column] = mse
    r2_scores[column] = r2

# Calculate overall error and R-squared (example: weighted average)
overall_error = np.average(list(errors.values()), weights=[1/err for err in errors.values() if err != 0])
overall_r2 = np.average(list(r2_scores.values()), weights=[score for score in r2_scores.values() if score != 0])

# Display results
print("Overall MSE:", overall_error)
print("Overall R-squared:", overall_r2)
print(f"Results saved to {results_file_path}")
