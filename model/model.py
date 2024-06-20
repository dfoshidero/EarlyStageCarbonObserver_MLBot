import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import warnings

# Ignore convergence warnings during GridSearchCV
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')

# Load the data
BUILDING_DATA_PATH = os.path.join(models_dir, 'BUILDING_DATA.csv')
df = pd.read_csv(BUILDING_DATA_PATH)

# Define features and target
features = df.drop(columns=['Actual_Total_Carbon'])
target = df['Actual_Total_Carbon']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = features.select_dtypes(include=['object']).columns

# Preprocessing for numerical data: fill missing values with the mean
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data: fill missing values with 'missing' and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Define the hyperparameters to tune
param_grid = {
    'model__learning_rate': [0.01, 0.1],
    'model__max_iter': [100, 200],
    'model__max_leaf_nodes': [31, 50],
    'model__max_depth': [None, 10],
    'model__min_samples_leaf': [20, 30],
    'model__l2_regularization': [0.0, 0.1],
    'model__max_bins': [2, 255]
}

# Define the model
model = HistGradientBoostingRegressor(random_state=42)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Perform hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Save the trained model
MODEL_PATH = os.path.join(models_dir, 'building_model.pkl')
joblib.dump(best_model, MODEL_PATH)

# Evaluate the best model
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training R2: {train_r2}")
print(f"Test R2: {test_r2}")