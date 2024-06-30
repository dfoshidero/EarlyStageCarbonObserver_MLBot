"""
1. Load datasets and import relevant libraries.
"""

import pandas as pd
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import uniform, randint

# Define the base directory and data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
import_dir = os.path.join(current_dir, '../data/processed/model')

# Define the paths to the cleaned and encoded datasets
becd_df_PATH = os.path.join(import_dir, 'encoded_becd.csv')
carbenmats_df_PATH = os.path.join(import_dir, 'encoded_carbenmats.csv')
clf_df_PATH = os.path.join(import_dir, 'encoded_clf.csv')

# Load the datasets
becd_df = pd.read_csv(becd_df_PATH)
carbenmats_df = pd.read_csv(carbenmats_df_PATH)
clf_df = pd.read_csv(clf_df_PATH)

"""
2. Prepare datasets for model training.
"""
# Define target column
target_column = 'Total_Embodied_Carbon_PER_m2'

# Prepare datasets (use all features except the target column)
X_becd = becd_df.drop(columns=[target_column])
y_becd = becd_df[target_column]

X_carbenmats = carbenmats_df.drop(columns=[target_column])
y_carbenmats = carbenmats_df[target_column]

X_clf = clf_df.drop(columns=[target_column])
y_clf = clf_df[target_column]

"""
3. Use hyperparameter tuning to find best estimators for each dataset.
"""
# Define parameter distribution for hyperparameter tuning
param_dist = {
    'regressor__n_estimators': randint(100, 1000),
    'regressor__learning_rate': uniform(0.01, 0.2),
    'regressor__max_depth': randint(3, 6)
}

# Function to perform hyperparameter tuning
def tune_model(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, n_iter=50, cv=3, n_jobs=-1, scoring='r2', random_state=42)
    random_search.fit(X, y)
    return random_search.best_estimator_

"""
4. Train models with optimal params, and store in array.
"""
# Tune models and store the best estimators
model_becd = tune_model(X_becd, y_becd)
model_carbenmats = tune_model(X_carbenmats, y_carbenmats)
model_clf = tune_model(X_clf, y_clf)

# Store models in an array
models_to_train = [
    ('model_becd', model_becd, X_becd, y_becd),
    ('model_carbenmats', model_carbenmats, X_carbenmats, y_carbenmats),
    ('model_clf', model_clf, X_clf, y_clf)
]

# Display the best parameters for each model
for model_name, model, X, y in models_to_train:
    print(f'Best parameters for {model_name}: {model.get_params()}')

"""
5. Evaluate R-squared for each model, and only use best performers (with r squared above 0.5).
"""
# List to keep models with R-squared >= 0.5
valid_models = []

# Split data into training and testing sets, train the models, and evaluate R-squared
for model_name, model, X, y in models_to_train:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    r_squared_train = r2_score(y_train, y_train_pred)
    r_squared_test = r2_score(y_test, y_test_pred)
    
    print(f'R-squared for {model_name} on training set: {r_squared_train}')
    print(f'R-squared for {model_name} on testing set: {r_squared_test}')
    
    if r_squared_train >= 0.5 and r_squared_test >= 0.5:
        valid_models.append((model_name, model, X, y))

# Display valid models
print("\nModels with R-squared >= 0.5 on both training and testing sets:")
for model_name, model, X, y in valid_models:
    print(f'{model_name}')
