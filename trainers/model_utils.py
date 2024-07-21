import pandas as pd
import os
import joblib
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import uniform, randint
import numpy as np

# Define parameter distributions for hyperparameter tuning
param_dist_gb = {
    "regressor__n_estimators": randint(50, 5000),
    "regressor__learning_rate": uniform(0.001, 0.2),
    "regressor__max_depth": randint(3, 10),
    "regressor__min_samples_split": randint(2, 20),
    "regressor__min_samples_leaf": randint(1, 20),
    "regressor__subsample": uniform(0.5, 0.5),
}

param_dist_hgb = {
    "regressor__max_iter": randint(50, 5000),
    "regressor__learning_rate": uniform(0.001, 0.2),
    "regressor__max_depth": randint(3, 10),
    "regressor__min_samples_leaf": randint(1, 20),
    "regressor__l2_regularization": uniform(0, 1),
    "regressor__max_bins": randint(10, 255),
    "regressor__max_leaf_nodes": randint(10, 255),
}

param_dist_rf = {
    "regressor__n_estimators": randint(50, 5000),
    "regressor__max_depth": randint(3, 20),
    "regressor__min_samples_split": randint(2, 20),
    "regressor__min_samples_leaf": randint(1, 20),
}

param_dist_lr = {"regressor__fit_intercept": [True, False]}

models = {
    "HistGradientBoosting": (
        HistGradientBoostingRegressor(random_state=42, verbose=1),
        param_dist_hgb,
        # "GradientBoosting": (
        #    GradientBoostingRegressor(random_state=42, verbose=1),
        #    param_dist_gb,
    ),
}


# Function to perform hyperparameter tuning
def tune_model(X, y):
    best_estimators = {}
    for model_name, (model, param_dist) in models.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", model)])

        param_space_size = np.prod(
            [len(v) if hasattr(v, "__len__") else 1 for v in param_dist.values()]
        )
        n_iter = min(50, param_space_size)

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=5,
            n_jobs=-1,
            scoring="r2",
            random_state=42,
            verbose=1,
        )
        random_search.fit(X, y)
        best_estimators[model_name] = random_search.best_estimator_
        print(f"Best parameters for {model_name}: {random_search.best_params_}")
    return best_estimators


# Function to load datasets
def load_datasets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    import_dir = os.path.join(current_dir, "../data/processed/inspect")

    synthetic_PATH = os.path.join(import_dir, "cleaned_synthetic.csv")

    synthetic_df = pd.read_csv(synthetic_PATH)

    return synthetic_df


# Function to encode categorical features and return label encoders
def encode_categorical(df):
    le_dict = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    return df, le_dict


# Function to prepare datasets for model training
def prepare_datasets(synthetic_df, target_column="Embodied Carbon (kgCO2e/m2)"):
    synthetic_df, synthetic_df_encoders = encode_categorical(synthetic_df)

    X_synthetic = synthetic_df.drop(columns=[target_column])
    y_synthetic = synthetic_df[target_column]

    return (X_synthetic, y_synthetic, synthetic_df_encoders)


# Function to save models and R-squared data
def save_model_and_data(model, model_name, model_dir, performance_logs):
    joblib_file = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, joblib_file)
    print(f"{model_name} saved to {joblib_file}")


# Function to load valid models
def load_valid_models(model_dir, model_files):
    valid_models = []
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            if isinstance(model, Pipeline):  # Check if the loaded object is a Pipeline
                valid_models.append(model)
            else:
                print(f"Error: {model_file} is not a valid Pipeline object")
        else:
            print(f"Model file {model_file} not found.")
    return valid_models


# Function to align input features with training features
def align_features(input_df, training_columns):
    input_df = pd.get_dummies(input_df)
    aligned_df = pd.DataFrame(columns=training_columns)
    for col in training_columns:
        if col in input_df.columns:
            aligned_df[col] = input_df[col]
        else:
            aligned_df[col] = 0
    return aligned_df


# Function to create stacking ensemble
def create_stacking_ensemble(models, final_estimator):
    estimators = [(name, model) for name, model in models.items()]
    stacking_regressor = StackingRegressor(
        estimators=estimators, final_estimator=final_estimator, n_jobs=-1
    )
    return stacking_regressor
