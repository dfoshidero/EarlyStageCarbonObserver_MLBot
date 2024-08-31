"""
This script is designed to load a dataset, preprocess it, train multiple machine learning models, 
evaluate their performance using cross-validation, and save the models along with relevant metadata. 
The script limits the dataset size for model training, tunes the models, and logs their performance metrics.
"""

import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.base import clone
from model_utils import tune_model, load_datasets, prepare_datasets, save_model_and_data

# Define the base directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "../src/model")
os.makedirs(model_dir, exist_ok=True)

df = load_datasets()

# Save unique values from the dataset before preprocessing for later use
unique_values = {col: df[col].dropna().unique().tolist() for col in df.columns}
joblib.dump(unique_values, os.path.join(model_dir, "unique_values.pkl"))

X_cleaned, y_cleaned, cleaned_label_encoders = prepare_datasets(df)

# Save feature names for later use
joblib.dump(X_cleaned.columns.tolist(), os.path.join(model_dir, "features.pkl"))

# Save label encoders for later use
joblib.dump(cleaned_label_encoders, os.path.join(model_dir, "label_encoders.pkl"))

# Limiter for the number of data points to train
LIMITER = 150000  # Modify this value as needed

# Ensure LIMITER does not exceed the available data points
LIMITER = min(LIMITER, X_cleaned.shape[0])

X_cleaned_limited = X_cleaned.iloc[:LIMITER]
y_cleaned_limited = y_cleaned.iloc[:LIMITER]

# Tune models and store the best estimators
model_cleaned = tune_model(X_cleaned_limited, y_cleaned_limited)

# Initialize performance logs
performance_logs = []

for model_name, model in model_cleaned.items():
    full_model_name = f"synthetic_{model_name}"

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_cleaned_limited, y_cleaned_limited, test_size=0.3, random_state=42
    )

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    r_squared_train = r2_score(y_train, y_train_pred)

    print(f"R-squared for {full_model_name} on training set: {r_squared_train}")
    performance_logs.append(f"{full_model_name}: Training R-squared: {r_squared_train}")

    # Evaluate the model on the test set
    y_test_pred = model.predict(X_test)
    r_squared_test = r2_score(y_test, y_test_pred)
    print(f"R-squared for {full_model_name} on testing set: {r_squared_test}")
    performance_logs.append(f"{full_model_name}: Testing R-squared: {r_squared_test}")

    # Clone the model to perform cross-validation without affecting the original model
    model_cv = clone(model)
    if hasattr(model_cv, "verbose"):
        model_cv.verbose = 0

    # Calculate cross-validation scores
    try:
        cv_scores = cross_val_score(
            model_cv, X_train, y_train, cv=5, scoring="r2", verbose=0
        )
        cv_mean = cv_scores.mean()
        print(f"Cross-validation scores for {full_model_name}: {cv_scores}")
        print(f"Mean cross-validation score for {full_model_name}: {cv_mean}")

        performance_logs.append(
            f"{full_model_name}: Cross-validation scores: {cv_scores}"
        )
        performance_logs.append(
            f"{full_model_name}: Mean cross-validation score: {cv_mean}\n"
        )
    except Exception as e:
        print(f"Error in cross-validation for {full_model_name}: {e}")
        performance_logs.append(f"{full_model_name}: Cross-validation error: {e}\n")

    # Save the model and associated data
    save_model_and_data(model, full_model_name, model_dir, performance_logs)

# Save performance logs to a text file with date and time
log_dir = os.path.join(current_dir, "../data/logs")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
performance_file_path = os.path.join(log_dir, f"performance_models_{timestamp}.txt")
with open(performance_file_path, "w") as f:
    for line in performance_logs:
        f.write(line + "\n")

print(f"Performance data saved to {performance_file_path}")
