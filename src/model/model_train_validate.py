import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from datetime import datetime
import joblib

from model_utils import tune_model, load_datasets, prepare_datasets, save_model_and_data, create_stacking_ensemble

# Define the base directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '../../data/processed/model')

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)

# Load the datasets
becd_df, carbenmats_df, clf_df = load_datasets()

# Prepare datasets for model training
(X_becd, y_becd, becd_label_encoders), (X_carbenmats, y_carbenmats, carbenmats_label_encoders), (X_clf, y_clf, clf_label_encoders) = prepare_datasets(becd_df, carbenmats_df, clf_df)

# Save feature names for later use
joblib.dump(X_becd.columns.tolist(), os.path.join(model_dir, 'becd_features.pkl'))
joblib.dump(X_carbenmats.columns.tolist(), os.path.join(model_dir, 'carbenmats_features.pkl'))

# Save label encoders for later use
joblib.dump(becd_label_encoders, os.path.join(model_dir, 'becd_label_encoders.pkl'))
joblib.dump(carbenmats_label_encoders, os.path.join(model_dir, 'carbenmats_label_encoders.pkl'))
joblib.dump(clf_label_encoders, os.path.join(model_dir, 'clf_label_encoders.pkl'))

# Tune models and store the best estimators
model_becd = tune_model(X_becd, y_becd)
model_carbenmats = tune_model(X_carbenmats, y_carbenmats)
model_clf = tune_model(X_clf, y_clf)

# Save the preprocessing pipelines along with the model
joblib.dump(model_becd['RandomForest'], os.path.join(model_dir, 'becd_pipeline.pkl'))
joblib.dump(model_carbenmats['GradientBoosting'], os.path.join(model_dir, 'carbenmats_pipeline.pkl'))

# Store models in an array
models_to_train = [
    ('becd', model_becd, X_becd, y_becd),
    ('carbenmats', model_carbenmats, X_carbenmats, y_carbenmats),
    ('clf', model_clf, X_clf, y_clf)
]

# Initialize performance logs
performance_logs = []

for dataset_name, best_models, X, y in models_to_train:
    for model_name, model in best_models.items():
        full_model_name = f'{dataset_name}_{model_name}'
        
        # First split: train and remaining (validation + test)
        X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.4, random_state=42)
        
        # Second split: validation and test from remaining data
        X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)
        
        # Train and tune the model using the training and validation sets
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        r_squared_train = r2_score(y_train, y_train_pred)
        r_squared_val = r2_score(y_val, y_val_pred)
        
        print(f'R-squared for {full_model_name} on training set: {r_squared_train}')
        print(f'R-squared for {full_model_name} on validation set: {r_squared_val}')
        
        performance_logs.append(f'{full_model_name}: Training R-squared: {r_squared_train}')
        performance_logs.append(f'{full_model_name}: Validation R-squared: {r_squared_val}')
        
        # Evaluate the model on the test set
        y_test_pred = model.predict(X_test)
        r_squared_test = r2_score(y_test, y_test_pred)
        print(f'R-squared for {full_model_name} on testing set: {r_squared_test}')
            
        performance_logs.append(f'{full_model_name}: Testing R-squared: {r_squared_test}')
        
        # Calculate cross-validation scores
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            print(f'Cross-validation scores for {full_model_name}: {cv_scores}')
            print(f'Mean cross-validation score for {full_model_name}: {cv_mean}')
            
            performance_logs.append(f'{full_model_name}: Cross-validation scores: {cv_scores}')
            performance_logs.append(f'{full_model_name}: Mean cross-validation score: {cv_mean}\n')
        except Exception as e:
            print(f'Error in cross-validation for {full_model_name}: {e}')
            performance_logs.append(f'{full_model_name}: Cross-validation error: {e}\n')
        
        # Save the model
        save_model_and_data(model, full_model_name, model_dir, performance_logs)

    # Create stacking ensemble using the best models
    final_estimator = Ridge()
    stacking_model = create_stacking_ensemble(best_models, final_estimator)
    
    # Train and evaluate the stacking ensemble
    full_model_name = f'{dataset_name}_Stacking'
    
    stacking_model.fit(X_train, y_train)
    
    y_train_pred = stacking_model.predict(X_train)
    y_val_pred = stacking_model.predict(X_val)
    
    r_squared_train = r2_score(y_train, y_train_pred)
    r_squared_val = r2_score(y_val, y_val_pred)
    
    print(f'R-squared for {full_model_name} on training set: {r_squared_train}')
    print(f'R-squared for {full_model_name} on validation set: {r_squared_val}')
    
    performance_logs.append(f'{full_model_name}: Training R-squared: {r_squared_train}')
    performance_logs.append(f'{full_model_name}: Validation R-squared: {r_squared_val}')
    
    # Evaluate the stacking model on the test set
    y_test_pred = stacking_model.predict(X_test)
    r_squared_test = r2_score(y_test, y_test_pred)
    print(f'R-squared for {full_model_name} on testing set: {r_squared_test}')
        
    performance_logs.append(f'{full_model_name}: Testing R-squared: {r_squared_test}')
    
    # Calculate cross-validation scores for the stacking model
    try:
        cv_scores = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        print(f'Cross-validation scores for {full_model_name}: {cv_scores}')
        print(f'Mean cross-validation score for {full_model_name}: {cv_mean}')
        
        performance_logs.append(f'{full_model_name}: Cross-validation scores: {cv_scores}')
        performance_logs.append(f'{full_model_name}: Mean cross-validation score: {cv_mean}\n')
    except Exception as e:
        print(f'Error in cross-validation for {full_model_name}: {e}')
        performance_logs.append(f'{full_model_name}: Cross-validation error: {e}\n')
    
    # Save the stacking model
    save_model_and_data(stacking_model, full_model_name, model_dir, performance_logs)

# Save performance logs to a text file with date and time
import_dir = os.path.join(current_dir, '../../data/logs')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
performance_file_path = os.path.join(import_dir, f'performance_models_{timestamp}.txt')
with open(performance_file_path, 'w') as f:
    for line in performance_logs:
        f.write(line + '\n')

print(f'Performance data saved to {performance_file_path}')
