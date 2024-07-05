import joblib
import os
import pandas as pd
from sklearn.pipeline import Pipeline

# Define the base directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '../../data/processed/model')

becd_model_filepath = os.path.join(model_dir, 'becd_GradientBoosting.pkl')
carbenmats_model_filepath = os.path.join(model_dir, 'carbenmats_Stacking.pkl')

becd_features_filepath = os.path.join(model_dir, 'becd_features.pkl')
carbenmats_features_filepath = os.path.join(model_dir, 'carbenmats_features.pkl')

becd_pipeline_filepath = os.path.join(model_dir, 'becd_pipeline.pkl')
carbenmats_pipeline_filepath = os.path.join(model_dir, 'carbenmats_pipeline.pkl')

becd_label_encoders_filepath = os.path.join(model_dir, 'becd_label_encoders.pkl')
carbenmats_label_encoders_filepath = os.path.join(model_dir, 'carbenmats_label_encoders.pkl')


# Load pre-trained models, pipelines, and label encoders
becd_model = joblib.load(becd_model_filepath)
carbenmats_model = joblib.load(carbenmats_model_filepath)

becd_features = joblib.load(becd_features_filepath)
carbenmats_features = joblib.load(carbenmats_features_filepath)

becd_pipeline = joblib.load(becd_pipeline_filepath)
carbenmats_pipeline = joblib.load(carbenmats_pipeline_filepath)

becd_label_encoders = joblib.load(becd_label_encoders_filepath)
carbenmats_label_encoders = joblib.load(carbenmats_label_encoders_filepath)

# Weights for the models based on mean cross-validation scores
weight_becd = 0.8032982028038453 / (0.8032982028038453 + 0.69744985346875)
weight_carbenmats = 0.69744985346875 / (0.8032982028038453 + 0.69744985346875)

def apply_label_encoding(user_input, label_encoders):
    """
    Apply label encoding to the user input using the provided label encoders.
    
    :param user_input: dictionary with user inputs
    :param label_encoders: dictionary with label encoders
    :return: DataFrame with label encoded features
    """
    encoded_input = {}
    for feature, values in user_input.items():
        if feature in label_encoders:
            encoder = label_encoders[feature]
            encoded_values = []
            for value in values:
                if value in encoder.classes_:
                    encoded_values.append(encoder.transform([value])[0])
                elif 'Other' in encoder.classes_:
                    encoded_values.append(encoder.transform(['Other'])[0])
                elif 'missing' in encoder.classes_:
                    encoded_values.append(encoder.transform(['missing'])[0])
                else:
                    encoded_values.append(None)
            encoded_input[feature] = encoded_values
        else:
            encoded_input[feature] = values
    return pd.DataFrame(encoded_input)

def preprocess_input(pipeline, user_input, features, label_encoders):
    """
    Preprocess user input using the provided pipeline and label encoders.
    
    :param pipeline: preprocessing pipeline
    :param user_input: dictionary with user inputs
    :param features: list of feature names used during training
    :param label_encoders: dictionary with label encoders
    :return: preprocessed input DataFrame
    """
    input_df = apply_label_encoding(user_input, label_encoders)
    aligned_df = align_features(input_df, features)

    if aligned_df.empty:
        raise ValueError("Aligned DataFrame is empty. Check if input features match training features.")

    return pipeline.named_steps['scaler'].transform(aligned_df)

def predict_becd(user_input):
    """
    Predict using the becd model pipeline.
    
    :param user_input: dictionary with user inputs
    :return: prediction result
    """
    preprocessed_input = preprocess_input(becd_pipeline, user_input, becd_features, becd_label_encoders)
    return becd_model.predict(preprocessed_input)

def predict_carbenmats(user_input):
    """
    Predict using the carbenmats model pipeline.
    
    :param user_input: dictionary with user inputs
    :return: prediction result
    """
    preprocessed_input = preprocess_input(carbenmats_pipeline, user_input, carbenmats_features, carbenmats_label_encoders)
    return carbenmats_model.predict(preprocessed_input)

def combined_prediction(user_input_becd, user_input_carbenmats):
    """
    Generate a combined prediction using both becd and carbenmats models.
    
    :param user_input_becd: dictionary with user inputs for becd model
    :param user_input_carbenmats: dictionary with user inputs for carbenmats model
    :return: combined prediction result
    """
    pred_becd = predict_becd(user_input_becd)
    pred_carbenmats = predict_carbenmats(user_input_carbenmats)
    
    final_prediction = (weight_becd * pred_becd) + (weight_carbenmats * pred_carbenmats)
    return final_prediction

def align_features(input_df, training_columns):
    """
    Align input features with training features.

    :param input_df: DataFrame with user inputs
    :param training_columns: List of feature names used during training
    :return: DataFrame with aligned features
    """
    aligned_df = pd.DataFrame(columns=training_columns)
    for col in training_columns:
        if col in input_df.columns:
            aligned_df[col] = input_df[col]
        else:
            aligned_df[col] = 0
    return aligned_df
