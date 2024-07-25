import joblib
import os
import pandas as pd
import numpy as np

# Define the base directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "model")

# Updated model file paths based on new files
features_filepath = os.path.join(model_dir, "features.pkl")
label_encoders_filepath = os.path.join(model_dir, "label_encoders.pkl")
synthetic_model_filepath = os.path.join(model_dir, "synthetic_HistGradientBoosting.pkl")
unique_values_filepath = os.path.join(model_dir, "unique_values.pkl")

# Load pre-trained models, label encoders, and unique values
model = joblib.load(synthetic_model_filepath)
features = joblib.load(features_filepath)
label_encoders = joblib.load(label_encoders_filepath)
unique_values = joblib.load(unique_values_filepath)


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
                elif "Other" in encoder.classes_:
                    encoded_values.append(encoder.transform(["Other"])[0])
                else:
                    # Create a new category "Unknown" if it doesn't exist
                    new_classes = np.append(encoder.classes_, "Unknown")
                    encoder.classes_ = new_classes
                    encoded_values.append(encoder.transform(["Unknown"])[0])
            encoded_input[feature] = encoded_values
        else:
            encoded_input[feature] = values
    return pd.DataFrame(encoded_input)


def preprocess_input(user_input, features, label_encoders):
    """
    Preprocess user input using the provided label encoders.

    :param user_input: dictionary with user inputs
    :param features: list of feature names used during training
    :param label_encoders: dictionary with label encoders
    :return: preprocessed input DataFrame
    """
    input_df = apply_label_encoding(user_input, label_encoders)
    aligned_df = align_features(input_df, features)

    if aligned_df.empty:
        raise ValueError(
            "Aligned DataFrame is empty. Check if input features match training features."
        )

    return aligned_df


def predict(user_input):
    """
    Predict using the model.

    :param user_input: dictionary with user inputs
    :return: prediction result
    """
    preprocessed_input = preprocess_input(user_input, features, label_encoders)
    return model.predict(preprocessed_input)


def predictor(user_input):
    """
    Generate a combined prediction using the model.

    :param user_input: dictionary with user inputs
    :return: combined prediction result
    """
    pred = predict(user_input)
    final_prediction = pred  # Adjust as necessary if you have multiple models
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
            aligned_df[col] = np.nan  # Keep missing values as NaN
    return aligned_df


def validate_user_input(user_input, unique_values):
    """
    Validate user input against unique values.

    :param user_input: dictionary with user inputs
    :param unique_values: dictionary with unique values for each feature
    :return: None, raises ValueError if validation fails
    """
    for feature, values in user_input.items():
        if feature in unique_values:
            for value in values:
                if value not in unique_values[feature]:
                    raise ValueError(
                        f"Value for {feature} can only be {unique_values[feature]}."
                    )
