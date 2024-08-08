import joblib
import os
import pandas as pd
import numpy as np


def load_resources():
    """
    Load the necessary resources.
    :return: tuple of loaded resources
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "model")

    features_filepath = os.path.join(model_dir, "features.pkl")
    label_encoders_filepath = os.path.join(model_dir, "label_encoders.pkl")
    synthetic_model_filepath = os.path.join(
        model_dir, "synthetic_HistGradientBoosting.pkl"
    )
    unique_values_filepath = os.path.join(model_dir, "unique_values.pkl")

    with open(synthetic_model_filepath, "rb") as f:
        model = joblib.load(f)
    with open(features_filepath, "rb") as f:
        features = joblib.load(f)
    with open(label_encoders_filepath, "rb") as f:
        label_encoders = joblib.load(f)
    with open(unique_values_filepath, "rb") as f:
        unique_values = joblib.load(f)

    return model, features, label_encoders, unique_values


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

    # Clear input DataFrame to free memory
    del input_df

    return aligned_df


def predict(user_input, model, features, label_encoders):
    """
    Predict using the model.

    :param user_input: dictionary with user inputs
    :param model: trained model
    :param features: list of feature names used during training
    :param label_encoders: dictionary with label encoders
    :return: prediction result
    """
    preprocessed_input = preprocess_input(user_input, features, label_encoders)
    prediction = model.predict(preprocessed_input)

    # Clear intermediate data
    del preprocessed_input

    return prediction


def predictor(user_input):
    """
    Generate a combined prediction using the model.

    :param user_input: dictionary with user inputs
    :return: combined prediction result
    """
    model, features, label_encoders, unique_values = load_resources()
    pred = predict(user_input, model, features, label_encoders)

    # Clear loaded resources to free memory
    del model, features, label_encoders, unique_values

    return pred


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

    # Clear input DataFrame to free memory
    del input_df

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

    # Clear user_input and unique_values to free memory
    del user_input, unique_values
