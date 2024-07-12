import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
import joblib
import re
from word2number import w2n

# Define the base directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '../data/processed/model')

# Load model and tokenizer from the specified directory
def load_model(model_dir):
    model_path = os.path.join(model_dir, 'context/saved_model')
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def parse_input(text, unique_values, tokenizer, model, numerical_features, threshold=0.5):
    """
    Parses input text and extracts features using a trained NLP model,
    placing them into the appropriate structure for prediction with actual values.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.squeeze(0)  # Use logits for classification tasks
    probs = torch.sigmoid(logits)  # Convert logits to probabilities

    # Extract all numbers from the text
    numbers_in_text = re.findall(r'\b\d+\b', text)
    words_in_text = text.split()

    # Convert number words to digits if found
    for word in words_in_text:
        try:
            number = w2n.word_to_num(word)
            numbers_in_text.append(str(number))
        except ValueError:
            continue

    # Convert to integers and remove duplicates
    numerical_values = list(set(map(int, numbers_in_text)))

    structured_input = {feature: [] for feature in unique_values.keys()}

    # Assign numbers to the most likely numerical features
    used_numbers = []
    for feature in numerical_features:
        if feature in numerical_features:
            number = pick_most_likely_num(feature, probs, numerical_values, used_numbers, threshold, numerical_features)
            structured_input[feature] = number
    
    # Assign categorical features
    for feature, values in unique_values.items():
        if feature not in numerical_features:
            structured_input[feature] = extract_most_likely_feature(text, values, threshold, probs)

    return structured_input

def extract_most_likely_feature(text, possible_values, threshold, probs):
    """
    Extract the most likely feature value from text based on the list of possible values
    and their corresponding probabilities.
    """
    text = text.lower()
    for value in possible_values:
        if isinstance(value, str) and value.lower() in text:
            # Find the index of the value in the possible_values list
            index = possible_values.index(value)
            # Check the probability
            if probs[index].item() > threshold:
                return [value]  # Return the value as a list to match expected format
    return [None]  # Return None if no value matches with high enough probability

def pick_most_likely_num(feature, probs, numerical_values, used_numbers, threshold, numerical_features):
    """
    Pick the most likely number for the given feature based on model probabilities and
    the list of available numbers.
    """
    feature_presence_prob = probs[numerical_features.index(feature)].item()
    if feature_presence_prob < threshold or not numerical_values:
        return [None]

    # Select the most probable number not already used
    for number in numerical_values:
        if number not in used_numbers:
            used_numbers.append(number)
            return [number]

    return [None]  # Return None if no suitable number is found

def main():
    model, tokenizer = load_model(model_dir)
    
    unique_values_becd_path = os.path.join(model_dir, 'becd_unique_values.pkl')
    unique_values_becd = joblib.load(unique_values_becd_path)
    
    unique_values_carbenmats_path = os.path.join(model_dir, 'carbenmats_unique_values.pkl')
    unique_values_carbenmats = joblib.load(unique_values_carbenmats_path)
    
    # Define numerical features
    numerical_features = ['Total Users', 'Floors Above Ground', 'Floors Below Ground']
    
    text = "I am designing a concrete building in Europe for 20 users with five floors above ground"
    features_becd = parse_input(text, unique_values_becd, tokenizer, model, numerical_features)
    features_carbenmats = parse_input(text, unique_values_carbenmats, tokenizer, model, numerical_features)
    
    print("")
    print("From this text:", text)
    print("")
    print("Extracted Features for BECD:", features_becd)
    print("")
    print("Extracted Features for Carbenmats:", features_carbenmats)

if __name__ == "__main__":
    main()
