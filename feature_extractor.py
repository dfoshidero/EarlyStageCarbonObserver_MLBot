import os
import re
import json
import joblib
import spacy
import nltk
import random
import multiprocessing
import numpy as np

from word2number import w2n
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# Load pre-trained NER model (spaCy example)
nlp = spacy.load("en_core_web_trf")

# Load pre-trained sentence transformer model for semantic similarity
model = SentenceTransformer("all-mpnet-base-v2")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

numerical_features = [
    "Gross Internal Area (m2)",
    "Building Perimeter (m)",
    "Building Footprint (m2)",
    "Building Width (m)",
    "Floor-to-Floor Height (m)",
    "Storeys Above Ground",
    "Storeys Below Ground",
    "Glazing Ratio (%)",
]

SIMILARITY_THRESHOLD = 0.7  # Define a similarity threshold


def load_json(json_path):
    with open(json_path, "r") as f:
        json_file = json.load(f)
    return json_file


def load_unique_values(model_dir):
    path_unq_vals = os.path.join(model_dir, "unique_values.pkl")
    unique_values = joblib.load(path_unq_vals)
    return unique_values


def get_related_terms(word, synonym_dict):
    related_terms = set()
    for key, synonyms in synonym_dict.items():
        if word.lower() == key.lower() or word.lower() in map(str.lower, synonyms):
            related_terms.add(key)
            related_terms.update(synonyms)
    return related_terms


def preprocess_text(text, synonym_dict):
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word.lower() not in stop_words
    ]

    processed_tokens = []

    for token in tokens:
        related_terms = get_related_terms(token, synonym_dict)
        if related_terms:
            # Add the original token and its related terms as separate tokens
            processed_tokens.extend(related_terms)
        else:
            processed_tokens.append(token)

    return processed_tokens


def filter_pos_tags(tokens):
    tagged_tokens = nltk.pos_tag(tokens)
    filtered_tokens = [
        word
        for word, pos in tagged_tokens
        if pos.startswith("NN") or pos.startswith("JJ")
    ]
    return filtered_tokens


def find_nearest_word(text, target_word, window_size=5):
    words = text.split()
    if target_word in words:
        target_idx = words.index(target_word)
        start_idx = max(0, target_idx - window_size)
        end_idx = min(len(words), target_idx + window_size + 1)
        return words[start_idx:end_idx]
    return []


def apply_building_logic(features):
    # Extract features for easier reference
    sector = features.get("Sector")
    sub_sector = features.get("Sub-Sector")
    storeys_below = features.get("Storeys Below Ground", 0)
    timber_joists = features.get("Joisted Floors Material")

    if storeys_below == 0:
        features["Basement Walls Material"] = None

    if sector == "Residential" and timber_joists:
        features["Joisted Floors"] = "Timber Joists (Domestic)"
    elif sector == "Non-residential" and timber_joists:
        features["Joisted Floors"] = "Timber Joists (Office)"

    if sector == "Residential" and sub_sector == "Non-residential":
        features["Sub-Sector"] = None
    elif sector == "Non-residential":
        features["Sub-Sector"] = "Non-residential"

    return features


def random_choice_conflicting_features(features, input_text):
    input_text_lower = input_text.lower()

    has_piles = features.get("Piles") is not None
    if not has_piles:
        features["Pile Caps Material"] = None
        features["Capping Beams Material"] = None

    # Choose "Raft" or "Pile Caps"/"Capping Beams" based on input text
    if "raft" in input_text_lower:
        features["Pile Caps Material"] = None
        features["Capping Beams Material"] = None
    elif "pile caps" in input_text_lower or "capping beams" in input_text_lower:
        features["Raft Material"] = None
    elif features.get("Raft Material") and (
        features.get("Pile Caps Material") or features.get("Capping Beams Material")
    ):
        if random.choice([True, False]):
            features["Pile Caps Material"] = None
            features["Capping Beams Material"] = None
        else:
            features["Raft Material"] = None

    # Choose "Joisted Floors" or "Floor Slab" based on input text
    if "joists" in input_text_lower:
        features["Floor Slab Material"] = None
    elif "slab" in input_text_lower:
        features["Joisted Floors Material"] = None
    elif features.get("Joisted Floors Material") and features.get(
        "Floor Slab Material"
    ):
        if random.choice([True, False]):
            features["Floor Slab Material"] = None
        else:
            features["Joisted Floors Material"] = None

    return features


def extract_feature_values(
    input_text,
    unique_values,
    numerical_features,
    synonym_dict,
    threshold=SIMILARITY_THRESHOLD,
):
    doc = nlp(input_text)
    explicit_features, filtered_text = extract_explicit_features(
        input_text, unique_values, synonym_dict, model, numerical_features
    )
    doc_filtered = nlp(filtered_text)
    ner_entities = [ent.text for ent in doc_filtered.ents]

    preprocessed_tokens = preprocess_text(filtered_text, synonym_dict)
    filtered_tokens = filter_pos_tags(preprocessed_tokens)

    candidates = set(ner_entities + filtered_tokens)

    feature_matches = explicit_features.copy()
    matched_features = set(explicit_features.keys())

    # Process general cases for remaining features
    for feature, values in unique_values.items():
        if (
            feature in numerical_features
            or feature == "Embodied Carbon (kgCO2e/m2)"
            or feature in feature_matches
        ):
            continue

        unique_embeddings = model.encode(values)
        candidate_embeddings = model.encode(list(candidates))

        best_match = None
        highest_score = float("-inf")

        for candidate, candidate_embedding in zip(candidates, candidate_embeddings):
            similarities = util.pytorch_cos_sim(candidate_embedding, unique_embeddings)
            max_similarity = similarities.max().item()
            if max_similarity > highest_score:
                highest_score = max_similarity
                best_match = values[similarities.argmax().item()]

        if highest_score >= threshold:
            feature_matches[feature] = best_match
        else:
            feature_matches[feature] = None

    # Apply the building logic rules
    feature_matches = apply_building_logic(feature_matches)

    # Randomly choose between conflicting features
    feature_matches = random_choice_conflicting_features(feature_matches, input_text)

    return feature_matches


def extract_numerical_feature(text, label, feature_keywords):
    pattern = re.compile(
        r"(\b\d+\.?\d*(?:sqm|sqft|km|m|cm|mm|in|ft|yd|mg|g|kg|lb|oz|liters|ml|gal|kw|hp)?\b)",
        re.IGNORECASE,
    )
    feature_numbers = {feature: [] for feature in feature_keywords.keys()}

    words = text.split()
    converted_text = []
    for word in words:
        try:
            number = w2n.word_to_num(word)
            converted_text.append(str(number))
        except ValueError:
            converted_text.append(word)
    updated_text = " ".join(converted_text)
    words = updated_text.split()

    for i, word in enumerate(words):
        for feature, keywords in feature_keywords.items():
            if any(kw in word.lower() for kw in keywords):
                window = words[max(i - 3, 0) : min(i + 4, len(words))]
                for w in window:
                    match = pattern.match(w)
                    if match:
                        # Extract the numerical value
                        num_str = match.group(1)
                        # Remove any non-numeric characters for conversion
                        num_val = re.sub(r"[^\d.]", "", num_str)
                        feature_numbers[feature].append(float(num_val))

    for feature in feature_numbers:
        if feature_numbers[feature]:
            feature_numbers[feature] = max(
                set(feature_numbers[feature]), key=feature_numbers[feature].count
            )
        else:
            feature_numbers[feature] = "None"

    # Special rule: Set "Storeys Below Ground" to 1 if "a basement" is mentioned
    if "a basement" in text.lower():
        if feature_numbers["Storeys Below Ground"] == "None":
            feature_numbers["Storeys Below Ground"] = 1

    return feature_numbers


def extract_explicit_features(
    input_text,
    unique_values,
    synonym_dict,
    model,
    numerical_features,
    threshold=SIMILARITY_THRESHOLD,
):
    explicit_features = {}
    word_count = Counter(input_text.lower().split())
    context_count = Counter()

    for feature in unique_values.keys():
        if feature in numerical_features or feature == "Embodied Carbon (kgCO2e/m2)":
            continue

        feature_cleaned = feature.lower().replace(" material", "")
        pattern = rf"\b{feature_cleaned}\b"
        matches = re.finditer(pattern, input_text, re.IGNORECASE)

        for match in matches:
            nearby_words = find_nearest_word(input_text, match.group(), window_size=5)
            preprocessed_tokens = preprocess_text(" ".join(nearby_words), synonym_dict)
            filtered_tokens = filter_pos_tags(preprocessed_tokens)

            if filtered_tokens:
                unique_embeddings = model.encode(unique_values[feature])
                candidate_embeddings = model.encode(filtered_tokens)

                best_match = None
                highest_score = float("-inf")
                original_word = None

                for candidate, candidate_embedding in zip(
                    filtered_tokens, candidate_embeddings
                ):
                    similarities = util.pytorch_cos_sim(
                        candidate_embedding, unique_embeddings
                    )
                    max_similarity = similarities.max().item()
                    if max_similarity > highest_score:
                        highest_score = max_similarity
                        best_match = unique_values[feature][
                            similarities.argmax().item()
                        ]
                        original_word = candidate

                if highest_score >= threshold:
                    explicit_features[feature] = best_match
                    context_count.update([original_word.lower()])
                    break

    filtered_words = [
        word
        for word in input_text.split()
        if context_count[word.lower()] < word_count[word.lower()]
    ]
    filtered_text = " ".join(filtered_words)
    return explicit_features, filtered_text


def extract(input_text):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "model")
    synonyms_path = os.path.join(current_dir, "config/synonyms.json")
    numerical_keywords_path = os.path.join(
        current_dir, "config/numerical_keywords.json"
    )

    unique_values = load_unique_values(model_dir)
    synonym_dict = load_json(synonyms_path)
    numerical_keywords = load_json(numerical_keywords_path)

    feature_values = extract_feature_values(
        input_text,
        unique_values,
        numerical_features,
        synonym_dict,
        SIMILARITY_THRESHOLD,
    )

    for feature in numerical_features:
        numerical_values = extract_numerical_feature(
            input_text, feature, numerical_keywords
        )
        feature_values[feature] = numerical_values[feature]

    # DEBUG
    for feature, value in feature_values.items():
        print(f"{feature}: {value}")
    return feature_values
