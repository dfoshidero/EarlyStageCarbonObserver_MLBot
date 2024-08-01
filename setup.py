import nltk
import spacy
from sentence_transformers import SentenceTransformer

# Download NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

# Download spaCy model
spacy.cli.download("en_core_web_trf")

# Load Sentence-Transformer model to ensure it's downloaded
model = SentenceTransformer("all-mpnet-base-v2")
