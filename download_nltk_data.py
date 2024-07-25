import nltk
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
nltk_path = os.path.join(current_dir, "nltk.txt")

# Read the NLTK data packages from nltk.txt
with open(nltk_path) as f:
    packages = f.read().splitlines()

# Download each package
for package in packages:
    nltk.download(package)
