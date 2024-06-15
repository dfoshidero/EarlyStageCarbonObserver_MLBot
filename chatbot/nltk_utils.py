# Import relevant packages
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

import numpy as np 


stemmer = PorterStemmer()
# Begin by preprocessing query given by user
def tokenize(query):
    return nltk.word_tokenize(query)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):

    tokenized_sentence = [
        stem(w) for w in tokenized_sentence
        ]
    
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag