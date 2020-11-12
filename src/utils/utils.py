import nltk
import pandas as pd
import numpy as np

def tokenize_file(file_path):
    with open(file_path, "r") as f:
        corpus = f.read()
        tokens = nltk.word_tokenize(corpus)
        return tokens
      

def word_idx(word_list):
    word_set = set(word_list)
    word_df = pd.DataFrame(word_set, columns=['word'])
    word_to_idx = {}
    word_to_idx = {
    	**word_to_idx, 
    	**dict(zip(word_df['word'], 
    	np.arange(len(word_df))))
    	}
    return word_df, word_to_idx