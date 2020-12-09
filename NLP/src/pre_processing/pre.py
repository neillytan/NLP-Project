import nltk
#nltk.download('punkt')
import pandas as pd
import numpy as np

def tokenize_file(file_path):
    with open(file_path, "r") as f:
        text = f.read()
        tokens = nltk.word_tokenize(text)
        tokens = set(tokens) #use set to remove duplicates
        tokens = list(tokens)
        return tokens

#tokenize_file("text.txt")

def pre_process(file_path):
    word_list = tokenize_file(file_path)
    words_to_ints = { word_list[i] : i for i in range(len(word_list))}
    ints_to_words = {val : key for key, val in words_to_ints.items()}
    return words_to_ints, ints_to_words

#pre_process("text.txt")
