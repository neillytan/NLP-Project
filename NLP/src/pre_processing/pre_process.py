"""
This module will process text files to a format that is readable by the
RNN model for later training and decoding propose
"""
import nltk
nltk.download('punkt')

def tokenize_file(file_path):
    """
    Tokenize a text file, ignore stop words
    param file_path: path of the text file to be tokenized
    return: list of words, all lower case
    """
    with open(file_path, "r") as f:
        text = f.read()
        tokens = nltk.word_tokenize(text)
        tokens = [w.lower() for w in tokens if w.isalpha()]
        return tokens


def pre_process(file_path):
    """
    Map the text in the text file to a list of integers
    param file_path: path of the text file to be processed
    return: tuple (text as list of integers, word -> integer dictionary, integer -> word dictionary)
    """
    word_list = tokenize_file(file_path)
    word_set = set(word_list)
    words_to_ints = {w: i for i, w in enumerate(word_set)}
    ints_to_words = {val: key for key, val in words_to_ints.items()}
    input_list = [words_to_ints[w] for w in word_list]
    r1,r2,r3 = input_list, words_to_ints, ints_to_words
    return r1,r2,r3
