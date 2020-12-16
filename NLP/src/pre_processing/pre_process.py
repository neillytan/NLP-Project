import nltk
nltk.download('punkt')

def tokenize_file(file_path):
    with open(file_path, "r") as f:
        text = f.read()
        tokens = nltk.word_tokenize(text)
        tokens = [w.lower() for w in tokens if w.isalpha()]
        return tokens


def pre_process(file_path):
    word_list = tokenize_file(file_path)
    word_set = set(word_list)
    words_to_ints = {w: i for i, w in enumerate(word_set)}
    ints_to_words = {val: key for key, val in words_to_ints.items()}
    input_list = [words_to_ints[w] for w in word_list]
    return input_list, words_to_ints, ints_to_words
