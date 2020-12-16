import mxnet as mx
import gensim.downloader as api

def batchify(data, batch_size):
    """
    Reshape data into (num_example, batch_size)
    param data: 1D list of input sequence
    param batch_size: batch size
    return: reshaped input list with shape (num_example, batch_size)
    """
    n_batch = data.shape[0] // batch_size
    data = data[:n_batch * batch_size]
    data = data.reshape((batch_size, n_batch)).T
    return data


def get_batch(source, i):
    """
    Get the i^th batch together with the target
    param source: 2D array with shape (num_example, batch_size)
    param i: which batch to get
    return: tuple (the i^th batch with shape (1, batch_size),
        target for prediction (1D array))
    """
    data = source[i]
    target = source[i + 1]
    return data.reshape((1, len(data))), target.reshape((-1,))

def get_pretrained_weights(idx_word):
    """
    Generate the weight matrix using pretrained GloVe embeddings
    ('glove-twitter-25')
    param idx_word: integer -> word mapping for the RNN
    return: mx array of the weight matrix using pretrained GloVe embeddings
    """
    n = len(idx_word)
    embed_dim = 25
    weights = mx.ndarray.zeros((n, embed_dim))
    print('Start downloading pre-trained vectors, this will take some time')
    glov = api.load("glove-twitter-25")
    print('Pre-trained vectors downloading complete')
    not_in_vocab = 0
    for i in range(n):
        word = idx_word[i]
        try:
            weights[i] = glov[word]
        except: #if not in glove vocabulary
            not_in_vocab += 1
            weights[i] = mx.nd.random.normal(0, 0.1, embed_dim)
    if not_in_vocab > 0:
        print('Warning: {} words not in vocab of pretrained embeddings (glove-twitter-25)'.format(not_in_vocab))
    return weights
