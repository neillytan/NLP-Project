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
