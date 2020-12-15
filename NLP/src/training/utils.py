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


# Compute loss from data_source and the current net (why did I need this?)
"""def get_loss(data_source, model, loss, context=mx.cpu()):

    Compute loss using passed loss function on data_source
    The data passed in is assumed to be of shape
    (num_example, batch_size) so we can infer batch size

    total_loss = 0.0
    n_total = 0
    hidden = model.begin_state(func=mx.nd.zeros,
                               batch_size=data_source.shape[1], ctx=context)
    for i in range(0, data_source.shape[0] - 1):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_loss += mx.nd.sum(L).asscalar()
        n_total += L.size
    return total_loss / n_total
"""