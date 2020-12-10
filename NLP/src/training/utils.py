import numpy as np
import mxnet as mx

def batchify(data, batch_size):
    """
    Reshape data into (num_example, batch_size)
    """
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

def get_batch(source, i):
    """
    Get the i^th batch together with the target
    """
    data = source[i]
    target = source[i + 1]
    return data.reshape((1,len(data))), target.reshape((-1,))

# Compute loss from data_source and the current net
def eval(data_source, model, loss, context=mx.cpu()):
    """
    Compute loss using passed loss function on data_source
    The data passed in is assumed to be of shape
    (num_example, batch_size) so we can infer batch size
    """
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, 
        batch_size = data_source.shape[1], ctx=context)
    for i in range(0, data_source.shape[0] - 1):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal