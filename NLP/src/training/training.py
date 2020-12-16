import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from mxnet import autograd

from NLP.src.training import utils


def train(train_data, model, loss, epochs, batch_size,
          context, trainer, freeze_embedding=False):
    """
    Train an RNN model, given the input list
    param train_data: training data, 2D mx array of size (n, batch_size)
    param model: RNN model
    param loss: loss function, e.g. gluon.loss.SoftmaxCrossEntropyLoss()
    param epochs: number of epochs to run
    param batch_size: batch size
    param context: cpu or (which) gpu
    param trainer: trainer (optimizer)
    param freeze_embedding: freeze training of embedding layers (input and output)
    """
    if freeze_embedding:
        for param in model.encoder.collect_params().values():
            param.grad_req = 'null'
        for param in model.decoder.collect_params().values():
            param.grad_req = 'null'
    loss_progress = []
    for epoch in range(epochs):
        total_loss = 0.0
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
        for i in range(0, train_data.shape[0] - 1):
            data, target = utils.get_batch(train_data, i)
            # need this to work? but it doesn't atm
            # hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            trainer.step(batch_size)
            total_loss += mx.nd.sum(L).asscalar()

        # print and record loss every epoch
        epoch_loss = total_loss / batch_size / i
        print('[Epoch %d] loss %.2f' % (epoch + 1, epoch_loss), end='\r')
        loss_progress.append(epoch_loss)

    plt.plot(np.arange(epochs), loss_progress)
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
