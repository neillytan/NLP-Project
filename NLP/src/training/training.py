import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
from NLP.src.training import utils

    
def train(train_data, args_epochs, args_batch_size, context, trainer):
    loss_progress = []
    for epoch in range(args_epochs):
        total_L = 0.0
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx = context)
        for ibatch in range(0, train_data.shape[0] - 1):
            data, target = utils.get_batch(train_data, ibatch)
            #need this to work, but it doesn't atm
            #hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = utils.loss(output, target)
                L.backward()

            trainer.step(args_batch_size)
            total_L += mx.nd.sum(L).asscalar()
            
        # print and record loss every epoch
        epoch_L = total_L / args_batch_size / ibatch
        print('[Epoch %d] loss %.2f' % (epoch + 1, epoch_L), end='\r')
        loss_progress.append(epoch_L)
        total_L = 0.0
        
    plt.plot(np.arange(args_epochs), loss_progress)
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')