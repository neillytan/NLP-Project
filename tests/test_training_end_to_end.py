'''
Test functionality of knn regression module
'''
import unittest
import os,sys,inspect
import numpy as np
import mxnet as mx
from mxnet import gluon

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from NLP.src.training import model, training, utils


class UnitTests(unittest.TestCase):
    '''
    Test functionality of src/training/utils.py
    '''
    # Each method in the class to execute a test
    def test_training_end_to_end(self):
        '''
        test that the batchify method produces correct
        output dimension
        '''
        context = mx.cpu()
        seq = mx.nd.array(np.arange(100))
        vocab_size = 100
        #model constants
        num_embed = 5
        num_hidden = 5
        num_layers = 1

        #training constants
        args_lr = 1
        args_epochs = 10
        args_batch_size = 32
        train_data = utils.batchify(seq, 
            args_batch_size).as_in_context(context)

        model_ = model.RNNModel(mode='gru', vocab_size=vocab_size, num_embed=num_embed, num_hidden=num_hidden,
                                num_layers=num_layers, dropout=0)
        model_.collect_params().initialize(mx.init.Xavier(), ctx=context)
        trainer = gluon.Trainer(model_.collect_params(), 'sgd',
                            {'learning_rate': args_lr, 'momentum': 0, 'wd': 0})
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        try:
            training.train(train_data=train_data, model=model_,
                           trainer=trainer, loss=loss, epochs=args_epochs,
                           batch_size=args_batch_size,
                           context=context)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
