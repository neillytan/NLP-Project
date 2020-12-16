"""
Test functionality of decoder module
"""
import unittest
import os,sys,inspect
import mxnet as mx

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from NLP.src.decoder import decoder
from NLP.src.training import utils, model, training


class UnitTests(unittest.TestCase):
    """
    Test functionality of src/decoder/decoder.py
    """
    # Each method in the class to execute a test
    def test_decoder_end_to_end(self):
        """
        test that the decoder class works and do not throw exception
        since this needs a model, we use a untrained, randomly initialized
        model for testing propose
        """
        vocab_size = 100
        #model constants
        num_embed = 5
        num_hidden = 5
        num_layers = 1

        model_ = model.RNNModel(mode='gru', vocab_size=vocab_size, num_embed=num_embed, num_hidden=num_hidden,
                                num_layers=num_layers, dropout=0)
        model_.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
        try:
            decoder_sample = decoder.Decoder(model_, context=mx.cpu())
            decoder_sample.decode([0], 20, mode='sample', sample_count=10)
            decoder_sample.decode([0], 5, mode='sample', sample_count=1)
            decoder_sample.decode([0], 20, mode='greedy', sample_count=10)
            decoder_sample.decode([0], 2, mode='greedy', sample_count=10)
        except:
            self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()