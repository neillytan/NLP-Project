"""
Test functionality of training/utils.py
"""
import unittest
import os,sys,inspect
import numpy as np
import mxnet as mx

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from NLP.src.training import utils

class UnitTests(unittest.TestCase):
    """
    Test functionality of src/training/utils.py
    """
    # Each method in the class to execute a test
    def test_batchify_correct_dimension(self):
        """
        test that the batchify method produces correct
        output dimension
        """
        data = mx.ndarray.ones(100)
        batch_size = 2
        res = utils.batchify(data, batch_size)
        self.assertEqual(res.shape, (50, 2))

    def test_get_batch_correct_output(self):
        """
        test that the get_batch returns the correct
        batch data and target
        """
        data = mx.nd.array([[1,2,3],[4,5,6]])

        res = utils.get_batch(data, 0)
        batch, target = res[0].asnumpy(), res[1].asnumpy()  
        np.testing.assert_array_equal(batch, [[1,2,3]])
        np.testing.assert_array_equal(target, [4,5,6])

    # TODO: test and handle edge cases

if __name__ == '__main__':
    unittest.main()
