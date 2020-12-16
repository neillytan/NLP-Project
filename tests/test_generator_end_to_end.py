"""
Test functionality of generator module
"""
import unittest
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from NLP.src.generator import generator


class UnitTests(unittest.TestCase):
    """
    Test functionality of src/training/utils.py
    """
    # Each method in the class to execute a test
    def test_generator_end_to_end(self):
        """
        test that the generator class works and do not throw exception
        """
        with open("test_sample.txt", "w") as text_file:
            print("Sample text for testing generator, Sample text for testing generator,"
                  "Sample text for testing generator,Sample text for testing generator,"
                  "Sample text for testing generator,Sample text for testing generator", file=text_file)
        try:
            g = generator.Generator('test_sample.txt', epoch=30, rnn_type='lstm', num_embed=5,
                                    num_hidden=5, num_layers=2, dropout=0.1, lr=1, batch_size=2)
            g.decode('text', output_length=10)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()