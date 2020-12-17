"""
Test functionality of pre-processing module
"""
import unittest
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from NLP.src.pre_processing import pre_process

class UnitTests(unittest.TestCase):
    """
    Tests functionality of tokenize_file functionality
    """
    def test_tokenize_file(self):
        result = pre_process.tokenize_file('text.txt')
        exp_result = ['two', 'words', 'two']
        self.assertEqual(result, exp_result)

    """
    Test functionality of pre_process
    """
    def test_pre_process(self):
        r1, r2, r3 = pre_process.pre_process('text.txt')
        self.assertEqual(r1[0], r1[2])
        self.assertNotEqual(r1[0], r1[1])

if __name__== '__main__':
    unittest.main()
