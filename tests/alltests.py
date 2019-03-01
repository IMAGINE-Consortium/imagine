"""
dash through all tests
"""

import unittest
import os

loader = unittest.TestLoader()
start_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
suite = loader.discover(start_dir, '*tests.py')

runner = unittest.TextTestRunner()
runner.run(suite)
