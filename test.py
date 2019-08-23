#!venv/bin/python
import unittest

from test_dicom_util import *
from test_scalar_analysis import *
from test_thermometry import *
from test_voi import test_voi

class TestVOI(unittest.TestCase):
    # this isn't really a unit test, but convenient to wrap up with rest of unit tests
    def test_voi(self):
        self.assertEqual(test_voi(), 0)


if __name__ == "__main__":
    # running test.py won't show logging.info statements by default, only logging.warn
    unittest.main()
