import hashlib
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.etl.dataframe.cleaner import Parser


class TestCleanFunctions(unittest.TestCase):

    def test_parse_float(self):
        self.assertEqual(Parser.parse_float('123.45'), 123.45)
        self.assertEqual(Parser.parse_float('$1,234.56'), 1234.56)
        self.assertIsNone(Parser.parse_float(None))

    def test_parse_date(self):
        self.assertEqual(Parser.parse_date('2021-01-01'), datetime(2021, 1, 1))
        self.assertEqual(Parser.parse_date('01/01/2021'), datetime(2021, 1, 1))
        self.assertIsNone(Parser.parse_date(None))
        self.assertIsNone(Parser.parse_date(np.nan))

    def test_parse_int(self):
        self.assertEqual(Parser.parse_integer(123), 123)
        self.assertEqual(Parser.parse_integer(123.0), 123)
        with self.assertRaises(ValueError):
            Parser.parse_integer(123.45)
        self.assertIsNone(Parser.parse_integer(None))
        self.assertIsNone(Parser.parse_integer(np.nan))


if __name__ == '__main__':
    unittest.main()
