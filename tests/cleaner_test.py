import hashlib
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from src.etl.dataframe.cleaner import clean_float, clean_date, clean_int, hash_str, Cleaner


class TestCleanFunctions(unittest.TestCase):

    def test_clean_float(self):
        self.assertEqual(clean_float('123.45'), 123.45)
        self.assertEqual(clean_float('$1,234.56'), 1234.56)
        self.assertIsNone(clean_float(None))

    def test_clean_date(self):
        self.assertEqual(clean_date('2021-01-01'), datetime(2021, 1, 1))
        self.assertEqual(clean_date('01/01/2021'), datetime(2021, 1, 1))
        self.assertIsNone(clean_date(None))
        self.assertIsNone(clean_date(np.nan))

    def test_clean_int(self):
        self.assertEqual(clean_int(123), 123)
        self.assertEqual(clean_int(123.0), 123)
        with self.assertRaises(ValueError):
            clean_int(123.45)
        self.assertIsNone(clean_int(None))
        self.assertIsNone(clean_int(np.nan))

    def test_hash_str(self):
        self.assertEqual(hash_str('test'), hashlib.sha1('test'.encode()).hexdigest())


class TestCleaner(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'MixedCase': ['value1', 'value2'],
            'with spaces': ['value3', 'value4'],
            'with$Special#Chars': ['value5', 'value6']
        })

    def test_column_names_to_snake_case(self):
        expected_columns = ['mixed_case', 'with_spaces', 'with_dollars_special_num_chars']
        Cleaner.column_names_to_snake_case(self.df)
        self.assertEqual(expected_columns, self.df.columns.tolist())

    def test_clean_numbers(self):
        df = pd.DataFrame({
            'ints': ['$1,000', '2,000', '3,000'],
            'floats': ['$1,000.55', '2,000.66', '3,000.33']
        })
        clean_df = Cleaner.clean_numbers(df)
        self.assertEqual(clean_df['ints'].dtype, 'int64')
        self.assertEqual(clean_df['floats'].dtype, 'float64')

    def test_clean_dates(self):
        df = pd.DataFrame({
            'dates': ['2021-01-01', '01/02/2021']
        })
        clean_df = Cleaner.clean_dates(df)
        self.assertEqual(clean_df['dates'].dtype, 'datetime64[ns]')

    def test_clean_bools(self):
        df = pd.DataFrame({
            'bools': ['yes', 'no', 'true', 'false']
        })
        clean_df = Cleaner.clean_bools(df)
        self.assertEqual(clean_df['bools'].dtype, 'bool')

    def test_clean_all(self):
        df = pd.DataFrame({
            'numbers': ['$1,000', '2,000', '3,000', '%100'],
            'dates': ['2021-01-01', '01/02/2021', '2021-01-01', '01/02/2021'],
            'bools': ['yes', 'no', 'true', 'false']
        })
        clean_df = Cleaner.clean_all(df)
        self.assertEqual(clean_df['numbers'].dtype, 'int64')
        self.assertEqual(clean_df['dates'].dtype, 'datetime64[ns]')
        self.assertEqual(clean_df['bools'].dtype, 'bool')

    def test_generate_hash_column(self):
        df = pd.DataFrame({
            'col1': ['value1', 'value2'],
            'col2': ['value3', 'value4']
        })
        expected_hash = df['col1'].apply(str) + df['col2'].apply(str)
        expected_hash = expected_hash.apply(hash_str)
        expected_hash.name = 'hash'
        result_df = Cleaner.generate_hash_column(df, ['col1', 'col2'], 'hash')
        result_series = result_df['hash']
        pd.testing.assert_series_equal(result_series, expected_hash)

    def test_coalesce_columns(self):
        df = pd.DataFrame({
            'col1': [None, 'B'],
            'col2': ['A', 'B'],
            'col3': [None, 'C']
        })
        expected_df = pd.DataFrame({
            'col1': [None, 'B'],
            'col2': ['A', 'B'],
            'col3': [None, 'C'],
            'coalesced': ['A', 'B']
        })
        result_df = Cleaner.coalesce_columns(df, ['col1', 'col2'], 'coalesced')
        pd.testing.assert_frame_equal(result_df, expected_df)


if __name__ == '__main__':
    unittest.main()
