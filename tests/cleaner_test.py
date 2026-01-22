import hashlib
import unittest
import pandas as pd
from src.etl.dataframe.cleaner import compute_hash, Cleaner


class TestCleanFunctions(unittest.TestCase):

    def test_compute_hash(self):
        self.assertEqual(compute_hash('test'), hashlib.sha1('test'.encode()).hexdigest())


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

    def test_column_names_to_pascal_case(self):
        expected_columns = ['MixedCase', 'WithSpaces', 'WithDollarsSpecialNumChars']
        Cleaner.column_names_to_pascal_case(self.df)
        self.assertEqual(expected_columns, self.df.columns.tolist())

    def test_parse_numbers(self):
        df = pd.DataFrame({
            'ints': ['$1,000', '2,000', '3,000', '244385297.0'],
            'floats': ['$1,000.55', '2,000.66', '3,000.33', '1209347']
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

    def test_clean_all_types(self):
        df = pd.DataFrame({
            'numbers': ['$1,000', '248166676', '3,000', '%100'],
            'dates': ['2021-01-01', '01/02/2021', '2021-01-01', '01/02/2021'],
            'bools': ['yes', 'no', 'true', 'false'],
            'empty': [None, None, None, None]
        })
        clean_df = Cleaner.clean_all_types(df)
        self.assertEqual(clean_df['numbers'].dtype, pd.Int64Dtype.name)
        self.assertEqual(clean_df['dates'].dtype, 'datetime64[ns]')
        self.assertEqual(clean_df['bools'].dtype, pd.BooleanDtype.name)

    def test_clean_df(self):
        df = pd.DataFrame({
            'numbers': [None, '248166676', '3,000', '%100'],
            'bools': [None, 'no', 'true', 'false'],
            'empty': [None, None, None, None]
        })
        clean_df = Cleaner.clean_df(df)
        self.assertEqual(clean_df.shape, (3, 2))


    def test_generate_hash_column(self):
        df = pd.DataFrame({
            'col1': ['value1', 'value2'],
            'col2': ['value3', 'value4']
        })
        expected_hash = df['col1'].apply(str) + df['col2'].apply(str)
        expected_hash = expected_hash.apply(compute_hash)
        expected_hash.name = 'hash'
        result_df = Cleaner.generate_hash_column(df, ['col1', 'col2'], 'hash')
        result_series = result_df['hash']
        pd.testing.assert_series_equal(result_series, expected_hash)

    def test_coalesce_columns(self):
        df = pd.DataFrame({
            'col1': [None, 'B'],
            'col2': ['A', None],
        })
        expected_df = pd.DataFrame({
            'col1': ['A', 'B']
        })
        result_df = Cleaner.coalesce_columns(df, ['col1', 'col2'], 'col1',drop=True)
        self.assertEqual(result_df.shape, (2, 1))
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_parse_numbers_with_empty_strings(self):
        df = pd.DataFrame({
            'ints': ['100', '', '200', '   '],
            'floats': ['1.5', '', '2.5', '   ']
        })
        clean_df = Cleaner.clean_numbers(df)
        # With None values, clean_numbers returns float64 (standard pandas behavior)
        # Use convert_dtypes() or clean_all_types() for nullable Int64
        self.assertEqual(clean_df['ints'].dtype, 'float64')
        self.assertEqual(clean_df['floats'].dtype, 'float64')
        # Verify empty strings become NaN
        self.assertTrue(pd.isna(clean_df['ints'].iloc[1]))
        self.assertTrue(pd.isna(clean_df['ints'].iloc[3]))
        self.assertTrue(pd.isna(clean_df['floats'].iloc[1]))
        self.assertTrue(pd.isna(clean_df['floats'].iloc[3]))
        # Verify valid values are parsed correctly
        self.assertEqual(clean_df['ints'].iloc[0], 100.0)
        self.assertEqual(clean_df['floats'].iloc[0], 1.5)

    def test_clean_bools_with_empty_strings(self):
        df = pd.DataFrame({
            'bools': ['yes', '', 'no', '   ']
        })
        clean_df = Cleaner.clean_bools(df)
        # With None values, clean_bools returns object dtype (standard pandas behavior)
        # Use convert_dtypes() or clean_all_types() for nullable boolean
        self.assertEqual(clean_df['bools'].dtype, 'object')
        # Verify empty strings become None
        self.assertIsNone(clean_df['bools'].iloc[1])
        self.assertIsNone(clean_df['bools'].iloc[3])
        # Verify valid values are parsed correctly
        self.assertTrue(clean_df['bools'].iloc[0])
        self.assertFalse(clean_df['bools'].iloc[2])

    def test_clean_dates_with_empty_strings(self):
        df = pd.DataFrame({
            'dates': ['2021-01-01', '', '2021-02-02', '   ']
        })
        clean_df = Cleaner.clean_dates(df)
        self.assertEqual(clean_df['dates'].dtype, 'datetime64[ns]')
        self.assertTrue(pd.isna(clean_df['dates'].iloc[1]))
        self.assertTrue(pd.isna(clean_df['dates'].iloc[3]))

    def test_clean_all_types_with_empty_strings(self):
        df = pd.DataFrame({
            'numbers': ['100', '', '200', '   '],
            'bools': ['yes', '', 'no', '   '],
            'dates': ['2021-01-01', '', '2021-02-02', '   ']
        })
        clean_df = Cleaner.clean_all_types(df)
        self.assertEqual(clean_df['numbers'].dtype, pd.Int64Dtype.name)
        self.assertEqual(clean_df['bools'].dtype, pd.BooleanDtype.name)
        self.assertEqual(clean_df['dates'].dtype, 'datetime64[ns]')
        # Verify empty strings became None/NaN
        self.assertTrue(pd.isna(clean_df['numbers'].iloc[1]))
        self.assertTrue(pd.isna(clean_df['bools'].iloc[1]))
        self.assertTrue(pd.isna(clean_df['dates'].iloc[1]))


if __name__ == '__main__':
    unittest.main()
