import unittest
import pandas as pd
from src.etl.dataframe.analyzer import Analyzer


class TestAnalyzer(unittest.TestCase):

    def setUp(self):
        # Setup sample DataFrames for testing
        self.df_single_id = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Eve', 'Bob', 'Charlie', 'David', 'Eve'],
            'empty': [None, None, None, None, None]
        })

        self.df_id_pairs = pd.DataFrame({
            'first': [1, 1, 2, 2, 3],
            'second': [1, 1, 2, 2, 3],
            'other': [10, 20, 30, 40, 10]
        })

        self.df_no_id_pairs = pd.DataFrame({
            'first': [1, 1, 2, 2, 3],
            'second': [1, 1, 2, 2, 3]
        })

        self.df = pd.DataFrame({
            'A': ['foo', 'bar', 'baz', 'foo', 'bar', 'baz', 'foo', 'foo'],
            'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
            'C': pd.Series(list(range(8))),
            'D': pd.Series(list(range(8))),
        })

    def test_find_unique_columns(self):
        # Test case with single ID candidate
        expected_candidates = ['id']
        actual_candidates = Analyzer.find_unique_columns(self.df_single_id)
        self.assertEqual(expected_candidates, actual_candidates)

        # Test case with no single ID candidate
        expected_candidates = []
        actual_candidates = Analyzer.find_unique_columns(self.df_no_id_pairs)
        self.assertEqual(expected_candidates, actual_candidates)

    def test_find_unique_column_pairs(self):
        # Test case with ID pair candidates
        expected_pairs = [('first', 'other'), ('second', 'other')]
        actual_pairs = Analyzer.find_unique_column_pairs(self.df_id_pairs)
        self.assertEqual(expected_pairs, actual_pairs)

        # Test case with no ID pair candidates
        expected_pairs = []
        actual_pairs = Analyzer.find_unique_column_pairs(self.df_no_id_pairs)
        self.assertEqual(expected_pairs, actual_pairs)

    def test_find_empty_columns(self):
        # Test case with single ID candidate
        expected_candidates = ['empty']
        actual_candidates = Analyzer.find_empty_columns(self.df_single_id)
        self.assertEqual(expected_candidates, actual_candidates)

    def test_find_categorical_columns_uniformly_distributed(self):
        result = Analyzer.find_categorical_columns(self.df, 0.5)
        expected = ['A', 'B']
        self.assertListEqual(result, expected)

    def test_find_categorical_columns_high_threshold(self):
        result = Analyzer.find_categorical_columns(self.df, 1)
        expected = ['A', 'B', 'C', 'D']
        self.assertListEqual(result, expected)

    def test_find_categorical_columns_low_threshold(self):
        result = Analyzer.find_categorical_columns(self.df, 0.25)
        expected = []
        self.assertListEqual(result, expected)

    def test_find_categorical_columns_invalid_threshold(self):
        with self.assertRaises(ValueError):
            Analyzer.find_categorical_columns(self.df, -0.5)
        with self.assertRaises(ValueError):
            Analyzer.find_categorical_columns(self.df, 1.5)

if __name__ == '__main__':
    unittest.main()
