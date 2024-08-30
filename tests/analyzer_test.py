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

if __name__ == '__main__':
    unittest.main()
