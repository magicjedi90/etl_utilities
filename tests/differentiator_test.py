import unittest
from unittest.mock import Mock

import pandas as pd
from src.etl.database.differentiator import Differentiator
from src.etl.database.utils import DatabaseUtils


class TestDifferentiator(unittest.TestCase):
    def setUp(self):
        self.mock_connection = Mock()
        self.mock_db_utils = Mock(spec=DatabaseUtils)
        self.mock_logger = Mock()
        self.differentiator = Differentiator(self.mock_connection, similarity_threshold=0.8)
        self.differentiator.db_utils = self.mock_db_utils

    def test_find_table_similarities(self):
        self.mock_db_utils.get_column_names_and_types.side_effect = [
            pd.DataFrame({'name': ['col1', 'col2'], 'type': ['int', 'int']}),
            pd.DataFrame({'name': ['col2', 'col3'], 'type': ['int', 'int']}),
        ]
        self.mock_db_utils.get_column_data.side_effect = [
            pd.Series([1, 2, 3]),
            pd.Series([4, 5, 6]),
            pd.Series([4, 5, 6]),
            pd.Series([7, 8, 9]),
        ]

        similarity_df, same_name_df, unique_df = self.differentiator.find_table_similarities(
            'schema1', 'table1', 'schema2', 'table2'
        )

        self.assertIsInstance(similarity_df, pd.DataFrame)
        self.assertIsInstance(same_name_df, pd.DataFrame)
        self.assertIsInstance(unique_df, pd.DataFrame)

    def test_find_table_similarities_no_common_columns(self):
        self.mock_db_utils.get_column_names_and_types.side_effect = [
            pd.DataFrame({'name': ['col1', 'col2'], 'type': ['int', 'int']}),
            pd.DataFrame({'name': ['col3', 'col4'], 'type': ['int', 'int']}),
        ]
        self.mock_db_utils.get_column_data.side_effect = [
            pd.Series([1, 2, 3]),
            pd.Series([4, 5, 6]),
            pd.Series([7, 8, 9]),
            pd.Series([10, 11, 12]),
        ]

        similarity_df, same_name_df, unique_df = self.differentiator.find_table_similarities(
            'schema1', 'table1', 'schema2', 'table2'
        )

        self.assertEqual(len(similarity_df), 0)
        self.assertEqual(len(same_name_df), 0)
        self.assertGreater(len(unique_df), 0)

    def test_find_schema_similarities(self):
        self.mock_db_utils.get_table_list.return_value = ['table1', 'table2', 'table3']
        self.mock_db_utils.get_column_names_and_types.side_effect = [
            pd.DataFrame({'name': ['col1', 'col2'], 'type': ['int', 'int']}),
            pd.DataFrame({'name': ['col2', 'col3'], 'type': ['int', 'int']}),
            pd.DataFrame({'name': ['col1', 'col2'], 'type': ['int', 'int']}),
            pd.DataFrame({'name': ['col1', 'col4'], 'type': ['int', 'int']}),
            pd.DataFrame({'name': ['col2', 'col3'], 'type': ['int', 'int']}),
            pd.DataFrame({'name': ['col1', 'col4'], 'type': ['int', 'int']}),
        ]
        self.mock_db_utils.get_column_data.side_effect = [
            pd.Series([1, 2, 3]),
            pd.Series([4, 5, 6]),
            pd.Series([4, 5, 6]),
            pd.Series([7, 8, 9]),
            pd.Series([1, 2, 3]),
            pd.Series([4, 5, 6]),
            pd.Series([1, 2, 3]),
            pd.Series([10, 11, 12]),
            pd.Series([4, 5, 6]),
            pd.Series([7, 8, 9]),
            pd.Series([1, 2, 3]),
            pd.Series([10, 11, 12]),
        ]

        schema_same_name, schema_similarity, schema_unique = self.differentiator.find_schema_similarities('schema')

        self.assertIsInstance(schema_same_name, pd.DataFrame)
        self.assertIsInstance(schema_similarity, pd.DataFrame)
        self.assertIsInstance(schema_unique, pd.DataFrame)

    def test_find_schema_similarities_empty_schema(self):
        self.mock_db_utils.get_table_list.return_value = []

        schema_same_name, schema_similarity, schema_unique = self.differentiator.find_schema_similarities('schema')

        self.assertIsNone(schema_same_name)
        self.assertIsNone(schema_similarity)
        self.assertIsNone(schema_unique)


if __name__ == '__main__':
    unittest.main()
