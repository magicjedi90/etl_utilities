import unittest
from unittest.mock import Mock, MagicMock

import pandas as pd
from sqlalchemy import PoolProxiedConnection
from src.etl.database.differentiator import Differentiator
from src.etl.database.utils import DatabaseUtils


class TestDifferentiator(unittest.TestCase):

    def setUp(self):
        self.mock_connection = Mock(spec=PoolProxiedConnection)
        self.mock_db_utils = Mock(spec=DatabaseUtils)
        self.differentiator = Differentiator(self.mock_connection, similarity_threshold=0.8)
        self.differentiator.db_utils = self.mock_db_utils

    def test_find_table_similarities(self):
        self.mock_db_utils.get_column_names.side_effect = [
            ['col1', 'col2'],  # source_columns
            ['col2', 'col3']  # target_columns
        ]
        self.mock_db_utils.get_column_data.side_effect = [
            pd.Series([1, 2, 3]),  # col1 data
            pd.Series([4, 5, 6]),  # col2 data
            pd.Series([4, 5, 6]),  # col2 data (target)
            pd.Series([7, 8, 9])  # col3 data (target)
        ]

        similarity_df, same_name_df, unique_df = self.differentiator.find_table_similarities(
            "source_schema", "source_table", "target_schema", "target_table"
        )

        self.mock_db_utils.get_column_names.assert_called_with("target_schema", "target_table")
        self.assertIsInstance(similarity_df, pd.DataFrame)
        self.assertIsInstance(same_name_df, pd.DataFrame)
        self.assertIsInstance(unique_df, pd.DataFrame)

    def test_find_table_similarities_no_similarities(self):
        self.mock_db_utils.get_column_names.side_effect = [
            ['col1'],  # source_columns
            ['col2']  # target_columns
        ]
        self.mock_db_utils.get_column_data.side_effect = [
            pd.Series([1, 2, 3]),  # col1 data
            pd.Series([4, 5, 6])  # col2 data
        ]

        similarity_df, same_name_df, unique_df = self.differentiator.find_table_similarities(
            "source_schema", "source_table", "target_schema", "target_table"
        )

        self.assertTrue(similarity_df.empty)
        self.assertTrue(same_name_df.empty)
        self.assertFalse(unique_df.empty)

    def test_find_schema_similarities(self):
        self.mock_db_utils.get_table_list.return_value = ["table1", "table2"]
        self.differentiator.find_table_similarities = MagicMock(return_value=(
            pd.DataFrame([{'source_table': 'table1', 'target_table': 'table2', 'similarity': 0.85}]),
            pd.DataFrame([{'column_name': 'col1'}]),
            pd.DataFrame([{'column_name': 'unique_col'}])
        ))

        schema_same_name, schema_similarity, schema_unique = self.differentiator.find_schema_similarities("test_schema")

        self.mock_db_utils.get_table_list.assert_called_with("test_schema")
        self.assertIsInstance(schema_same_name, pd.DataFrame)
        self.assertIsInstance(schema_similarity, pd.DataFrame)
        self.assertIsInstance(schema_unique, pd.DataFrame)

    def test_create_dataframes(self):
        same_name_columns = [{'source_table': 'table1', 'target_table': 'table2', 'column_name': 'col1'}]
        similar_columns = [{'source_table': 'table1', 'source_column': 'col1', 'target_table': 'table2',
                            'target_column': 'col2', 'similarity': 0.9}]
        unique_source_columns = [{'table_name': 'table1', 'column_name': 'col3'}]
        unique_target_columns = [{'table_name': 'table2', 'column_name': 'col4'}]

        similarity_df, same_name_df, unique_df = Differentiator._create_dataframes(
            same_name_columns, similar_columns, unique_source_columns, unique_target_columns
        )

        self.assertIsInstance(similarity_df, pd.DataFrame)
        self.assertIsInstance(same_name_df, pd.DataFrame)
        self.assertIsInstance(unique_df, pd.DataFrame)
