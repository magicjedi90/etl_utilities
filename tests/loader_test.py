# test_loader.py
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from src.etl.database.loader import Loader
from sqlalchemy.engine.interfaces import DBAPICursor


class TestLoader(unittest.TestCase):

    def setUp(self):
        self.cursor = Mock(spec=DBAPICursor)
        self.df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        self.schema = 'test_schema'
        self.table = 'test_table'
        self.loader = Loader(self.cursor, self.df, self.schema, self.table)
        self.column_string = 'col1, col2'
        self.placeholders = ['%s', '%s']
        self.location = f"{self.schema}.{self.table}"

    @patch('src.etl.database.loader.insert_to_db')
    def test_insert_to_table_with_small_dataframe(self, mock_insert_to_db):
        Loader._insert_to_table(self.column_string, self.cursor, self.df, self.location, self.placeholders)
        mock_insert_to_db.assert_called_once()
        self.assertEqual(mock_insert_to_db.call_args[0][0], self.column_string)
        self.assertEqual(mock_insert_to_db.call_args[0][1], self.cursor)

    @patch('src.etl.database.loader.insert_to_db')
    def test_insert_to_table_with_empty_dataframe(self, mock_insert_to_db):
        empty_df = pd.DataFrame(columns=['col1', 'col2'])
        Loader._insert_to_table(self.column_string, self.cursor, empty_df, self.location, self.placeholders)
        mock_insert_to_db.assert_not_called()

    @patch('src.etl.database.loader.insert_to_db')
    def test_insert_to_table_raises_exception(self, mock_insert_to_db):
        error_message = "test error"
        mock_insert_to_db.side_effect = Exception(error_message)
        with self.assertRaises(Exception) as context:
            Loader._insert_to_table(self.column_string, self.cursor, self.df, self.location, self.placeholders)
        self.assertTrue(error_message in str(context.exception))


if __name__ == '__main__':
    unittest.main()
