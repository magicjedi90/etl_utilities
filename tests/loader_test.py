import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.etl.database.loader import Loader, ExtraColumnsException, ColumnDataException


class TestLoader(unittest.TestCase):

    @patch('src.etl.database.loader.insert_to_mssql_db')
    def test_insert_to_mssql_table(self, mock_insert):
        cursor = Mock()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.7]
        })
        schema = 'dbo'
        table = 'test_table'

        Loader.insert_to_mssql_table(cursor, df, schema, table)

        self.assertTrue(mock_insert.called)
        self.assertGreater(mock_insert.call_count, 0)

    @patch('pandas.read_sql')
    def test_validate_mssql_upload(self, mock_read_sql):
        connection = Mock()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.7]
        })
        schema = 'dbo'
        table = 'test_table'

        mock_read_sql.return_value = pd.DataFrame({
            'COLUMN_NAME': ['id', 'name', 'value'],
            'DATA_TYPE': ['int', 'varchar', 'float'],
            'CHARACTER_MAXIMUM_LENGTH': [None, 50, None],
            'NUMERIC_PRECISION': [10, None, 10]
        })

        try:
            Loader.validate_mssql_upload(connection, df, schema, table)
        except (ExtraColumnsException, ColumnDataException):
            self.fail("validate_mssql_upload raised an exception unexpectedly")


if __name__ == '__main__':
    unittest.main()
