import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.etl.database.validator import Validator, ExtraColumnsException, ColumnDataException


class TestValidator(unittest.TestCase):

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
            Validator.validate_mssql_upload(connection, df, schema, table)
        except (ExtraColumnsException, ColumnDataException):
            self.fail("validate_mssql_upload raised an exception unexpectedly")


if __name__ == '__main__':
    unittest.main()
