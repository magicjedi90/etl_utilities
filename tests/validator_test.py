import unittest
import pandas as pd
from unittest.mock import Mock, patch
from src.etl.database.validator import Validator, ExtraColumnsException, ColumnDataException
from src.etl.dataframe.analyzer import Analyzer

class TestValidator(unittest.TestCase):

    def setUp(self):
        self.connection = Mock()  # Mock your database connection
        self.schema = "test_schema"
        self.table = "test_table"

    @patch('pandas.read_sql')
    def test_validate_upload(self, mock_read_sql):
        connection = Mock()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.3, 30.7],
            'empty': [None, None, None]
        })
        schema = 'dbo'
        table = 'test_table'

        mock_read_sql.return_value = pd.DataFrame({
            'COLUMN_NAME': ['id', 'name', 'value', 'empty'],
            'DATA_TYPE': ['int', 'varchar', 'float', 'varchar'],
            'CHARACTER_MAXIMUM_LENGTH': [None, 50, None, -1],
            'NUMERIC_PRECISION': [10, None, 10, None]
        })

        try:
            Validator.validate_upload(connection, df, schema, table)
        except (ExtraColumnsException, ColumnDataException):
            self.fail("validate_mssql_upload raised an exception unexpectedly")

    def test_truncation_exception(self):
        # Create a DataFrame that will cause truncation
        df = pd.DataFrame({'col1': [10, 2, 3], 'col2': ['a' * 300, 'b' * 300, 'c' * 300]})
        column_info_df = pd.DataFrame({
            'COLUMN_NAME': ['col1', 'col2'],
            'DATA_TYPE': ['int', 'varchar'],
            'CHARACTER_MAXIMUM_LENGTH': [None, 255],
            'NUMERIC_PRECISION': [1, None]
        })

        df_metadata = Analyzer.generate_column_metadata(df, None, None, 2)

        with self.assertRaises(ColumnDataException):
            Validator._validate_column_types(df_metadata, column_info_df)

    def test_extra_columns_exception(self):
        # Create a DataFrame with extra columns
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': [4, 5, 6]})
        column_info_df = pd.DataFrame({
            'COLUMN_NAME': ['col1', 'col2'],
            'DATA_TYPE': ['int', 'varchar'],
            'CHARACTER_MAXIMUM_LENGTH': [None, 255],
            'NUMERIC_PRECISION': [1, None]
        })

        with self.assertRaises(ExtraColumnsException):
            Validator._check_extra_columns(df, column_info_df, self.schema, self.table)


    def test_mismatched_columns(self):
        # Create a DataFrame with mismatched columns
        df = pd.DataFrame({'wrong_type_col': ['one', 'two', 'three']})
        column_info_df = pd.DataFrame({
            'COLUMN_NAME': ['wrong_type_col'],
            'DATA_TYPE': ['int'],
            'CHARACTER_MAXIMUM_LENGTH': [None],
            'NUMERIC_PRECISION': [1]
        })

        df_metadata = Analyzer.generate_column_metadata(df, None, None, 0)

        with self.assertRaises(ColumnDataException):
            Validator._validate_column_types(df_metadata, column_info_df)


if __name__ == '__main__':
    unittest.main()
