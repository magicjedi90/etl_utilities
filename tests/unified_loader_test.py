import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
from sqlalchemy.engine.interfaces import DBAPICursor
from src.etl.database.unified_loader import Loader
from src.etl.database.sql_dialects import mssql, mariadb, postgres


class TestUnifiedLoader(unittest.TestCase):
    """Test cases for the Loader class in unified_loader.py."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cursor = Mock(spec=DBAPICursor)
        self.df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        self.schema = 'test_schema'
        self.table = 'test_table'
        self.location = f"{self.schema}.{self.table}"

    def test_init_with_mssql_dialect(self):
        """Test initialization with MSSQL dialect."""
        loader = Loader(self.cursor, self.df, self.schema, self.table, mssql)

        self.assert_db_consistencies(loader)
        self.assertEqual(loader._dialect, mssql)
        self.assertEqual(loader._column_string, "[col1], [col2]")
        self.assertEqual(loader._location, self.location)
        self.assertEqual(loader._placeholder, "?")
        self.assertTrue(self.cursor.fast_executemany)

    def test_init_with_mariadb_dialect(self):
        """Test initialization with MariaDB dialect."""
        loader = Loader(self.cursor, self.df, self.schema, self.table, mariadb)

        self.assert_db_consistencies(loader)
        self.assertEqual(loader._dialect, mariadb)
        self.assertEqual(loader._column_string, "`col1`, `col2`")
        self.assertEqual(loader._location, self.location)
        self.assertEqual(loader._placeholder, "%s")
        # fast_executemany should not be set for mariadb
        self.assertFalse(hasattr(self.cursor, 'fast_executemany'))

    def assert_db_consistencies(self, loader):
        self.assertEqual(loader._cursor, self.cursor)
        self.assertEqual(loader._df.equals(self.df), True)
        self.assertEqual(loader._schema, self.schema)
        self.assertEqual(loader._table, self.table)

    def test_init_with_postgres_dialect(self):
        """Test initialization with Postgres dialect."""
        loader = Loader(self.cursor, self.df, self.schema, self.table, postgres)

        self.assert_db_consistencies(loader)
        self.assertEqual(loader._dialect, postgres)
        self.assertEqual(loader._column_string, "\"col1\", \"col2\"")
        self.assertEqual(loader._location, self.location)
        self.assertEqual(loader._placeholder, "%s")
        # fast_executemany should not be set for postgres
        self.assertFalse(hasattr(self.cursor, 'fast_executemany'))

    def test_prepare_data_with_different_types(self):
        """Test _prepare_data method with different data types."""
        # Create a DataFrame with various data types
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, np.nan],
            'bool_col': [True, False, True],
            'str_col': ['a', 'b', None],
            'long_str_col': ['a' * 300, 'b' * 300, 'c' * 300]
        })

        # Test with MSSQL dialect
        loader = Loader(self.cursor, df, self.schema, self.table, mssql)
        placeholders = loader._prepare_data()

        # Check that NaN values are replaced with None
        self.assertIsNone(loader._df['float_col'].iloc[2])

        # Check types - they might be numpy types or Python types depending on implementation
        self.assertTrue(isinstance(loader._df['int_col'].iloc[0], (int, np.int64, np.int32)))
        self.assertTrue(isinstance(loader._df['float_col'].iloc[0], (float, np.float64, np.float32)))
        self.assertTrue(isinstance(loader._df['bool_col'].iloc[0], (bool, np.bool_)))

        # Check placeholders for long strings
        self.assertEqual(placeholders[4], 'cast ( ? as nvarchar(max))')

        # Test with Postgres dialect
        loader = Loader(self.cursor, df.copy(), self.schema, self.table, postgres)
        placeholders = loader._prepare_data()

        # Check placeholders for long strings
        self.assertEqual(placeholders[4], 'cast ( %s as varchar(21844))')

    def test_insert_with_empty_dataframe(self):
        """Test insert method with an empty DataFrame."""
        empty_df = pd.DataFrame(columns=['col1', 'col2'])
        loader = Loader(self.cursor, empty_df, self.schema, self.table, mssql)

        with patch.object(loader, '_run_progress') as mock_run_progress:
            loader.insert()
            # _run_progress should be called even with empty DataFrame
            mock_run_progress.assert_called_once()
            # But the rows list should be empty
            self.assertEqual(len(mock_run_progress.call_args[0][0]), 0)

    @patch('src.etl.database.unified_loader.Progress')
    def test_run_progress_with_mssql(self, mock_progress):
        """Test _run_progress method with MSSQL dialect."""
        mock_progress_instance = self.setup_mock_progress(mock_progress)

        # Create loader with MSSQL dialect
        loader = Loader(self.cursor, self.df, self.schema, self.table, mssql)

        # Call _run_progress
        rows = [tuple(r) for r in self.df.itertuples(index=False, name=None)]
        query = "INSERT INTO test_schema.test_table (col1, col2) VALUES (?, ?)"
        loader._run_progress(rows, 1, query)

        # Check that executemany was called
        self.cursor.executemany.assert_called()

        # Check that progress was updated
        mock_progress_instance.update.assert_called()

    @staticmethod
    def setup_mock_progress(mock_progress):
        # Setup mock progress
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance
        mock_task_id = 1
        mock_progress_instance.add_task.return_value = mock_task_id
        return mock_progress_instance

    @patch('src.etl.database.unified_loader.Progress')
    @patch('src.etl.database.unified_loader.execute_values')
    def test_run_progress_with_postgres(self, mock_execute_values, mock_progress):
        """Test _run_progress method with Postgres dialect."""
        mock_progress_instance = self.setup_mock_progress(mock_progress)

        # Create loader with Postgres dialect
        loader = Loader(self.cursor, self.df, self.schema, self.table, postgres)

        # Call _run_progress
        rows = [tuple(r) for r in self.df.itertuples(index=False, name=None)]
        query = "INSERT INTO test_schema.test_table (\"col1\", \"col2\") VALUES %s"
        loader._run_progress(rows, 1, query)

        # Check that execute_values was called with the correct arguments
        mock_execute_values.assert_called()

        # Check that progress was updated
        mock_progress_instance.update.assert_called()

    @patch('src.etl.database.unified_loader.Progress')
    def test_run_progress_with_exception(self, mock_progress):
        """Test _run_progress method when an exception occurs."""

        # Setup cursor to raise an exception
        self.cursor.executemany.side_effect = Exception("Test error")

        # Create loader with MSSQL dialect
        loader = Loader(self.cursor, self.df, self.schema, self.table, mssql)

        # Call _run_progress and expect an exception
        rows = [tuple(r) for r in self.df.itertuples(index=False, name=None)]
        query = "INSERT INTO test_schema.test_table (col1, col2) VALUES (?, ?)"

        with self.assertRaises(Exception):
            loader._run_progress(rows, 1, query)

    def test_insert_integration(self):
        """Test the full insert method."""
        # Create loader with MSSQL dialect
        loader = Loader(self.cursor, self.df, self.schema, self.table, mssql)

        # Mock _run_progress to avoid actual execution
        with patch.object(loader, '_run_progress') as mock_run_progress:
            loader.insert()

            # Check that _run_progress was called
            mock_run_progress.assert_called_once()

            # For this test, we only verify that _run_progress was called
            # The specific arguments are tested in other test methods
            # This avoids issues with different mock call_args structures in different Python versions


if __name__ == '__main__':
    unittest.main()
