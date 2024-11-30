import unittest
from unittest.mock import Mock

import pandas as pd
from src.etl.database.mssql_loader import MsSqlLoader
from sqlalchemy.engine.interfaces import DBAPICursor


class TestMsSqlLoader(unittest.TestCase):

    def test_insert_to_table_fast_success(self):
        mock_cursor = Mock(spec=DBAPICursor)
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        MsSqlLoader.insert_to_table_fast(mock_cursor, df, 'test_schema', 'test_table', 2)

        mock_cursor.executemany.assert_called()
        self.assertTrue(mock_cursor.fast_executemany)

    def test_insert_to_table_fast_empty_dataframe(self):
        mock_cursor = Mock(spec=DBAPICursor)
        df = pd.DataFrame(columns=['col1', 'col2'])

        MsSqlLoader.insert_to_table_fast(mock_cursor, df, 'test_schema', 'test_table', 2)

        mock_cursor.executemany.assert_not_called()

    def test_insert_to_table_fast_exception(self):
        mock_cursor = Mock(spec=DBAPICursor)
        mock_cursor.rollback = Mock()
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        mock_cursor.executemany.side_effect = Exception("Insert error")

        with self.assertRaises(RuntimeError):
            MsSqlLoader.insert_to_table_fast(mock_cursor, df, 'test_schema', 'test_table')

        mock_cursor.rollback.assert_called()

    def test_insert_to_table_success(self):
        mock_cursor = Mock(spec=DBAPICursor)
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        MsSqlLoader.insert_to_table(mock_cursor, df, 'test_schema', 'test_table')

        mock_cursor.execute.assert_called()

    def test_insert_to_table_empty_dataframe(self):
        mock_cursor = Mock(spec=DBAPICursor)
        df = pd.DataFrame(columns=['col1', 'col2'])

        MsSqlLoader.insert_to_table(mock_cursor, df, 'test_schema', 'test_table')

        mock_cursor.execute.assert_not_called()

if __name__ == '__main__':
    unittest.main()
