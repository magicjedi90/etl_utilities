import unittest
import pandas as pd
from unittest.mock import Mock, patch
from src.etl.database.loader import Loader


class TestLoader(unittest.TestCase):

    @patch('src.etl.database.loader.insert_to_db')
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


if __name__ == '__main__':
    unittest.main()
