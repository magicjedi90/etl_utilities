import unittest
from unittest.mock import Mock

import pandas as pd
from src.etl.database.mysql_loader import MySqlLoader
from sqlalchemy.engine.interfaces import DBAPICursor


class TestMsSqlLoader(unittest.TestCase):

    def test_insert_to_table_success(self):
        mock_cursor = Mock(spec=DBAPICursor)
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

        MySqlLoader.insert_to_table(mock_cursor, df, 'test_schema', 'test_table')

        mock_cursor.execute.assert_called()

    def test_insert_to_table_empty_dataframe(self):
        mock_cursor = Mock(spec=DBAPICursor)
        df = pd.DataFrame(columns=['col1', 'col2'])

        MySqlLoader.insert_to_table(mock_cursor, df, 'test_schema', 'test_table')

        mock_cursor.execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
