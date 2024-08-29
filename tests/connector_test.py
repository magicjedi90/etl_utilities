import importlib
import sys
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, '/home/jedi/Projects/Python/etl_utilities/src/etl/database')

# Import after updating sys.path
from connector import Connector


class TestConnector(unittest.TestCase):

    @patch('connector.urllib.parse.quote_plus')
    @patch('connector.create_engine')
    def test_get_mssql_trusted_connection(self, mock_create_engine, mock_quote_plus):
        mock_engine = Mock()
        mock_engine.connect.return_value.connection = 'connection-object'
        mock_create_engine.return_value = mock_engine
        host, instance, database = 'host', 'instance', 'database'
        Connector.get_mssql_trusted_connection(host, instance, database)
        mock_quote_plus.assert_called()
        mock_create_engine.assert_called()

    @patch('connector.urllib.parse.quote_plus')
    @patch('connector.create_engine')
    def test_get_mssql_user_connection(self, mock_create_engine, mock_quote_plus):
        mock_engine = Mock()
        mock_engine.connect.return_value.connection = 'connection-object'
        mock_create_engine.return_value = mock_engine
        Connector.get_mssql_user_connection('host', 'instance', 'database', 'username', 'password')
        mock_quote_plus.assert_called()
        mock_create_engine.assert_called()

    @patch('connector.create_engine')
    def test_get_postgres_user_connection(self, mock_create_engine):
        mock_engine = Mock()
        mock_engine.connect.return_value.connection = 'connection-object'
        mock_create_engine.return_value = mock_engine
        Connector.get_postgres_user_connection('host', 8080, 'database', 'username', 'password')
        mock_create_engine.assert_called()

    @patch('connector.create_engine')
    def test_get_mysql_user_connection(self, mock_create_engine):
        mock_engine = Mock()
        mock_engine.connect.return_value.connection = 'connection-object'
        mock_create_engine.return_value = mock_engine
        Connector.get_mysql_user_connection('host', 'database', 'username', 'password')
        mock_create_engine.assert_called()


if __name__ == '__main__':
    unittest.main()
