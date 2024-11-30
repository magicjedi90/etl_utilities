import urllib
from sqlalchemy import create_engine, PoolProxiedConnection


class Connector:
    def __init__(self, host: str, port: int, instance: str, database: str, username: str, password: str):
        self._host = host
        self._port = port
        self._instance = instance
        self._database = database
        self._username = username
        self._password = password


    @staticmethod
    def get_mssql_trusted_connection(host: str, instance: str, database: str) ->  PoolProxiedConnection:
        connection_string = f"SERVER={host}\\{instance};DATABASE={database};Trusted_Connection=yes;"
        quoted = urllib.parse.quote_plus("DRIVER={ODBC DRIVER 17 for SQL SERVER};" + connection_string)
        engine = create_engine('mssql+pyodbc:////?odbc_connect={}'.format(quoted))
        return engine.connect().connection

    @staticmethod
    def get_mssql_user_connection(host: str, instance: str, database: str, username: str, password: str) ->  PoolProxiedConnection:
        connection_string = f"SERVER={host}\\{instance};DATABASE={database};UID={username};PWD={password};"
        quoted = urllib.parse.quote_plus("DRIVER={ODBC DRIVER 17 for SQL SERVER};" + connection_string)
        engine = create_engine('mssql+pyodbc:////?odbc_connect={}'.format(quoted))
        return engine.connect().connection

    @staticmethod
    def get_postgres_user_connection(host: str, port: int, database: str, username: str, password: str) ->  PoolProxiedConnection:
        connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}'
        engine = create_engine(connection_string)
        return engine.connect().connection

    @staticmethod
    def get_mysql_user_connection(host: str, database: str, username: str, password: str) ->  PoolProxiedConnection:
        connection_string = f'mysql+pymysql://{username}:{password}@{host}/{database}?charset=utf8mb4'
        engine = create_engine(connection_string)
        return engine.connect().connection

    def to_trusted_msql(self) ->  PoolProxiedConnection:
        return self.get_mssql_trusted_connection(self._host, self._instance, self._database)

    def to_user_msql(self) ->  PoolProxiedConnection:
        return self.get_mssql_user_connection(self._host, self._instance, self._database, self._username, self._password)

    def to_user_postgres(self) ->  PoolProxiedConnection:
        return self.get_postgres_user_connection(self._host, self._port, self._database, self._username, self._password)

    def to_user_mysql(self) ->  PoolProxiedConnection:
        return self.get_mysql_user_connection(self._host, self._database, self._username, self._password)
