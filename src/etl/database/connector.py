import urllib
from sqlalchemy import create_engine


class Connector:
    @staticmethod
    def get_mssql_trusted_connection(host: str, instance: str, database: str):
        connection_string = f"SERVER={host}\\{instance};DATABASE={database};Trusted_Connection=yes;"
        quoted = urllib.parse.quote_plus("DRIVER={ODBC DRIVER 17 for SQL SERVER};" + connection_string)
        engine = create_engine('mssql+pyodbc:////?odbc_connect={}'.format(quoted))
        return engine.connect().connection

    @staticmethod
    def get_mssql_user_connection(host: str, instance: str, database: str, username: str, password: str):
        connection_string = f"SERVER={host}\\{instance};DATABASE={database};UID={username};PWD={password};"
        quoted = urllib.parse.quote_plus("DRIVER={ODBC DRIVER 17 for SQL SERVER};" + connection_string)
        engine = create_engine('mssql+pyodbc:////?odbc_connect={}'.format(quoted))
        return engine.connect().connection
