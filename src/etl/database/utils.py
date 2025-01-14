import pandas as pd
from sqlalchemy import PoolProxiedConnection


class DatabaseUtils:
    """
    Utility class for database operations, such as fetching table and column data.
    """

    def __init__(self, connection: PoolProxiedConnection):
        self.connection = connection

    def get_table_list(self, schema: str) -> list:
        query = (
            f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE = 'BASE TABLE';"
        )
        return pd.read_sql(query, self.connection)['TABLE_NAME'].tolist()

    def get_column_names(self, schema: str, table: str) -> list:
        query = (
            f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}';"
        )
        return pd.read_sql(query, self.connection)['COLUMN_NAME'].tolist()

    def get_column_data(self, schema: str, table: str, column: str) -> pd.Series:
        query = f"SELECT DISTINCT [{column}] FROM {schema}.{table}"
        return pd.read_sql(query, self.connection)[column].dropna()
