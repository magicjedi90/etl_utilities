import re

import pandas as pd
from pandas import DataFrame
from sqlalchemy import PoolProxiedConnection

# Identifiers are interpolated into SQL text (they cannot be parameterized, and
# the DBAPI paramstyle differs per driver), so restrict them to a safe charset.
_SAFE_IDENTIFIER_PATTERN = re.compile(r'^[A-Za-z0-9_$#@ .-]+$')


def assert_safe_identifier(name: str) -> str:
    """Validate a schema/table/column identifier before SQL interpolation.

    Raises ValueError on characters that could break out of the identifier
    (quotes, semicolons, brackets, comment markers, ...).
    """
    if not name or not _SAFE_IDENTIFIER_PATTERN.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


class DatabaseUtils:
    """
    Utility class for database operations, such as fetching table and column data.
    """

    def __init__(self, connection: PoolProxiedConnection):
        self.connection = connection

    def get_table_list(self, schema: str) -> list:
        assert_safe_identifier(schema)
        query = (
            f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE = 'BASE TABLE';"
        )
        return pd.read_sql(query, self.connection)['TABLE_NAME'].tolist()

    def get_column_names_and_types(self, schema: str, table: str) -> DataFrame:
        assert_safe_identifier(schema)
        assert_safe_identifier(table)
        query = (
            f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}';"
        )
        return pd.read_sql(query, self.connection)

    def get_column_info(self, schema: str, table: str) -> DataFrame:
        """Column name/type/length/precision metadata for one table."""
        assert_safe_identifier(schema)
        assert_safe_identifier(table)
        query = (
            f"SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION "
            f"FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}';"
        )
        return pd.read_sql(query, self.connection)

    def get_column_data(self, schema: str, table: str, column: str) -> pd.Series:
        assert_safe_identifier(schema)
        assert_safe_identifier(table)
        assert_safe_identifier(column)
        query = f"SELECT DISTINCT [{column}] FROM {schema}.{table}"
        return pd.read_sql(query, self.connection)[column].dropna()
