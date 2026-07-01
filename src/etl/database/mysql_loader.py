"""MySQL/MariaDB-specific loader. Deprecated: prefer
etl.database.unified_loader.Loader with the mariadb dialect, which handles
placeholders, casting, and batching for every supported backend."""

import warnings

from sqlalchemy.engine.interfaces import DBAPICursor

from .loader import Loader
from .sql_dialects import mariadb
from .unified_loader import prepare_dataframe
import pandas as pd

_DEPRECATION_MESSAGE = (
    "MySqlLoader is deprecated; use etl.database.unified_loader.Loader with the mariadb dialect."
)


class MySqlLoader(Loader):
    def __init__(self, cursor: DBAPICursor, df: pd.DataFrame, schema: str, table: str) -> None:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        super().__init__(cursor, df, schema, table)

    @staticmethod
    def insert_to_table(cursor: DBAPICursor, df: pd.DataFrame, schema: str, table: str) -> None:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        column_string = ", ".join(mariadb.escape(column) for column in df.columns)
        location = f"{schema}.{mariadb.escape(table)}"
        df, placeholders = prepare_dataframe(df, mariadb)
        Loader._insert_to_table(column_string, cursor, df, location, placeholders)

    def to_table(self) -> None:
        return self.insert_to_table(self._cursor, self._df, self._schema, self._table)
