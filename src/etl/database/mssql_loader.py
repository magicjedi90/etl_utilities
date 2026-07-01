"""MSSQL-specific loader. Deprecated: prefer etl.database.unified_loader.Loader
with the mssql dialect, which handles placeholders, casting, and batching
for every supported backend."""

import warnings

from sqlalchemy.engine.interfaces import DBAPICursor

from .loader import Loader
from .sql_dialects import mssql
from .unified_loader import prepare_dataframe
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from ..logger import Logger
logger = Logger().get_logger()

_DEPRECATION_MESSAGE = (
    "MsSqlLoader is deprecated; use etl.database.unified_loader.Loader with the mssql dialect."
)


def prepare_data(df: pd.DataFrame, schema: str, table: str) -> tuple[pd.DataFrame, str, str, list[str]]:
    column_string = ", ".join(mssql.escape(column) for column in df.columns)
    location = f"{schema}.{mssql.escape(table)}"
    df, placeholders = prepare_dataframe(df, mssql)
    return df, column_string, location, placeholders


class MsSqlLoader(Loader):
    def __init__(self, cursor: DBAPICursor, df: pd.DataFrame, schema: str, table: str) -> None:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        super().__init__(cursor, df, schema, table)

    @staticmethod
    def insert_to_table(cursor: DBAPICursor, df: pd.DataFrame, schema: str, table: str) -> None:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        df, column_string, location, placeholders = prepare_data(df, schema, table)
        Loader._insert_to_table(column_string, cursor, df, location, placeholders)

    @staticmethod
    def insert_to_table_fast(cursor: DBAPICursor, df: pd.DataFrame, schema: str, table: str, batch_size: int = 1000) -> None:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        df, column_string, location, placeholders = prepare_data(df, schema, table)
        placeholder_list = ", ".join(placeholders)
        query = f'INSERT INTO {location} ({column_string}) VALUES ({placeholder_list});'
        logger.debug(f'Query: {query}')

        # Convert DataFrame to list of tuples
        data = [tuple(row) for row in df.itertuples(index=False, name=None)]

        # Perform the bulk insert
        cursor.fast_executemany = True
        progress_location = location.replace('[', '').replace(']', '').replace('`', '')
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(),
                      MofNCompleteColumn()) as progress:
            try:
                table_task = progress.add_task(f'fast loading {progress_location}', total=len(data))
                for i in range(0, len(data), batch_size):
                    actual_batch_size = min(batch_size, len(data) - i)
                    cursor.executemany(query, data[i:i + actual_batch_size])
                    progress.update(table_task, advance=actual_batch_size)
            except Exception as e:
                cursor.rollback()
                logger.error(f'Error inserting data into {location}: {str(e)}')
                raise RuntimeError(f'Error inserting data into {location}: {str(e)}')

    def to_table(self) -> None:
        return self.insert_to_table(self._cursor, self._df, self._schema, self._table)

    def to_table_fast(self, batch_size: int = 1000) -> None:
        return self.insert_to_table_fast(self._cursor, self._df, self._schema, self._table, batch_size)
