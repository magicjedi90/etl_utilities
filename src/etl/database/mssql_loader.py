from sqlalchemy.engine.interfaces import DBAPICursor

from .loader import Loader
from .. import constants
import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from ..logger import Logger
logger = Logger().get_logger()


def prepare_data(df: pd.DataFrame, schema: str, table: str) -> tuple[pd.DataFrame, str, str, list[str]]:
    column_list = df.columns.tolist()
    column_list = [f'[{column}]' for column in column_list]
    column_string = ", ".join(column_list)
    location = f"{schema}.[{table}]"
    placeholders = []
    for column in df.columns:
        series = df[column]
        series_type = series.dtype
        str_column = series.apply(str)
        max_size = str_column.str.len().max()
        if max_size > 256:
            placeholders.append('cast ( ? as nvarchar(max))')
        else:
            placeholders.append('?')
        # switches from numpy class to python class for bool float and int
        if series_type in constants.NUMPY_BOOL_TYPES or series_type in constants.NUMPY_INT_TYPES or series_type in constants.NUMPY_FLOAT_TYPES:
            df[column] = series.tolist()
    return df, column_string, location, placeholders


class MsSqlLoader(Loader):
    def __init__(self, cursor: DBAPICursor, df: pd.DataFrame, schema: str, table: str) -> None:
        super().__init__(cursor, df, schema, table)

    @staticmethod
    def insert_to_table(cursor: DBAPICursor, df: pd.DataFrame, schema: str, table: str) -> None:
        df, column_string, location, placeholders = prepare_data(df, schema, table)
        Loader._insert_to_table(column_string, cursor, df, location, placeholders)

    @staticmethod
    def insert_to_table_fast(cursor: DBAPICursor, df: pd.DataFrame, schema: str, table: str, batch_size: int = 1000) -> None:
        df, column_string, location, placeholders = prepare_data(df, schema, table)
        df = df.replace({np.nan: None})
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
