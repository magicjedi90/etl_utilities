import math

from sqlalchemy.engine.interfaces import DBAPICursor

from src.etl.database.loader import Loader
from src.etl import constants
import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from src.etl.logger import Logger
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

        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(),
                      MofNCompleteColumn()) as progress:
            total_batches = math.ceil(len(data) / batch_size)
            table_task = progress.add_task(f'loading {location}', total=len(data))
            batch_task = progress.add_task(f'batch', total=total_batches)

            try:
                batch_count = 0
                for i in range(0, len(data), batch_size):
                    row_count = i + batch_size
                    batch_count += 1
                    cursor.executemany(query, data[i:row_count])
                    progress.update(table_task, advance=row_count)
                    progress.update(batch_task, advance=batch_count)
                logger.info(f'Inserted {len(data)} rows into {location}.')
            except Exception as e:
                cursor.rollback()
                logger.error(f'Error inserting data into {location}: {str(e)}')
                raise RuntimeError(f'Error inserting data into {location}: {str(e)}')

    def to_table(self) -> None:
        return self.insert_to_table(self._cursor, self._df, self._schema, self._table)

    def to_table_fast(self, batch_size: int = 1000) -> None:
        return self.insert_to_table_fast(self._cursor, self._df, self._schema, self._table, batch_size)
