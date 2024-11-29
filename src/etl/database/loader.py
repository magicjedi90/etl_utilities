from .. import constants
import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from ..logger import Logger
logger = Logger().get_logger()

def insert_to_db(column_string, cursor, data_list, location, row_placeholders):
    # inserts each row using a union select
    row_list = " union ".join(['select {}'.format(row) for row in row_placeholders])
    execute_query = (
        f"insert into {location} ({column_string}) {row_list}"
    )
    try:
        logger.debug(f'Execute Query:\n{execute_query}')
        logger.debug(
            f'Data List:\n{data_list}')
        cursor.execute(execute_query, data_list)
    except Exception as e:
        logger.error(execute_query)
        logger.error(data_list)
        raise e


class Loader:
    def __init__(self, cursor, df: pd.DataFrame, schema: str, table: str):
        self._cursor = cursor
        self._df = df
        self._schema = schema
        self._table = table

    @staticmethod
    def insert_to_mssql_table(cursor, df: pd.DataFrame, schema: str, table: str):
        df, column_string, location, placeholders = Loader.prepare_mssql_data(df, schema, table)
        Loader.insert_to_table(column_string, cursor, df, location, placeholders, table)

    @staticmethod
    def prepare_mssql_data(df, schema, table):
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

    @staticmethod
    def batch_insert_to_mssql_table(cursor, df: pd.DataFrame, schema: str, table: str, batch_size: int = 1000):
        df, column_string, location, placeholders = Loader.prepare_mssql_data(df, schema, table)
        df = df.replace({np.nan: None})
        query = f'INSERT INTO {location} ({column_string}) VALUES ({placeholders});'
        logger.debug(f'Query: {query}')

        # Convert DataFrame to list of tuples
        data = [tuple(row) for row in df.itertuples(index=False, name=None)]

        # Perform the bulk insert
        cursor.fast_executemany = True

        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(),
                      MofNCompleteColumn()) as progress:
            upload_task = progress.add_task(f'loading {table}', total=len(data))

            try:
                for i in range(0, len(data), batch_size):
                    row_count = i + batch_size
                    cursor.executemany(query, data[i:row_count])
                    progress.update(upload_task, advance=row_count)
                logger.info(f'Inserted {len(data)} rows into {location}.')
            except Exception as e:
                cursor.rollback()
                logger.error(f'Error inserting data into {location}: {str(e)}')
                raise RuntimeError(f'Error inserting data into {location}: {str(e)}')

    @staticmethod
    def insert_to_mysql_table(cursor, df: pd.DataFrame, schema: str, table: str):
        column_list = df.columns.tolist()
        column_list = [f'`{column}`' for column in column_list]
        column_string = ", ".join(column_list)
        location = f'{schema}.`{table}`'
        placeholders = []
        for column in df.columns:
            series = df[column]
            series_type = series.dtype
            str_column = series.apply(str)
            max_size = str_column.str.len().max()
            if max_size > 255:
                placeholders.append('cast ( %s as varchar(21844))')
            else:
                placeholders.append('%s')
            # switches from numpy class to python class for bool float and int
            if series_type in constants.NUMPY_BOOL_TYPES or series_type in constants.NUMPY_INT_TYPES or series_type in constants.NUMPY_FLOAT_TYPES:
                df[column] = series.tolist()
        Loader.insert_to_table(column_string, cursor, df, location, placeholders, table)

    @staticmethod
    def insert_to_table(column_string, cursor, df, location, placeholders, table):
        placeholder_list = ", ".join(placeholders)
        df = df.replace({np.nan: None})
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(),
                      MofNCompleteColumn()) as progress:
            total = df.shape[0]
            row_placeholder = []
            data_list = []
            data_count = 0
            row_count = 0
            upload_task = progress.add_task(f'loading {table}', total=total)
            for row in df.itertuples(index=False, name=None):
                row_size = len(row)
                row_count += 1
                data_count += row_size
                row_placeholder.append(placeholder_list)

                data_list.extend(row)
                next_size = data_count + row_size
                if next_size >= 2000:
                    insert_to_db(column_string, cursor, data_list, location, row_placeholder)
                    progress.update(upload_task, advance=row_count)
                    row_placeholder = []
                    data_list = []
                    data_count = 0
                    row_count = 0
            if row_count > 0:
                insert_to_db(column_string, cursor, data_list, location, row_placeholder)
                progress.update(upload_task, advance=row_count)

    def to_mysql_table(self):
        return self.insert_to_mysql_table(self._cursor, self._df, self._schema, self._table)

    def to_mssql_table(self):
        return self.insert_to_mssql_table(self._cursor, self._df, self._schema, self._table)

    def batch_to_mssql_table(self, batch_size: int = 1000):
        return self.batch_insert_to_mssql_table(self._cursor, self._df, self._schema, self._table, batch_size)
