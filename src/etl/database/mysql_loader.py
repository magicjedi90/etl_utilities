from sqlalchemy.engine.interfaces import DBAPICursor
from .loader import Loader
from .. import constants
import pandas as pd
from ..logger import Logger
logger = Logger().get_logger()


class MySqlLoader(Loader):
    def __init__(self, cursor: DBAPICursor, df: pd.DataFrame, schema: str, table: str) -> None:
        super().__init__(cursor, df, schema, table)

    @staticmethod
    def insert_to_table(cursor: DBAPICursor, df: pd.DataFrame, schema: str, table: str) -> None:
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
        Loader._insert_to_table(column_string, cursor, df, location, placeholders)

    def to_table(self) -> None:
        return self.insert_to_table(self._cursor, self._df, self._schema, self._table)
