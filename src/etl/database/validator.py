from pandas import DataFrame
from sqlalchemy import PoolProxiedConnection
from .extra_column_exception import ExtraColumnsException
from ..dataframe.analyzer import Analyzer
from .. import constants
import pandas as pd
import numpy as np
from ..logger import Logger
logger = Logger().get_logger()

class Validator:
    """
    Validates the upload of a DataFrame to a database table.

    Args:
        connection: The database connection object.
        df: The DataFrame to be uploaded.
        schema: The schema of the destination table.
        table: The name of the destination table.

    Raises:
        ExtraColumnsException: If the DataFrame has extra columns not present in the database table.
        ColumnDataException: If there are type mismatches or truncation issues with the columns in the DataFrame.
    """
    def __init__(self, connection: PoolProxiedConnection, df: pd.DataFrame, schema: str, table: str) -> None:
        self._connection = connection
        self._df = df
        self._schema = schema
        self._table = table

    @staticmethod
    def validate_upload(connection: PoolProxiedConnection, df: pd.DataFrame, schema: str, table: str)  -> None:
        df_metadata, column_info_df = Validator._fetch_column_info(connection, df, schema, table)
        Validator._check_extra_columns(df, column_info_df, schema, table)
        Validator._validate_column_types(df_metadata, column_info_df)

    @staticmethod
    def _fetch_column_info(connection: PoolProxiedConnection, df: pd.DataFrame, schema: str, table: str) -> tuple[
        list[dict], DataFrame]:
        get_column_info_query = (
            f'select COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION '
            f'from INFORMATION_SCHEMA.columns '
            f'where table_schema = \'{schema}\' and table_name = \'{table}\'')
        column_info_df = pd.read_sql(get_column_info_query, connection)
        df_metadata = Analyzer.generate_column_metadata(df, None, None, 2)
        return df_metadata, column_info_df

    @staticmethod
    def _check_extra_columns(df, column_info_df, schema, table):
        db_columns = column_info_df['COLUMN_NAME'].tolist()
        new_columns = np.setdiff1d(df.columns.tolist(), db_columns)
        if new_columns.size > 0:
            extra_columns_df = df[new_columns]
            raise ExtraColumnsException(extra_columns_df)

    @staticmethod
    def _validate_column_types(df_metadata, column_info_df):
        type_mismatch_columns = []
        truncated_columns = []

        for column in df_metadata:
            column_name = column['column_name']
            if column['is_empty']:
                logger.info(f'{column_name} is empty skipping type validation')
                continue
            db_column_info = column_info_df[column_info_df['COLUMN_NAME'] == column_name].iloc[0]
            db_column_data_type = db_column_info['DATA_TYPE']
            df_column_data_type = column['data_type']

            if Validator._is_type_mismatch(df_column_data_type, db_column_data_type):
                type_mismatch_columns.append(
                    f'{column_name} in dataframe is of type {df_column_data_type} while the database expects a type of {db_column_data_type}')
                continue

            if df_column_data_type in constants.DB_INT_TYPES + constants.DB_FLOAT_TYPES:
                truncate_message = Validator._check_numeric_truncation(column, db_column_info)
                if truncate_message is not None:
                    truncated_columns.append(truncate_message)
            elif df_column_data_type in constants.DB_STR_TYPES:
                truncate_message = Validator._check_string_truncation(column, db_column_info)
                if truncate_message is not None:
                    truncated_columns.append(truncate_message)
        if type_mismatch_columns or truncated_columns:
            error_message = '\n'.join(type_mismatch_columns) + '\n'.join(truncated_columns)
            raise ColumnDataException(error_message)

    @staticmethod
    def _is_type_mismatch(df_column_data_type, db_column_data_type):
        for db_type in constants.DB_TYPES:
            if db_column_data_type in db_type and df_column_data_type not in db_type:
                if 'string' in db_type:
                    return False
                if df_column_data_type == 'integer' and 'float' in db_type:
                    return False
                if df_column_data_type == 'boolean' and ('float' in db_type or 'integer' in db_type):
                    return False
                return True
        return False

    @staticmethod
    def _check_numeric_truncation(column, db_column_info):
        df_numeric_precision = column['float_precision']
        db_column_numeric_precision = db_column_info['NUMERIC_PRECISION']
        if df_numeric_precision is None:
            return
        if df_numeric_precision > db_column_numeric_precision:
            return f'{column["column_name"]} needs a minimum of {df_numeric_precision} precision to be inserted\n'


    @staticmethod
    def _check_string_truncation(column, db_column_info):
        df_max_string_length = column['max_str_size']
        db_column_string_length = db_column_info.get('CHARACTER_MAXIMUM_LENGTH')
        if df_max_string_length is None:
            return
        if db_column_string_length == -1:
            return
        if df_max_string_length > db_column_string_length:
            return f'{column["column_name"]} needs a minimum of {df_max_string_length} size to be inserted\n'

    def validate(self):
        return self.validate_upload(self._connection, self._df, self._schema, self._table)



class ColumnDataException(Exception):
    """
    Defines the ColumnDataException class, which is an exception subclass used for raising errors related to column data.

    Classes:
        ColumnDataException(Exception): An exception subclass for column data errors.
    """
    pass
