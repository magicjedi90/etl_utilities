import math

from .. import constants
import pandas as pd
import numpy as np


class Validator:
    @staticmethod
    def validate_upload(connection, df: pd.DataFrame, schema: str, table: str):
        df_columns, column_info_df = Validator._fetch_column_info(connection, df, schema, table)
        Validator._check_extra_columns(df_columns, column_info_df, schema, table)
        Validator._validate_column_types(df, df_columns, column_info_df)

    @staticmethod
    def _fetch_column_info(connection, df, schema, table):
        get_column_info_query = (
            f'select COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION '
            f'from INFORMATION_SCHEMA.columns '
            f'where table_schema = \'{schema}\' and table_name = \'{table}\'')
        column_info_df = pd.read_sql(get_column_info_query, connection)
        df_columns = df.columns.tolist()
        return df_columns, column_info_df

    @staticmethod
    def _check_extra_columns(df_columns, column_info_df, schema, table):
        db_columns = column_info_df['COLUMN_NAME'].tolist()
        new_columns = np.setdiff1d(df_columns, db_columns)
        if new_columns.size > 0:
            extra_columns_string = ", ".join(new_columns)
            type_mismatch_error_message = f'The table {schema}.{table} is missing the following columns: {extra_columns_string}'
            raise ExtraColumnsException(type_mismatch_error_message)

    @staticmethod
    def _validate_column_types(df, df_columns, column_info_df):
        type_mismatch_columns = []
        truncated_columns = []

        for column in df_columns:
            db_column_info = column_info_df[column_info_df['COLUMN_NAME'] == column].iloc[0]
            db_column_data_type = db_column_info['DATA_TYPE']
            df_column_data_type = df[column].dtype

            if Validator._is_type_mismatch(df_column_data_type, db_column_data_type):
                type_mismatch_columns.append(
                    f'{column} in dataframe is of type {df_column_data_type} while the database expects a type of {db_column_data_type}')
                continue

            if df_column_data_type in constants.NUMPY_INT_TYPES + constants.NUMPY_FLOAT_TYPES:
                truncate_message = Validator._check_numeric_truncation(column, df, db_column_info)
                if truncate_message is not None:
                    truncated_columns.append(truncate_message)
            elif df_column_data_type in constants.NUMPY_DATE_TYPES + constants.NUMPY_STR_TYPES:
                truncate_message = Validator._check_string_or_date_truncation(column, df, db_column_info)
                if truncate_message is not None:
                    truncated_columns.append(truncate_message)
        if type_mismatch_columns or truncated_columns:
            error_message = '\n'.join(type_mismatch_columns) + '\n'.join(truncated_columns)
            raise ColumnDataException(error_message)

    @staticmethod
    def _is_type_mismatch(df_column_data_type, db_column_data_type):

        for numpy_types, mssql_types in constants.TYPE_MAPPINGS.items():
            if df_column_data_type in numpy_types:
                if db_column_data_type not in mssql_types:
                    return True
                return False
        return False

    @staticmethod
    def _check_numeric_truncation(column, df, db_column_info):
        df_numeric_precision = int(math.log10(df[column].max())) + 1
        db_column_numeric_precision = db_column_info['NUMERIC_PRECISION']
        if df_numeric_precision > db_column_numeric_precision:
            return f'{column} needs a minimum of {df_numeric_precision} precision to be inserted'

    @staticmethod
    def _check_string_or_date_truncation(column, df, db_column_info):
        df_max_string_length = df[column].str.len().max()
        db_column_string_length = db_column_info.get('CHARACTER_MAXIMUM_LENGTH')
        if db_column_string_length and df_max_string_length > db_column_string_length:
            return f'{column} needs a minimum of {df_max_string_length} size to be inserted'


class ExtraColumnsException(Exception):
    pass


class ColumnDataException(Exception):
    pass
