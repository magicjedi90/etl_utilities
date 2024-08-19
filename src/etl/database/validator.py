import math

from .. import constants
import pandas as pd
import numpy as np


class Validator:
    @staticmethod
    def validate_mssql_upload(connection, df: pd.DataFrame, schema: str, table: str):
        get_column_info_query = (
            f'select COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION '
            f'from INFORMATION_SCHEMA.columns '
            f'where table_schema = \'{schema}\' and table_name = \'{table}\'')
        column_info_df = pd.read_sql(get_column_info_query, connection)
        # make sure df doesn't have any extra columns
        df_columns = df.columns.tolist()
        db_columns = column_info_df['COLUMN_NAME'].tolist()
        new_columns = np.setdiff1d(df_columns, db_columns)
        if len(new_columns) > 0:
            extra_columns_string = ", ".join(new_columns)
            type_mismatch_error_message = \
                f'The table {schema}.{table} is missing the following columns: {extra_columns_string} '
            raise ExtraColumnsException(type_mismatch_error_message)
        # make sure column types match up
        type_mismatch_columns = []
        truncated_columns = []
        for column in df_columns:
            db_column_info = column_info_df[column_info_df['COLUMN_NAME'] == column]
            db_column_data_type = db_column_info.iloc[0]['DATA_TYPE']
            df_column_data_type = df[column].dtype
            db_column_numeric_precision = db_column_info.iloc[0]['NUMERIC_PRECISION']
            db_column_string_length = db_column_info.iloc[0]['CHARACTER_MAXIMUM_LENGTH']
            type_mismatch_error_message = (f'{column} in dataframe is of type {df_column_data_type} '
                                           f'while the database expects a type of {db_column_data_type}')
            if df_column_data_type in constants.NUMPY_INT_TYPES:
                if db_column_data_type not in constants.MSSQL_INT_TYPES:
                    type_mismatch_columns.append(type_mismatch_error_message)
                    continue
                df_numeric_precision = int(math.log10(df[column].max())) + 1
                if df_numeric_precision > db_column_numeric_precision:
                    truncate_error_message = (f'{column} needs a minimum of {df_numeric_precision} '
                                              f'precision to be inserted')
                    truncated_columns.append(truncate_error_message)
                    continue

            elif df_column_data_type in constants.NUMPY_FLOAT_TYPES:
                if db_column_data_type not in constants.MSSQL_FLOAT_TYPES:
                    type_mismatch_columns.append(type_mismatch_error_message)
                    continue
                df_numeric_precision = int(math.log10(df[column].max())) + 1
                if df_numeric_precision > db_column_numeric_precision:
                    truncate_error_message = (f'{column} needs a minimum of {df_numeric_precision} '
                                              f'precision to be inserted')
                    truncated_columns.append(truncate_error_message)
                    continue

            elif df_column_data_type in constants.NUMPY_DATE_TYPES:
                if db_column_data_type not in constants.MSSQL_DATE_TYPES:
                    type_mismatch_columns.append(type_mismatch_error_message)
                    continue
            elif df_column_data_type in constants.NUMPY_STR_TYPES:
                if db_column_data_type not in constants.MSSQL_STR_TYPES:
                    type_mismatch_columns.append(type_mismatch_error_message)
                    continue
                df_max_string_length = df[column].str.len().max()
                if df_max_string_length > db_column_string_length:
                    truncate_error_message = (f'{column} needs a minimum of {df_max_string_length} '
                                              f'size to be inserted')
                    truncated_columns.append(truncate_error_message)
                    continue
        if len(truncated_columns) > 0 or len(type_mismatch_columns) > 0:
            error_message = '\n'.join(type_mismatch_columns) + '\n'.join(truncated_columns)
            raise ColumnDataException(error_message)


class ExtraColumnsException(Exception):
    pass


class ColumnDataException(Exception):
    pass
