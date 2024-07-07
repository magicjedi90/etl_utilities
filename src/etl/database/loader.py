import math

import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from rich import print
from ..exception import ExtraColumnsException, ColumnDataException

MSSQL_INT_TYPES = ['bigint', 'int', 'smallint', 'tinyint']
MSSQL_FLOAT_TYPES = ['decimal', 'numeric']
MSSQL_STR_TYPES = ['varchar', 'nvarchar', 'char', 'nchar']
MSSQL_DATE_TYPES = ['date', 'datetime', 'datetime2']
NUMPY_INT_TYPES = [np.int_, np.int64, np.int32, np.int8]
NUMPY_FLOAT_TYPES = [np.float64, np.float32, np.float16, np.float80, np.float96, np.float128, np.float256]
NUMPY_STR_TYPES = [np.str_, np.object_]


def insert_to_mssql_db(column_string, cursor, data_list, location, values):
    value_list = " union ".join(['select {}'.format(value) for value in values])
    execute_query = (
        f"insert into {location} ({column_string}) {value_list}"
    )
    try:
        cursor.execute(execute_query, data_list)
    except Exception as e:
        print(execute_query)
        print(data_list)
        raise e


class Loader:
    @staticmethod
    def insert_to_mssql_table(cursor, df: pd.DataFrame, schema: str, table: str):
        df = df.replace({np.nan: None})
        column_list = df.columns.tolist()
        column_list = [f'[{column}]' for column in column_list]
        column_string = ", ".join(column_list)
        location = f"{schema}.[{table}]"

        row_values = []
        for column in df.columns:
            str_column = df[column].apply(str)
            max_size = str_column.str.len().max()
            if max_size > 256:
                row_values.append('cast ( ? as nvarchar(max))')
            else:
                row_values.append('?')
        row_value_list = ", ".join(row_values)
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(),
                      MofNCompleteColumn()) as progress:
            total = df.shape[0]
            values = []
            data_list = []
            data_count = 0
            row_count = 0
            upload_task = progress.add_task(f'loading {table}', total=total)
            for row in df.itertuples(index=False, name=None):
                row_size = len(row)
                row_count += 1
                data_count += row_size
                values.append(row_value_list)

                data_list.extend(row)
                next_size = data_count + row_size
                if next_size >= 2000:
                    insert_to_mssql_db(column_string, cursor, data_list, location, values)
                    progress.update(upload_task, advance=row_count)
                    values = []
                    data_list = []
                    data_count = 0
                    row_count = 0
            if row_count > 0:
                insert_to_mssql_db(column_string, cursor, data_list, location, values)
                progress.update(upload_task, advance=row_count)

    @staticmethod
    def validate_mssql_upload(connection, df: pd.DataFrame, schema: str, table: str):
        get_column_info_query = (
            f'select COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION '
            f'from INFORMATION_SCHEMA.columns '
            f'where table_schema = \'{schema}\' and table_name = \'{table}\'')
        column_info_df = pd.read_sql(get_column_info_query, connection)
        # make sure df doesn't have any extra columns
        df_columns = df.columns.tolist()
        db_columns = column_info_df['column_name'].tolist()
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
            db_column_info = column_info_df[column_info_df['column_name'] == column]
            db_column_data_type = db_column_info.iloc[0]['data_type']
            df_column_data_type = df[column].dtype
            db_column_numeric_precision = db_column_info.iloc[0]['numerical_precision']
            db_column_string_length = db_column_info.iloc[0]['character_maximum_length']
            type_mismatch_error_message = (f'{column} in dataframe is of type {df_column_data_type} '
                                           f'while the database expects a type of {db_column_data_type}')
            if df_column_data_type in NUMPY_INT_TYPES:
                if db_column_data_type not in MSSQL_INT_TYPES:
                    type_mismatch_columns.append(type_mismatch_error_message)
                    continue
                df_numeric_precision = int(math.log10(df[column].max())) + 1
                if df_numeric_precision > db_column_numeric_precision:
                    truncate_error_message = (f'{column} needs a minimum of {df_numeric_precision} '
                                              f'precision to be inserted')
                    truncated_columns.append(truncate_error_message)
                    continue

            elif df_column_data_type in NUMPY_FLOAT_TYPES:
                if db_column_data_type not in MSSQL_FLOAT_TYPES:
                    type_mismatch_columns.append(type_mismatch_error_message)
                    continue
                df_numeric_precision = int(math.log10(df[column].max())) + 1
                if df_numeric_precision > db_column_numeric_precision:
                    truncate_error_message = (f'{column} needs a minimum of {df_numeric_precision} '
                                              f'precision to be inserted')
                    truncated_columns.append(truncate_error_message)
                    continue

            elif df_column_data_type is np.datetime64:
                if db_column_data_type not in MSSQL_DATE_TYPES:
                    type_mismatch_columns.append(type_mismatch_error_message)
                    continue
            elif df_column_data_type in NUMPY_STR_TYPES:
                if db_column_data_type not in MSSQL_STR_TYPES:
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
