import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from rich import print
from ..exception import ExtraColumnsException


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
            error_message = f'The table {schema}.{table} is missing the following columns: {extra_columns_string} '
            raise ExtraColumnsException(error_message)
        # make sure column types match up
        for column in df_columns:
            db_column_info = column_info_df[column_info_df['column_name'] == column]
            db_column_data_type = db_column_info.iloc[0]['DATA_TYPE']
            df_column_data_type = df[column].dtype
            db_column_string_length = db_column_info.iloc[0]['character_maximum_length']
            db_column_numeric_precision = db_column_info.iloc[0]['numerical_precision']
            df_max_string_length = df[column].str.len().max()

        # make sure no truncation happens
