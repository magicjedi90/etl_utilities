import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from rich import print


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
                row_values.append('cast ( ? as varchar(max))')
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
