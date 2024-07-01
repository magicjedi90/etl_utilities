import pandas as pd
from progress.bar import Bar


def insert_to_mssql_db(column_string, cursor, data_list, location, values):
    value_list = " union ".join(['select {}'.format(value) for value in values])
    execute_query = (
        f"insert into {location} ({column_string}) {value_list}"
    )
    try:
        cursor.execute(execute_query, data_list)
        # progress = f'Progress: {total_count} / {total}'
        # print(progress)
    except Exception as e:
        print(execute_query)
        print(data_list)
        raise e


class Loader:
    @staticmethod
    def insert_to_mssql_table(cursor, df: pd.DataFrame, schema: str, table: str):
        column_list = df.columns.tolist()
        column_list = [f'[{column}]' for column in column_list]
        column_string = ", ".join(column_list)
        location = f"{schema}.[{table}]"
        total_count = 0
        values = []
        data_list = []
        data_count = 0
        row_values = []
        total = df.shape[0]
        for column in df.columns:
            str_column = df[column].apply(str)
            max_size = str_column.str.len().max()
            if max_size > 256:
                row_values.append('cast ( ? as varchar(max))')
            else:
                row_values.append('?')
        row_value_list = ", ".join(row_values)
        progress_bar = Bar('Uploading', max=total)
        for row in df.itertuples(index=False, name=None):
            row_size = len(row)
            total_count += 1
            data_count += row_size
            values.append(row_value_list)

            data_list.extend(row)
            next_size = data_count + row_size
            if next_size >= 2000:
                insert_to_mssql_db(column_string, cursor, data_list, location, values)
                progress_bar.next(data_count)
                values = []
                data_list = []
                data_count = 0
            insert_to_mssql_db(column_string, cursor, data_list, location, values)
            progress_bar.finish()
