from sqlalchemy.engine.interfaces import DBAPICursor
import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from ..logger import Logger

logger = Logger().get_logger()


def insert_to_db(column_string: str, cursor: DBAPICursor, data_list: list, location: str,
                 row_placeholders: list[str]) -> None:
    # inserts each row using a union select
    row_list = " union ".join(['select {}'.format(row) for row in row_placeholders])
    execute_query = (
        f"insert into {location} ({column_string}) {row_list}"
    )
    try:
        cursor.execute(execute_query, data_list)
    except Exception as e:
        logger.error(execute_query)
        logger.error(data_list)
        raise e


class Loader:
    def __init__(self, cursor: DBAPICursor, df: pd.DataFrame, schema: str, table: str):
        self._cursor = cursor
        self._df = df
        self._schema = schema
        self._table = table

    @staticmethod
    def _insert_to_table(column_string: str, cursor: DBAPICursor, df: pd.DataFrame, location: str,
                         placeholders: list[str]):
        placeholder_list = ", ".join(placeholders)
        df = df.replace({np.nan: None})
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(),
                      MofNCompleteColumn()) as progress:
            total = df.shape[0]
            row_placeholder = []
            data_list = []
            data_count = 0
            row_count = 0
            progress_location = location.replace('[', '').replace(']', '').replace('`', '')

            upload_task = progress.add_task(f'loading {progress_location}', total=total)
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
