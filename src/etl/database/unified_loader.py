from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from sqlalchemy.engine.interfaces import DBAPICursor
from psycopg2.extras import execute_values
from .sql_dialects import SqlDialect
from .. import constants

MAX_PARAM_PER_STATEMENT = 100000  # safe default for most engines

# Text columns longer than this get an explicit cast placeholder so drivers
# don't guess an undersized parameter type.
LONG_TEXT_THRESHOLD = 255


def prepare_dataframe(df: pd.DataFrame, dialect: SqlDialect) -> tuple[pd.DataFrame, list[str]]:
    """
    Return a DBAPI-ready copy of the DataFrame plus per-column placeholders.

    Long text columns get an explicit cast placeholder, numpy scalar columns
    are converted to their plain-python equivalents so DBAPIs do not choke,
    and NaNs become None. The caller's DataFrame is never mutated.
    """
    df = df.copy()
    placeholders = []
    for column in df.columns:
        series = df[column]
        max_size = series.apply(str).str.len().max()
        if max_size > LONG_TEXT_THRESHOLD:
            if dialect.name == "mssql":
                placeholders.append(f'cast ( {dialect.placeholder} as nvarchar(max))')
            else:
                placeholders.append(f'cast ( {dialect.placeholder} as varchar(21844))')
        else:
            placeholders.append(dialect.placeholder)
        # switches from numpy class to python class for bool float and int
        series_type = series.dtype
        if series_type in constants.NUMPY_BOOL_TYPES or series_type in constants.NUMPY_INT_TYPES or series_type in constants.NUMPY_FLOAT_TYPES:
            df[column] = series.tolist()
    return df.replace({np.nan: None}), placeholders


class Loader:
    """
    Generic DataFrame-to-SQL loader.

    Two public upload helpers:
        • insert – INSERT INTO … VALUES (…), (…), …
    """

    def __init__(
            self,
            cursor: DBAPICursor,
            df: pd.DataFrame,
            schema: str,
            table: str,
            dialect: SqlDialect,
    ) -> None:
        self._cursor = cursor
        self._df = df
        self._schema = schema
        self._table = table
        self._dialect = dialect

        # derived, reused in both strategies
        self._column_string = ", ".join(self._dialect.escape(c) for c in df.columns)
        self._location = f"{self._schema}.{self._table}"
        self._placeholder = self._dialect.placeholder
        self._max_rows_per_query = math.floor(MAX_PARAM_PER_STATEMENT // len(df.columns))
        if self._dialect.name == "mssql":
            self._cursor.fast_executemany = True

    def _prepare_data(self) -> list[str]:
        """Prepare a DBAPI-ready copy of the DataFrame; returns the placeholders."""
        self._df, placeholders = prepare_dataframe(self._df, self._dialect)
        return placeholders

    # ―――― INSERT (VALUES, VALUES, …) ―――― #
    def insert(self) -> None:
        placeholders = self._prepare_data()
        row_placeholder = f"({', '.join(placeholders)})"
        insert_sql = f"INSERT INTO {self._location} ({self._column_string}) VALUES {row_placeholder}"

        rows: List[tuple] = [tuple(r) for r in self._df.itertuples(index=False, name=None)]

        self._run_progress(
            rows,
            batch_size=self._max_rows_per_query,
            query=insert_sql,
        )

    # ―――― generic executor with progress bar ―――― #
    def _run_progress(self, rows: List[tuple], batch_size: int, query: str) -> None:
        """
        Execute the query in chunks with a progress bar.
        """
        is_postgres = self._dialect.name == "postgres"
        if is_postgres:
            # psycopg2's execute_values expands a single VALUES %s template itself
            query = query.split("VALUES")[0] + "VALUES %s"

        total_rows = len(rows)
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
        ) as progress:
            task_id = progress.add_task(f"loading {self._location}", total=total_rows)

            for i in range(0, total_rows, batch_size):
                chunk = rows[i: i + batch_size]

                if not chunk:  # Skip if the chunk is empty
                    continue

                if is_postgres:
                    execute_values(self._cursor, query, chunk)
                else:
                    self._cursor.executemany(query, chunk)
                progress.update(task_id, advance=len(chunk))
