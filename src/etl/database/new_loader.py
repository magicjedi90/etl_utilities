from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from sqlalchemy.engine.interfaces import DBAPICursor

from .sql_dialects import SqlDialect
from .. import constants

MAX_PARAM_PER_STATEMENT = 2000            # safe default for most engines


class Loader:
    """
    Generic DataFrame-to-SQL loader.

    Two public upload helpers:
        • insert_bulk    – INSERT INTO … VALUES (…), (…), …
        • insert_union   – INSERT INTO … SELECT … UNION ALL SELECT …
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
        self._max_rows_per_query = MAX_PARAM_PER_STATEMENT // len(df.columns)
        if self._dialect.name == "mssql":
            self._cursor.fast_executemany = True

    def _prepare_data(self) -> tuple[pd.DataFrame, list[str]]:
        """
        Replace NaNs with None and down-cast numpy scalar types to
        their plain-python equivalents so DBAPIs do not choke.
        """
        df = self._df.replace({np.nan: None})
        # keep the method short – any column-specific logic can be added here later
        placeholders = []
        for column in df.columns:
            series = df[column]
            series_type = series.dtype
            str_column = series.apply(str)
            max_size = str_column.str.len().max()
            if max_size > 256:
                if self._dialect.name == "mssql":
                    placeholders.append('cast ( ? as nvarchar(max))')
                elif self._dialect.name == "mariadb":
                    placeholders.append('cast ( %s as varchar(21844))')
            else:
                placeholders.append(self._placeholder)
            # switches from numpy class to python class for bool float and int
            if series_type in constants.NUMPY_BOOL_TYPES or series_type in constants.NUMPY_INT_TYPES or series_type in constants.NUMPY_FLOAT_TYPES:
                df[column] = series.tolist()
        return df, placeholders

    # ―――― INSERT (VALUES, VALUES, …) ―――― #
    def insert(self) -> None:
        df, placeholders = self._prepare_data()
        row_placeholder = f"({', '.join(placeholders)})"
        insert_sql = f"INSERT INTO {self._location} ({self._column_string}) VALUES {{values}}"

        rows: List[tuple] = [tuple(r) for r in df.itertuples(index=False, name=None)]

        self._run_progress(
            rows,
            batch_size=self._max_rows_per_query,
            build_query=lambda chunk: insert_sql.format(
                values=", ".join([row_placeholder] * len(chunk))
            ),
        )

    # ―――― UNION ALL SELECT strategy ―――― #
    def insert_union(self) -> None:
        df, placeholders = self._prepare_data()
        select_placeholder = ", ".join(placeholders)
        select_fragment = f"SELECT {select_placeholder}"

        insert_sql = f"INSERT INTO {self._location} ({self._column_string}) {{selects}}"

        rows: List[tuple] = [tuple(r) for r in df.itertuples(index=False, name=None)]

        # every UNION ALL adds a fixed number of params, keep within MAX_PARAM_PER_STATEMENT

        self._run_progress(
            rows,
            batch_size=self._max_rows_per_query,
            build_query=lambda chunk: insert_sql.format(
                selects=" UNION ALL ".join([select_fragment] * len(chunk))
            ),
        )

    # ―――― generic executor with progress bar ―――― #
    def _run_progress(
        self,
        rows: List[tuple],
        *,
        batch_size: int,
        build_query,
    ) -> None:
        """
        Execute query builder in chunks with a progress bar.
        build_query(chunk_rows) must return a ready SQL string.
        """
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

                if not chunk:  # Skip if chunk is empty
                    continue

                actual_chunk_size = len(chunk)

                try:
                    # build_query is a function, expected to return a single SQL string for the chunk
                    query = build_query(chunk)
                    # Parameters need to be flattened from the chunk of tuples
                    data = [value for row in chunk for value in row]
                    self._cursor.executemany(query, data)


                except Exception:    # pragma: no cover
                    self._cursor.rollback()
                    raise
                progress.update(task_id, advance=actual_chunk_size)
