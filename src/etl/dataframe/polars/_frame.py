# src/etl/dataframe/polars/_frame.py
"""Frame-type plumbing shared by the Polars cleaning modules.

Every public cleaning function accepts a DataFrame or a LazyFrame and returns
the same kind (FrameT). Schema questions go through collect_schema(), which
never executes a LazyFrame. Data questions (probe counts, null counts, min/max)
go through collect_row() — the single place a LazyFrame is ever executed.
"""
from typing import Any, Dict, Sequence, TypeVar

import polars as pl

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


def collect_row(frame: pl.DataFrame | pl.LazyFrame, exprs: Sequence[pl.Expr]) -> Dict[str, Any]:
    """Evaluate aggregation expressions and return their single result row.

    A DataFrame is selected eagerly. A LazyFrame runs one query on the streaming
    engine, so inference over a scan_parquet/scan_csv source stays memory-bounded;
    the caller's frame itself stays lazy.
    """
    if isinstance(frame, pl.LazyFrame):
        return frame.select(exprs).collect(engine="streaming").row(0, named=True)
    return frame.select(exprs).row(0, named=True)
