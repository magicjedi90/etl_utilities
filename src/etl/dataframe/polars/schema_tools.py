# src/etl/dataframe/polars/schema_tools.py
"""Dtype/schema stabilizers for Polars frames (DataFrame or LazyFrame).

These operate on a frame's *schema* rather than parsing values: memory
down-casting and the helpers that make a frame safe to hand to a strict sink
(Delta/Parquet/warehouse append) — non-finite floats, naive datetimes, and
all-null columns. All but optimize_dtypes are pure transforms that never
execute a LazyFrame.
"""
from typing import Any, Dict, Iterable, List, Optional

import polars as pl

from ._frame import FrameT, collect_row

# String spellings of NaN / +-Infinity. These parse to valid Float64 special
# values but are unrepresentable in JSON and rejected by many SQL/ODBC readers,
# so sanitize_float_columns nulls them out even before any numeric cast.
_NONFINITE_STRINGS = frozenset({
    "nan", "+nan", "-nan",
    "inf", "+inf", "-inf",
    "infinity", "+infinity", "-infinity",
})


def _int_range_exprs(columns: Iterable[str]) -> List[pl.Expr]:
    """min/max aggregations per column, aliased <col>__min / <col>__max."""
    return [
        expr
        for col in columns
        for expr in (pl.col(col).min().alias(f"{col}__min"), pl.col(col).max().alias(f"{col}__max"))
    ]


def _narrowest_int_dtype(min_val: Any, max_val: Any) -> Optional[type[pl.DataType]]:
    """Smallest integer dtype holding [min_val, max_val], or None to keep Int64."""
    if min_val >= 0:
        if max_val <= 255:
            return pl.UInt8
        if max_val <= 65535:
            return pl.UInt16
        if max_val <= 4294967295:
            return pl.UInt32
        return None
    if min_val >= -128 and max_val <= 127:
        return pl.Int8
    if min_val >= -32768 and max_val <= 32767:
        return pl.Int16
    if min_val >= -2147483648 and max_val <= 2147483647:
        return pl.Int32
    return None


def _narrow_int_casts(stats: Dict[str, Any], columns: Iterable[str]) -> List[pl.Expr]:
    """Down-cast expressions for Int64 columns from already-collected
    <col>__min / <col>__max stats. Columns with no stats (all null) are skipped.

    Lets callers that gathered the range in an earlier scan (clean_all_types
    folds it into the inference probe) narrow without scanning again.
    """
    cast_exprs = []
    for col in columns:
        min_val = stats.get(f"{col}__min")
        max_val = stats.get(f"{col}__max")
        if min_val is None or max_val is None:
            continue
        target_dtype = _narrowest_int_dtype(min_val, max_val)
        if target_dtype is not None:
            cast_exprs.append(pl.col(col).cast(target_dtype))
    return cast_exprs


def optimize_dtypes(df: FrameT, skip_columns: Optional[set] = None) -> FrameT:
    """
    Optimize data types for memory efficiency.

    Needs each Int64 column's min/max, so on a LazyFrame this runs one
    aggregation query; the returned frame is still lazy.
    :param df: Polars DataFrame or LazyFrame
    :param skip_columns: Set of column names to exclude from optimization
    :return: frame (same kind as the input) with optimized data types
    """
    if skip_columns is None:
        skip_columns = set()
    int_cols = [col for col, dtype in df.collect_schema().items() if dtype == pl.Int64 and col not in skip_columns]
    if not int_cols:
        return df

    # Compute min/max for all integer columns in a single pass
    cast_exprs = _narrow_int_casts(collect_row(df, _int_range_exprs(int_cols)), int_cols)
    return df.with_columns(cast_exprs) if cast_exprs else df


def sanitize_float_columns(df: FrameT) -> FrameT:
    """
    Replace NaN and +/-Infinity with null so strict sinks can read the data.

    NaN/+-Inf are unrepresentable in JSON, rejected by many SQL/ODBC readers, and
    corrupt CSV round-trips. Handles two arrival forms:
      * Float32/Float64 columns holding NaN/+-Inf -> null.
      * String columns whose trimmed, lower-cased value is a NaN/Infinity literal
        -> null (these only become float NaN after a later cast). Genuine nulls,
        real numbers, and ordinary text are preserved.
    :param df: Polars DataFrame or LazyFrame
    :return: DataFrame with non-finite values replaced by null
    """
    exprs = []
    for name, dtype in df.collect_schema().items():
        if dtype in (pl.Float32, pl.Float64):
            exprs.append(
                pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
            )
        elif dtype == pl.String:
            normalized = pl.col(name).str.strip_chars().str.to_lowercase()
            exprs.append(
                pl.when(normalized.is_in(list(_NONFINITE_STRINGS)))
                .then(pl.lit(None, dtype=pl.String))
                .otherwise(pl.col(name)).alias(name)
            )
    return df.with_columns(exprs) if exprs else df


def localize_naive_datetimes(df: FrameT, tz: str = "UTC") -> FrameT:
    """
    Stamp a timezone onto timezone-naive Datetime columns (default UTC).

    Timezone-naive timestamps are a portability landmine: Arrow/Parquet and most
    warehouses prefer tz-aware, and some engines can't read naive timestamps.
    Only naive Datetime columns are touched; tz-aware columns are left alone.
    :param df: Polars DataFrame or LazyFrame
    :param tz: timezone name to apply (default "UTC")
    :return: DataFrame with naive Datetime columns made tz-aware
    """
    naive = [c for c, dt in df.collect_schema().items()
             if isinstance(dt, pl.Datetime) and dt.time_zone is None]
    if not naive:
        return df
    return df.with_columns([pl.col(c).dt.replace_time_zone(tz) for c in naive])


def cast_null_columns(df: FrameT, to: pl.DataType | type[pl.DataType] = pl.String) -> FrameT:
    """
    Cast columns whose dtype is pl.Null (all-null in this frame) to `to` (default String).

    An all-null batch infers Polars' Null dtype, which most sinks reject. Giving
    those columns a concrete dtype keeps schemas stable across batches/files.
    :param df: Polars DataFrame or LazyFrame
    :param to: target dtype for all-null columns (default String)
    :return: DataFrame with Null-dtype columns cast to `to`
    """
    null_cols = [c for c, dt in df.collect_schema().items() if dt == pl.Null]
    if not null_cols:
        return df
    return df.with_columns([pl.col(c).cast(to) for c in null_cols])
