# src/etl/dataframe/polars/schema_tools.py
"""Dtype/schema stabilizers for Polars DataFrames.

These operate on a frame's *schema* rather than parsing values: memory
down-casting and the helpers that make a frame safe to hand to a strict sink
(Delta/Parquet/warehouse append) — non-finite floats, naive datetimes, and
all-null columns.
"""
from typing import Optional

import polars as pl

# String spellings of NaN / +-Infinity. These parse to valid Float64 special
# values but are unrepresentable in JSON and rejected by many SQL/ODBC readers,
# so sanitize_float_columns nulls them out even before any numeric cast.
_NONFINITE_STRINGS = frozenset({
    "nan", "+nan", "-nan",
    "inf", "+inf", "-inf",
    "infinity", "+infinity", "-infinity",
})


def optimize_dtypes(df: pl.DataFrame, skip_columns: Optional[set] = None) -> pl.DataFrame:
    """
    Optimize data types for memory efficiency.
    :param df: Polars DataFrame
    :param skip_columns: Set of column names to exclude from optimization
    :return: DataFrame with optimized data types
    """
    if skip_columns is None:
        skip_columns = set()
    int_cols = [col for col in df.columns if df[col].dtype == pl.Int64 and col not in skip_columns]
    if not int_cols:
        return df

    # Compute min/max for all integer columns in a single pass
    agg_exprs = []
    for col in int_cols:
        agg_exprs.extend([
            pl.col(col).min().alias(f"{col}__min"),
            pl.col(col).max().alias(f"{col}__max"),
        ])
    stats = df.select(agg_exprs).row(0, named=True)

    # Build cast expressions for columns that can be optimized
    cast_exprs = []
    for col in int_cols:
        min_val = stats[f"{col}__min"]
        max_val = stats[f"{col}__max"]

        if min_val is None or max_val is None:
            continue

        target_dtype = None
        if min_val >= 0:
            if max_val <= 255:
                target_dtype = pl.UInt8
            elif max_val <= 65535:
                target_dtype = pl.UInt16
            elif max_val <= 4294967295:
                target_dtype = pl.UInt32
        else:
            if min_val >= -128 and max_val <= 127:
                target_dtype = pl.Int8
            elif min_val >= -32768 and max_val <= 32767:
                target_dtype = pl.Int16
            elif min_val >= -2147483648 and max_val <= 2147483647:
                target_dtype = pl.Int32

        if target_dtype is not None:
            cast_exprs.append(pl.col(col).cast(target_dtype))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


def sanitize_float_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Replace NaN and +/-Infinity with null so strict sinks can read the data.

    NaN/+-Inf are unrepresentable in JSON, rejected by many SQL/ODBC readers, and
    corrupt CSV round-trips. Handles two arrival forms:
      * Float32/Float64 columns holding NaN/+-Inf -> null.
      * String columns whose trimmed, lower-cased value is a NaN/Infinity literal
        -> null (these only become float NaN after a later cast). Genuine nulls,
        real numbers, and ordinary text are preserved.
    :param df: Polars DataFrame
    :return: DataFrame with non-finite values replaced by null
    """
    exprs = []
    for name, dtype in df.schema.items():
        if dtype in (pl.Float32, pl.Float64):
            exprs.append(
                pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None).alias(name)
            )
        elif dtype == pl.Utf8:
            normalized = pl.col(name).str.strip_chars().str.to_lowercase()
            exprs.append(
                pl.when(normalized.is_in(list(_NONFINITE_STRINGS)))
                .then(pl.lit(None, dtype=pl.Utf8))
                .otherwise(pl.col(name)).alias(name)
            )
    return df.with_columns(exprs) if exprs else df


def localize_naive_datetimes(df: pl.DataFrame, tz: str = "UTC") -> pl.DataFrame:
    """
    Stamp a timezone onto timezone-naive Datetime columns (default UTC).

    Timezone-naive timestamps are a portability landmine: Arrow/Parquet and most
    warehouses prefer tz-aware, and some engines can't read naive timestamps.
    Only naive Datetime columns are touched; tz-aware columns are left alone.
    :param df: Polars DataFrame
    :param tz: timezone name to apply (default "UTC")
    :return: DataFrame with naive Datetime columns made tz-aware
    """
    naive = [c for c, dt in zip(df.columns, df.dtypes)
             if isinstance(dt, pl.Datetime) and dt.time_zone is None]
    if not naive:
        return df
    return df.with_columns([pl.col(c).dt.replace_time_zone(tz) for c in naive])


def cast_null_columns(df: pl.DataFrame, to: pl.DataType = pl.Utf8) -> pl.DataFrame:
    """
    Cast columns whose dtype is pl.Null (all-null in this frame) to `to` (default Utf8).

    An all-null batch infers Polars' Null dtype, which most sinks reject. Giving
    those columns a concrete dtype keeps schemas stable across batches/files.
    :param df: Polars DataFrame
    :param to: target dtype for all-null columns (default Utf8)
    :return: DataFrame with Null-dtype columns cast to `to`
    """
    null_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Null]
    if not null_cols:
        return df
    return df.with_columns([pl.col(c).cast(to) for c in null_cols])
