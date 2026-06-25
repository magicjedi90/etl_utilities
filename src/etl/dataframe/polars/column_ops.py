# src/etl/dataframe/polars/column_ops.py
"""Column-deriving operations for Polars DataFrames: coalesce and hashing."""
from typing import List

import polars as pl

from ..common.utils import compute_hash


def coalesce_columns(df: pl.DataFrame, columns_to_coalesce: List[str], target_column: str, drop: bool = False) -> pl.DataFrame:
    """
    Coalesce multiple columns into one, taking the first non-null value.
    :param df: Polars DataFrame
    :param columns_to_coalesce: List of column names to coalesce
    :param target_column: Name for the coalesced column
    :param drop: Whether to drop the original columns
    :return: DataFrame with coalesced column
    """
    # Use coalesce function
    coalesce_expr = pl.coalesce([pl.col(col) for col in columns_to_coalesce])
    df = df.with_columns(coalesce_expr.alias(target_column))

    if drop:
        cols_to_drop = [col for col in columns_to_coalesce if col != target_column]
        df = df.drop(cols_to_drop)

    return df


def generate_hash_column(
    df: pl.DataFrame,
    columns_to_hash: List[str],
    new_column_name: str,
    algorithm: str = "sha1",
) -> pl.DataFrame:
    """
    Generate a hash column based on specified columns.
    :param df: Polars DataFrame
    :param columns_to_hash: List of column names to include in hash
    :param new_column_name: Name for the new hash column
    :param algorithm: "sha1" (default) keeps the row-wise cryptographic SHA-1 hex
        string — stable across runs/machines, the right choice when the hash is
        persisted and compared later. "xxhash" uses Polars' vectorized, non-crypto
        UInt64 hash — much faster at scale, but not stable across Polars versions,
        so use it only for in-run dedup/change-detection.
    :return: DataFrame with added hash column
    """
    # Validate inputs
    if not columns_to_hash:
        raise ValueError("columns_to_hash cannot be empty")

    missing_cols = [col for col in columns_to_hash if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    if new_column_name in df.columns:
        raise ValueError(f"Column '{new_column_name}' already exists in DataFrame")

    concat = pl.concat_str([pl.col(col).cast(pl.Utf8) for col in columns_to_hash])
    if algorithm == "sha1":
        hash_expr = concat.map_elements(compute_hash, return_dtype=pl.Utf8)
    elif algorithm == "xxhash":
        hash_expr = concat.hash()  # vectorized UInt64
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Use 'sha1' or 'xxhash'.")
    return df.with_columns(hash_expr.alias(new_column_name))
