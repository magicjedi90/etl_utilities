# src/etl/dataframe/polars/column_names.py
"""Column-name normalization for Polars DataFrames (snake_case / PascalCase)."""
import logging
from typing import Dict, List

import polars as pl

from ..common.utils import standardize_column_name, to_pascal_case

logger = logging.getLogger(__name__)


def column_names_to_snake_case(df: pl.DataFrame, on_collision: str = "coalesce") -> pl.DataFrame:
    """
    Convert column names to snake_case format.
    :param df: Polars DataFrame
    :param on_collision: behavior when two source columns normalize to the same
        name (e.g. "NUSC Delta" and "NUSCDelta" both map to "nusc_delta"):
          "coalesce" (default): merge colliding columns with pl.coalesce (first
              non-null per row wins), keep the first, drop the rest. Logs a warning.
          "error":    raise ValueError listing the collisions.
          "suffix":   keep all columns, disambiguating with _2, _3, ... suffixes.
    :return: DataFrame with standardized column names
    """
    new_names = [standardize_column_name(name) for name in df.columns]

    groups: Dict[str, List[str]] = {}
    for orig, snake in zip(df.columns, new_names):
        groups.setdefault(snake, []).append(orig)
    collisions = {snake: cols for snake, cols in groups.items() if len(cols) > 1}

    if not collisions:
        return df.rename(dict(zip(df.columns, new_names)))

    if on_collision == "error":
        raise ValueError(f"snake_case name collisions: {collisions}")

    if on_collision == "suffix":
        rename_map: Dict[str, str] = {}
        seen: Dict[str, int] = {}
        for orig, snake in zip(df.columns, new_names):
            n = seen.get(snake, 0)
            seen[snake] = n + 1
            rename_map[orig] = snake if n == 0 else f"{snake}_{n + 1}"
        return df.rename(rename_map)

    # default: coalesce colliding columns into the first occurrence
    for snake, cols in collisions.items():
        logger.warning(f"Columns {cols} all map to '{snake}' — coalescing into '{cols[0]}'")
        df = df.with_columns(pl.coalesce([pl.col(c) for c in cols]).alias(cols[0]))
        df = df.drop(cols[1:])
    # Survivors still carry their original names — normalize them now (after the
    # drops, each snake name has exactly one surviving source column).
    return df.rename({c: standardize_column_name(c) for c in df.columns})


def column_names_to_pascal_case(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert column names to PascalCase format.
    :param df: Polars DataFrame
    :return: DataFrame with PascalCase column names
    """
    new_columns = [to_pascal_case(name) for name in df.columns]
    return df.rename(dict(zip(df.columns, new_columns)))
