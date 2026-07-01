# src/etl/dataframe/polars/type_cleaning.py
"""Value cleaning and one-shot type inference for Polars DataFrames.

The in-memory cleaning family: clean a single column family (numbers/dates/
bools), infer-and-cast every column in one pass (clean_all_types), and the
clean_df orchestrator that drops empty columns first.

clean_all_types delegates its decisions to cleaning_plan.infer_cleaning_plan
and applies them with cleaning_plan.apply_cleaning_plan — it is the one-shot
wrapper over the same logic the batch/streaming API uses, so the two paths
cannot drift.
"""
import logging
import time
from typing import Callable, Dict, List, Optional

import polars as pl

from .cleaning_plan import _infer_plan_with_counts, apply_cleaning_plan
from .parser import PolarsParser
from .schema_tools import optimize_dtypes

logger = logging.getLogger(__name__)


def _clean_columns(
    df: pl.DataFrame,
    columns: Optional[List[str]],
    expr_builder: Callable[[str], pl.Expr],
    kind: str,
) -> pl.DataFrame:
    """Apply one parser expression to each target column, keeping failures as-is."""
    if columns is None:
        columns = df.columns

    for column in columns:
        try:
            df = df.with_columns(expr_builder(column).alias(column))
        except Exception as e:
            logger.debug(f"Column {column} could not be cleaned as {kind}: {e}")

    return df


def clean_numbers(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Clean numeric columns by parsing floats and integers.
    :param df: Polars DataFrame
    :param columns: List of columns to clean (if None, clean all columns)
    :return: DataFrame with cleaned numeric columns
    """
    return _clean_columns(df, columns, PolarsParser.parse_integer_expr, "number")


def clean_dates(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Clean date columns by parsing various date formats.
    :param df: Polars DataFrame
    :param columns: List of columns to clean (if None, clean all columns)
    :return: DataFrame with cleaned date columns
    """
    return _clean_columns(df, columns, PolarsParser.parse_date_expr, "date")


def clean_bools(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Clean boolean columns by parsing various truthy/falsy values.
    :param df: Polars DataFrame
    :param columns: List of columns to clean (if None, clean all columns)
    :return: DataFrame with cleaned boolean columns
    """
    return _clean_columns(df, columns, PolarsParser.parse_boolean_expr, "boolean")


def _filter_overrides(df: pl.DataFrame, type_overrides: Optional[Dict[str, pl.DataType]]) -> Dict[str, pl.DataType]:
    """Keep only overrides for columns present in the frame, warning on the rest."""
    if not type_overrides:
        return {}
    for col in type_overrides:
        if col not in df.columns:
            logger.warning(f"type_overrides column '{col}' not found in DataFrame, ignoring")
    return {col: dtype for col, dtype in type_overrides.items() if col in df.columns}


def _log_plan_decisions(plan: Dict[str, str], counts: Dict, overrides: Dict[str, pl.DataType]) -> None:
    """Log empty-column skips and a per-kind summary of the inferred plan."""
    for column, kind in plan.items():
        if kind == "keep" and counts.get(f"{column}__orig") == 0:
            logger.info(f"{column} is empty, skipping cleaning")

    decisions: Dict[str, List[str]] = {}
    for column, kind in plan.items():
        decisions.setdefault(kind, []).append(column)
    if overrides:
        decisions["override"] = list(overrides)
    for kind, cols in decisions.items():
        names = ", ".join(cols[:10])
        suffix = f" ... +{len(cols) - 10} more" if len(cols) > 10 else ""
        logger.debug(f"  {kind}: {names}{suffix}")


def clean_all_types(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    type_overrides: Optional[Dict[str, pl.DataType]] = None,
) -> pl.DataFrame:
    """
    Perform comprehensive cleaning on all columns by inferring the best dtype
    per column (bool/int/float/datetime, else keep) and casting in one pass.

    Decisions come from cleaning_plan.infer_cleaning_plan with its all-or-nothing
    default: a parser must cover every non-null, non-blank value to win a column.

    :param df: Polars DataFrame
    :param columns: List of columns to clean (if None, clean all columns)
    :param type_overrides: Dict mapping column names to Polars DataTypes.
           Overridden columns skip inference and are cast directly.
    :return: DataFrame with all columns cleaned
    """
    if columns is None:
        columns = df.columns

    overrides = _filter_overrides(df, type_overrides)
    infer_candidates = [col for col in columns if col not in overrides]
    logger.debug(f"clean_all_types: {len(df):,} rows x {len(infer_candidates)} columns")

    plan, counts = _infer_plan_with_counts(df, infer_candidates, threshold=1.0, prefer_float=False)
    _log_plan_decisions(plan, counts, overrides)

    if overrides:
        df = df.with_columns([
            pl.col(column).cast(dtype, strict=False).alias(column)
            for column, dtype in overrides.items()
        ])
    df = apply_cleaning_plan(df, plan)
    return optimize_dtypes(df, skip_columns=set(overrides))


def clean_df(
    df: pl.DataFrame,
    type_overrides: Optional[Dict[str, pl.DataType]] = None,
    drop_all_null_columns: bool = True,
) -> pl.DataFrame:
    """
    Comprehensive DataFrame cleaning - removes empty rows/columns and cleans all types.
    :param df: Polars DataFrame
    :param type_overrides: Dict mapping column names to Polars DataTypes.
           Overridden columns skip inference and are cast directly.
    :param drop_all_null_columns: when True (default), drop columns that are all
           null. Set False to keep them (recommended when schema stability across
           files/batches matters — e.g. before a Delta/Parquet append; pair with
           cast_null_columns to give those columns a concrete dtype).
    :return: Cleaned DataFrame
    """
    if not df.columns:
        return df
    logger.debug(
        f"clean_df: {len(df):,} rows x {len(df.columns)} columns"
    )
    # Remove columns that are all null — single scan for all columns
    if drop_all_null_columns:
        null_scan_start = time.perf_counter()
        has_data = df.select([
            pl.col(col).is_not_null().any() for col in df.columns
        ]).row(0)
        keep_cols = [col for col, keep in zip(df.columns, has_data) if keep]
        dropped_count = len(df.columns) - len(keep_cols)
        null_scan_duration = time.perf_counter() - null_scan_start
        if dropped_count:
            logger.debug(
                f"  dropped {dropped_count} all-null column(s), "
                f"{len(keep_cols)} remaining ({null_scan_duration:.2f}s)"
            )
        df = df.select(keep_cols)

    return clean_all_types(df, type_overrides=type_overrides)
