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
from typing import Callable, Dict, List, Optional

import polars as pl

from .cleaning_plan import (
    _infer_plan_with_counts,
    apply_cleaning_plan,
    effective_non_null_expr,
    is_already_typed,
)
from .parser import PolarsParser
from .schema_tools import optimize_dtypes

logger = logging.getLogger(__name__)


_COLUMNS_DOC = """
    :param columns: Columns to convert. Named columns are converted
        unconditionally — values the parser cannot read become null. When None,
        every column is considered but only those whose non-blank values the
        parser covers *completely* are converted; the rest are left untouched,
        and columns that are already numeric/boolean/temporal are skipped.
    :raises ValueError: if a named column is not in the DataFrame."""


def _fully_covered_columns(
    df: pl.DataFrame,
    expr_builder: Callable[[str], pl.Expr],
) -> List[str]:
    """Columns (not already typed) whose every non-blank value the parser reads."""
    candidates = [c for c, dt in df.schema.items() if not is_already_typed(dt)]
    if not candidates:
        return []
    probe = [
        expr
        for c in candidates
        for expr in (
            effective_non_null_expr(c).sum().alias(f"{c}__orig"),
            expr_builder(c).is_not_null().sum().alias(f"{c}__parsed"),
        )
    ]
    counts = df.select(probe).row(0, named=True)
    return [
        c for c in candidates
        if counts[f"{c}__orig"] > 0 and counts[f"{c}__parsed"] == counts[f"{c}__orig"]
    ]


def _clean_columns(
    df: pl.DataFrame,
    columns: Optional[List[str]],
    expr_builder: Callable[[str], pl.Expr],
    kind: str,
    coverage_builder: Optional[Callable[[str], pl.Expr]] = None,
) -> pl.DataFrame:
    """Apply one parser expression to the target columns.

    Explicit columns are converted unconditionally (unparseable values -> null).
    With columns=None, only columns the parser covers completely are converted,
    so a stray text column is never silently wiped to null. `coverage_builder`
    (default: expr_builder) is the expression used for that coverage probe.
    """
    if columns is None:
        columns = _fully_covered_columns(df, coverage_builder or expr_builder)
        skipped = [c for c in df.columns if c not in columns]
        if skipped:
            logger.debug(f"clean_{kind}: left {len(skipped)} column(s) untouched: {skipped[:10]}")
    else:
        missing = [c for c in columns if c not in df.schema]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

    if not columns:
        return df
    return df.with_columns([expr_builder(column).alias(column) for column in columns])


def clean_numbers(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Clean numeric columns by parsing floats and integers. The result dtype is
    Float64 (see PolarsParser.parse_integer_expr); use clean_all_types or a
    cleaning plan for strict Int64 columns.
    :param df: Polars DataFrame
    :return: DataFrame with cleaned numeric columns
    """
    return _clean_columns(df, columns, PolarsParser.parse_integer_expr, "numbers")


def clean_dates(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Clean date columns by parsing various date formats (naive UTC Datetime).
    With columns=None, coverage is judged by the vectorized formats alone (same
    rule as infer_cleaning_plan), so the lenient dateutil fallback cannot claim
    a column of small integers as dates.
    :param df: Polars DataFrame
    :return: DataFrame with cleaned date columns
    """
    return _clean_columns(
        df, columns, PolarsParser.parse_date_expr, "dates",
        coverage_builder=PolarsParser.parse_date_expr_vectorized,
    )


def clean_bools(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Clean boolean columns by parsing various truthy/falsy values.
    :param df: Polars DataFrame
    :return: DataFrame with cleaned boolean columns
    """
    return _clean_columns(df, columns, PolarsParser.parse_boolean_expr, "bools")


for _fn in (clean_numbers, clean_dates, clean_bools):
    _fn.__doc__ = (_fn.__doc__ or "").rstrip() + _COLUMNS_DOC + "\n    "


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
    Comprehensive DataFrame cleaning: drop all-null columns (optional) and
    all-null rows, then clean all types. Mirrors the pandas Cleaner.clean_df
    (dropna(axis=1, how="all") then dropna(axis=0, how="all")).
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
    if drop_all_null_columns:
        # null_count() reads Arrow validity metadata — no data scan required
        null_counts = df.null_count().row(0, named=True)
        keep_cols = [col for col in df.columns if null_counts[col] < len(df)]
        dropped_count = len(df.columns) - len(keep_cols)
        if dropped_count:
            logger.debug(f"  dropped {dropped_count} all-null column(s), {len(keep_cols)} remaining")
        df = df.select(keep_cols)
        if not df.columns:
            return df

    rows_before = len(df)
    df = df.filter(~pl.all_horizontal(pl.all().is_null()))
    if len(df) != rows_before:
        logger.debug(f"  dropped {rows_before - len(df)} all-null row(s)")

    return clean_all_types(df, type_overrides=type_overrides)
