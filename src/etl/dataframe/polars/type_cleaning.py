# src/etl/dataframe/polars/type_cleaning.py
"""Value cleaning and one-shot type inference for Polars frames.

The in-memory cleaning family: clean a single column family (numbers/dates/
bools), infer-and-cast every column in one pass (clean_all_types), and the
clean_df orchestrator that drops empty columns first.

clean_all_types delegates its decisions to cleaning_plan.infer_cleaning_plan
and applies them with cleaning_plan.apply_cleaning_plan — it is the one-shot
wrapper over the same logic the batch/streaming API uses, so the two paths
cannot drift.

Every function accepts a DataFrame or a LazyFrame and returns the same kind.
On a LazyFrame the data-driven decisions (coverage, inferred dtype, empty
columns, Int64 narrowing) are gathered in ONE aggregation query, so the source
is read once to infer and once more when the caller collects or sinks.
"""
import logging
from typing import Any, Callable, Dict, List, Optional

import polars as pl

from ._frame import FrameT, collect_row
from .cleaning_plan import (
    _infer_plan_with_counts,
    apply_cleaning_plan,
    effective_non_null_expr,
    is_already_typed,
)
from .parser import PolarsParser
from .schema_tools import _int_range_exprs, _narrow_int_casts, optimize_dtypes

logger = logging.getLogger(__name__)

# Accepts dtype classes (pl.Int32) as well as instances (pl.Datetime("us", "UTC")).
TypeOverrides = Dict[str, pl.DataType | type[pl.DataType]]


_COLUMNS_DOC = """
    :param columns: Columns to convert. Named columns are converted
        unconditionally — values the parser cannot read become null. When None,
        every column is considered but only those whose non-blank values the
        parser covers *completely* are converted; the rest are left untouched,
        and columns that are already numeric/boolean/temporal are skipped.
        On a LazyFrame, columns=None runs one aggregation query to measure
        coverage; named columns keep the frame fully lazy.
    :raises ValueError: if a named column is not in the DataFrame."""


def _row_label(df: pl.DataFrame | pl.LazyFrame) -> str:
    """Row count for debug logs — unknown for a LazyFrame without executing it."""
    return f"{df.height:,}" if isinstance(df, pl.DataFrame) else "lazy"


def _fully_covered_columns(
    df: pl.DataFrame | pl.LazyFrame,
    expr_builder: Callable[[str], pl.Expr],
) -> List[str]:
    """Columns (not already typed) whose every non-blank value the parser reads."""
    candidates = [c for c, dt in df.collect_schema().items() if not is_already_typed(dt)]
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
    counts = collect_row(df, probe)
    return [
        c for c in candidates
        if counts[f"{c}__orig"] > 0 and counts[f"{c}__parsed"] == counts[f"{c}__orig"]
    ]


def _clean_columns(
    df: FrameT,
    columns: Optional[List[str]],
    expr_builder: Callable[[str], pl.Expr],
    kind: str,
    coverage_builder: Optional[Callable[[str], pl.Expr]] = None,
) -> FrameT:
    """Apply one parser expression to the target columns.

    Explicit columns are converted unconditionally (unparseable values -> null).
    With columns=None, only columns the parser covers completely are converted,
    so a stray text column is never silently wiped to null. `coverage_builder`
    (default: expr_builder) is the expression used for that coverage probe.
    """
    names = df.collect_schema().names()
    if columns is None:
        columns = _fully_covered_columns(df, coverage_builder or expr_builder)
        skipped = [c for c in names if c not in columns]
        if skipped:
            logger.debug(f"clean_{kind}: left {len(skipped)} column(s) untouched: {skipped[:10]}")
    else:
        missing = [c for c in columns if c not in names]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

    if not columns:
        return df
    return df.with_columns([expr_builder(column).alias(column) for column in columns])


def clean_numbers(df: FrameT, columns: Optional[List[str]] = None) -> FrameT:
    """
    Clean numeric columns by parsing floats and integers. The result dtype is
    Float64 (see PolarsParser.parse_integer_expr); use clean_all_types or a
    cleaning plan for strict Int64 columns.
    :param df: Polars DataFrame or LazyFrame
    :return: frame (same kind as the input) with cleaned numeric columns
    """
    return _clean_columns(df, columns, PolarsParser.parse_integer_expr, "numbers")


def clean_dates(df: FrameT, columns: Optional[List[str]] = None) -> FrameT:
    """
    Clean date columns by parsing various date formats (naive UTC Datetime).
    With columns=None, coverage is judged by the vectorized formats alone (same
    rule as infer_cleaning_plan), so the lenient dateutil fallback cannot claim
    a column of small integers as dates.
    :param df: Polars DataFrame or LazyFrame
    :return: frame (same kind as the input) with cleaned date columns
    """
    return _clean_columns(
        df, columns, PolarsParser.parse_date_expr, "dates",
        coverage_builder=PolarsParser.parse_date_expr_vectorized,
    )


def clean_bools(df: FrameT, columns: Optional[List[str]] = None) -> FrameT:
    """
    Clean boolean columns by parsing various truthy/falsy values.
    :param df: Polars DataFrame or LazyFrame
    :return: frame (same kind as the input) with cleaned boolean columns
    """
    return _clean_columns(df, columns, PolarsParser.parse_boolean_expr, "bools")


for _fn in (clean_numbers, clean_dates, clean_bools):
    _fn.__doc__ = (_fn.__doc__ or "").rstrip() + _COLUMNS_DOC + "\n    "


def _present_overrides(
    columns: List[str],
    type_overrides: Optional[TypeOverrides],
) -> TypeOverrides:
    """Overrides restricted to columns that exist (no logging)."""
    return {col: dtype for col, dtype in (type_overrides or {}).items() if col in columns}


def _filter_overrides(
    columns: List[str],
    type_overrides: Optional[TypeOverrides],
) -> TypeOverrides:
    """Keep only overrides for columns present in the frame, warning on the rest."""
    for col in type_overrides or {}:
        if col not in columns:
            logger.warning(f"type_overrides column '{col}' not found in DataFrame, ignoring")
    return _present_overrides(columns, type_overrides)


def _log_plan_decisions(plan: Dict[str, str], counts: Dict, overrides: TypeOverrides) -> None:
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


def _lazy_range_exprs(
    df: pl.DataFrame | pl.LazyFrame,
    schema: pl.Schema,
    overrides: TypeOverrides,
) -> List[pl.Expr]:
    """On a LazyFrame: min/max for Int64 columns that are already typed (and not
    overridden), so they are narrowed like optimize_dtypes would — from the
    inference query. Eager frames narrow after casting instead (_apply_with_stats)."""
    if not isinstance(df, pl.LazyFrame):
        return []
    return _int_range_exprs(
        col for col, dtype in schema.items() if dtype == pl.Int64 and col not in overrides
    )


def _apply_with_stats(
    df: FrameT,
    plan: Dict[str, str],
    stats: Dict[str, Any],
    overrides: TypeOverrides,
) -> FrameT:
    """Apply overrides and the plan, then narrow Int64 columns (never overrides).

    Eager: narrow with optimize_dtypes — its min/max pass over the already-cast
    in-memory columns is cheaper than carrying ranges through the parse probe.
    Lazy: narrow from the probe's <col>__min/__max (gathered with with_ranges),
    because a rescan would re-run the whole parse pipeline from the source.
    """
    if overrides:
        df = df.with_columns([
            pl.col(column).cast(dtype, strict=False).alias(column)
            for column, dtype in overrides.items()
        ])
    df = apply_cleaning_plan(df, plan)
    if isinstance(df, pl.DataFrame):
        return optimize_dtypes(df, skip_columns=set(overrides))
    int_columns = [
        col for col, dtype in df.collect_schema().items()
        if dtype == pl.Int64 and col not in overrides
    ]
    narrow = _narrow_int_casts(stats, int_columns)
    return df.with_columns(narrow) if narrow else df


def clean_all_types(
    df: FrameT,
    columns: Optional[List[str]] = None,
    type_overrides: Optional[TypeOverrides] = None,
) -> FrameT:
    """
    Perform comprehensive cleaning on all columns by inferring the best dtype
    per column (bool/int/float/datetime, else keep) and casting in one pass.

    Decisions come from cleaning_plan.infer_cleaning_plan with its all-or-nothing
    default: a parser must cover every non-null, non-blank value to win a column.
    Int64 results are then narrowed to the smallest integer dtype
    (optimize_dtypes); on a LazyFrame the value ranges come from the inference
    query, so no second query runs.

    :param df: Polars DataFrame or LazyFrame. A LazyFrame runs one aggregation
           query to infer; the returned frame is still lazy.
    :param columns: List of columns to clean (if None, clean all columns)
    :param type_overrides: Dict mapping column names to Polars DataTypes.
           Overridden columns skip inference and are cast directly.
    :return: frame (same kind as the input) with all columns cleaned
    """
    schema = df.collect_schema()
    names = schema.names()
    if columns is None:
        columns = names

    overrides = _filter_overrides(names, type_overrides)
    infer_candidates = [col for col in columns if col not in overrides]
    logger.debug(f"clean_all_types: {_row_label(df)} rows x {len(infer_candidates)} columns")

    plan, counts = _infer_plan_with_counts(
        df, infer_candidates, threshold=1.0, prefer_float=False,
        extra_exprs=_lazy_range_exprs(df, schema, overrides),
        with_ranges=isinstance(df, pl.LazyFrame),
    )
    _log_plan_decisions(plan, counts, overrides)
    return _apply_with_stats(df, plan, counts, overrides)


def clean_df(
    df: FrameT,
    type_overrides: Optional[TypeOverrides] = None,
    drop_all_null_columns: bool = True,
) -> FrameT:
    """
    Comprehensive DataFrame cleaning: drop all-null columns (optional) and
    all-null rows, then clean all types. Mirrors the pandas Cleaner.clean_df
    (dropna(axis=1, how="all") then dropna(axis=0, how="all")).

    Null counts and type inference share a single aggregation scan (plus Int64
    ranges on a LazyFrame); on a LazyFrame that is the only query run here, and
    the returned frame is still lazy.
    :param df: Polars DataFrame or LazyFrame
    :param type_overrides: Dict mapping column names to Polars DataTypes.
           Overridden columns skip inference and are cast directly.
    :param drop_all_null_columns: when True (default), drop columns that are all
           null. Set False to keep them (recommended when schema stability across
           files/batches matters — e.g. before a Delta/Parquet append; pair with
           cast_null_columns to give those columns a concrete dtype).
    :return: Cleaned frame, the same kind as the input
    """
    schema = df.collect_schema()
    columns = schema.names()
    if not columns:
        return df
    logger.debug(f"clean_df: {_row_label(df)} rows x {len(columns)} columns")

    # All-null rows add nothing to any probe aggregate (every count skips nulls,
    # int_safe treats null as safe), so inferring on the unfiltered frame gives
    # the same plan — which is what lets the null counts share the probe scan.
    present = _present_overrides(columns, type_overrides)
    extra_exprs = _lazy_range_exprs(df, schema, present)
    if drop_all_null_columns:
        extra_exprs += [pl.col(c).null_count().alias(f"{c}__nulls") for c in columns]
        extra_exprs.append(pl.len().alias("__clean_df_rows"))
    plan, counts = _infer_plan_with_counts(
        df, [c for c in columns if c not in present],
        threshold=1.0, prefer_float=False, extra_exprs=extra_exprs,
        with_ranges=isinstance(df, pl.LazyFrame),
    )

    if drop_all_null_columns:
        keep_cols = [c for c in columns if counts[f"{c}__nulls"] < counts["__clean_df_rows"]]
        dropped_count = len(columns) - len(keep_cols)
        if dropped_count:
            logger.debug(f"  dropped {dropped_count} all-null column(s), {len(keep_cols)} remaining")
        df = df.select(keep_cols)
        if not keep_cols:
            return df
        plan = {c: kind for c, kind in plan.items() if c in keep_cols}
        columns = keep_cols

    rows_before = df.height if isinstance(df, pl.DataFrame) else 0
    df = df.filter(~pl.all_horizontal(pl.all().is_null()))
    if isinstance(df, pl.DataFrame) and df.height != rows_before:
        logger.debug(f"  dropped {rows_before - df.height} all-null row(s)")

    # Validate overrides against the surviving columns (an override for a dropped
    # all-null column is reported as missing, as before).
    overrides = _filter_overrides(columns, type_overrides)
    _log_plan_decisions(plan, counts, overrides)
    return _apply_with_stats(df, plan, counts, overrides)
