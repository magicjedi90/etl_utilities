# src/etl/dataframe/polars/cleaning_plan.py
"""Batch-safe cleaning: decide a cleaning plan once, apply it to every batch.

Per-frame inference is correct for one in-memory table but unstable
batch-by-batch, because batch A and batch B can infer incompatible schemas.
Splitting inference from application fixes that: infer_cleaning_plan() on a
representative sample, then apply_cleaning_plan() on each batch so every batch
lands on the same schema.

This module owns the type-inference decision logic. clean_all_types() in
type_cleaning.py is the one-shot wrapper — it infers a plan and applies it to
the same frame — so the in-memory and batch/streaming paths cannot drift.
"""
from typing import Dict, List, Optional, Tuple

import polars as pl

from .parser import PolarsParser
from .schema_tools import optimize_dtypes


def _probe_count_exprs(column: str) -> List[pl.Expr]:
    """Aggregation expressions probing one column's bool/num/date parseability.

    Blank/whitespace-only strings count as null for coverage purposes, so a
    column containing empties can still win a numeric/date/bool dtype while
    preserving the empties as nulls.
    """
    original = pl.col(column)
    stripped = original.cast(pl.Utf8, strict=False).str.strip_chars()
    effective = pl.when(stripped == "").then(None).otherwise(original)
    float_expr = PolarsParser.parse_float_expr(column)
    return [
        effective.is_not_null().sum().alias(f"{column}__orig"),
        PolarsParser.parse_boolean_expr(column).is_not_null().sum().alias(f"{column}__bool"),
        float_expr.is_not_null().sum().alias(f"{column}__num"),
        PolarsParser.parse_date_expr_vectorized(column).is_not_null().sum().alias(f"{column}__date"),
        (float_expr.round(0) == float_expr).fill_null(True).all().alias(f"{column}__whole"),
    ]


def _decide_kind(counts: Dict, column: str, threshold: float, prefer_float: bool) -> str:
    """Pick the target dtype kind for one column from its probe counts."""
    effective_non_null = counts[f"{column}__orig"]
    # Tie-break order bool > num > date (stable sort preserves it).
    candidates = [
        (counts[f"{column}__bool"], "boolean"),
        (counts[f"{column}__num"], "num"),
        (counts[f"{column}__date"], "datetime"),
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_count, top_kind = candidates[0]

    if not (effective_non_null > 0 and top_count > 0 and top_count >= threshold * effective_non_null):
        return "keep"
    if top_kind == "num":
        # int64 only if every parsed value is whole; else float64.
        if prefer_float or not counts[f"{column}__whole"]:
            return "float64"
        return "int64"
    return top_kind


def _infer_plan_with_counts(
    df: pl.DataFrame,
    columns: List[str],
    threshold: float,
    prefer_float: bool,
) -> Tuple[Dict[str, str], Dict]:
    """Infer a plan in a single scan; also return the raw probe counts (for logging)."""
    if not columns:
        return {}, {}
    count_exprs = [expr for column in columns for expr in _probe_count_exprs(column)]
    counts = df.select(count_exprs).row(0, named=True)
    plan = {column: _decide_kind(counts, column, threshold, prefer_float) for column in columns}
    return plan, counts


def infer_cleaning_plan(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 1.0,
    prefer_float: bool = False,
) -> Dict[str, str]:
    """
    Decide a deterministic cleaning plan: column -> target dtype kind.

    A parser wins a column only when it covers >= `threshold` of the non-null,
    non-blank values (tie-break bool > num > date). Apply the same plan to every
    batch of a stream via apply_cleaning_plan() for schema stability. All probe
    counts are computed in a single scan over the frame.

    Plan values are one of:
        "boolean" | "int64" | "float64" | "datetime" | "utf8" | "keep"
    "keep" leaves the column's existing dtype untouched.

    :param df: Polars DataFrame to infer from (use a representative sample/superset).
    :param columns: columns to plan (if None, plan all columns).
    :param threshold: fraction of meaningful values a parser must cover to win
        (1.0 == all-or-nothing; lower to be more aggressive).
    :param prefer_float: when True, numeric columns are always planned as float64
        (recommended for streaming — see the int64 caveat in apply_cleaning_plan).
    :return: dict mapping column name to target dtype kind.
    """
    if columns is None:
        columns = df.columns
    plan, _ = _infer_plan_with_counts(df, columns, threshold, prefer_float)
    return plan


def apply_cleaning_plan(df: pl.DataFrame, plan: Dict[str, str], optimize: bool = False) -> pl.DataFrame:
    """
    Apply a plan from infer_cleaning_plan() deterministically.

    Every cast is strict=False, so dirty values become null rather than raising
    — the same per-value resilience as the parser expressions. Because the plan is
    fixed up front, every batch lands on the same schema.

    int64 caveat: an int64-planned column casts via float -> Int64. If a *later*
    batch contains a fractional value in that column, the cast truncates it
    silently. For streaming where later batches may diverge, infer with
    prefer_float=True (or infer on a representative superset).

    :param df: Polars DataFrame (typically one batch of a stream).
    :param plan: column -> dtype kind, from infer_cleaning_plan().
    :param optimize: when True, downcast Int64 columns via optimize_dtypes. Leave
        False for batch stability (optimize is per-frame and breaks schema parity).
    :return: cleaned DataFrame.
    """
    exprs = []
    for column, kind in plan.items():
        if column not in df.columns or kind == "keep":
            continue
        if kind == "boolean":
            exprs.append(PolarsParser.parse_boolean_expr(column).cast(pl.Boolean, strict=False).alias(column))
        elif kind == "int64":
            exprs.append(PolarsParser.parse_float_expr(column).cast(pl.Int64, strict=False).alias(column))
        elif kind == "float64":
            exprs.append(PolarsParser.parse_float_expr(column).cast(pl.Float64, strict=False).alias(column))
        elif kind == "datetime":
            exprs.append(PolarsParser.parse_date_expr(column).alias(column))
        elif kind == "utf8":
            exprs.append(pl.col(column).cast(pl.Utf8, strict=False).alias(column))

    if exprs:
        df = df.with_columns(exprs)
    if optimize:
        df = optimize_dtypes(df)
    return df
