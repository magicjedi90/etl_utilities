# src/etl/dataframe/polars/cleaning_plan.py
"""Batch-safe cleaning: decide a cleaning plan once, apply it to every batch.

Per-frame inference (clean_all_types / optimize_dtypes) is correct for one
in-memory table but unstable batch-by-batch, because batch A and batch B can
infer incompatible schemas. Splitting inference from application fixes that:
infer_cleaning_plan() on a representative sample, then apply_cleaning_plan() on
each batch so every batch lands on the same schema.
"""
from typing import Dict, List, Optional

import polars as pl

from .parser import PolarsParser
from .schema_tools import optimize_dtypes


def infer_cleaning_plan(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 1.0,
    prefer_float: bool = False,
) -> Dict[str, str]:
    """
    Decide a deterministic cleaning plan: column -> target dtype kind.

    Mirrors clean_all_types' selection logic (bool > num > date, applied only
    when a parser covers >= `threshold` of the non-null, non-blank values), but
    returns the decision instead of applying it. Apply the same plan to every
    batch of a stream via apply_cleaning_plan() for schema stability — the one
    place clean_all_types/optimize_dtypes fail, because they infer per-frame and
    so give batch A and batch B incompatible schemas.

    Plan values are one of:
        "boolean" | "int64" | "float64" | "datetime" | "utf8" | "keep"
    "keep" leaves the column's existing dtype untouched.

    :param df: Polars DataFrame to infer from (use a representative sample/superset).
    :param columns: columns to plan (if None, plan all columns).
    :param threshold: fraction of meaningful values a parser must cover to win
        (1.0 == clean_all_types' "all-or-nothing"; lower to be more aggressive).
    :param prefer_float: when True, numeric columns are always planned as float64
        (recommended for streaming — see the int64 caveat in apply_cleaning_plan).
    :return: dict mapping column name to target dtype kind.
    """
    if columns is None:
        columns = df.columns

    plan: Dict[str, str] = {}
    for column in columns:
        if df.select(pl.col(column).drop_nulls()).height == 0:
            plan[column] = "keep"
            continue

        original = pl.col(column)
        stripped = original.cast(pl.Utf8, strict=False).str.strip_chars()
        # Blank/whitespace strings count as null for coverage purposes.
        effective_non_null = df.select(
            pl.when(stripped == "").then(None).otherwise(original).is_not_null().sum()
        ).item()

        float_expr = PolarsParser.parse_float_expr(column)
        counts = df.select([
            PolarsParser.parse_boolean_expr(column).is_not_null().sum().alias("bool"),
            float_expr.is_not_null().sum().alias("num"),
            PolarsParser.parse_date_expr_vectorized(column).is_not_null().sum().alias("date"),
        ]).row(0, named=True)

        # Tie-break order bool > num > date (stable sort preserves it).
        candidates = [
            (counts["bool"], "boolean"),
            (counts["num"], "num"),
            (counts["date"], "datetime"),
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_count, top_kind = candidates[0]

        if not (effective_non_null > 0 and top_count > 0 and top_count >= threshold * effective_non_null):
            plan[column] = "keep"
            continue

        if top_kind == "num":
            if prefer_float:
                plan[column] = "float64"
            else:
                # int64 only if every parsed value is whole; else float64.
                is_whole = df.select(
                    (float_expr.round(0) == float_expr).fill_null(True).all()
                ).item()
                plan[column] = "int64" if is_whole else "float64"
        else:
            plan[column] = top_kind

    return plan


def apply_cleaning_plan(df: pl.DataFrame, plan: Dict[str, str], optimize: bool = False) -> pl.DataFrame:
    """
    Apply a plan from infer_cleaning_plan() deterministically.

    Every cast is strict=False, so dirty values become null rather than raising
    — the same per-value resilience as the parser expressions. Because the plan is
    fixed up front, every batch lands on the same schema (unlike clean_all_types).

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
