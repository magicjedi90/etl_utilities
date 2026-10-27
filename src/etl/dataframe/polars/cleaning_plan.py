# src/etl/dataframe/polars/cleaning_plan.py
"""Batch-safe cleaning: decide a cleaning plan once, apply it to every batch.

Per-frame inference is correct for one in-memory table but unstable
batch-by-batch, because batch A and batch B can infer incompatible schemas.
Splitting inference from application fixes that: infer_cleaning_plan() on a
representative sample, then apply_cleaning_plan() on each batch so every batch
lands on the same schema.

Both accept a DataFrame or a LazyFrame. On a LazyFrame (e.g. pl.scan_parquet),
inference runs one streaming aggregation query and apply_cleaning_plan stays
fully lazy, so scan -> infer -> apply -> sink_parquet never holds the data in
memory.

This module owns the type-inference decision logic. clean_all_types() in
type_cleaning.py is the one-shot wrapper — it infers a plan and applies it to
the same frame — so the in-memory and batch/streaming paths cannot drift.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import polars as pl

from ._frame import FrameT, collect_row
from .parser import PolarsParser
from .schema_tools import optimize_dtypes


# Largest magnitude a Float64 can hold and still cast to Int64 without
# overflowing (2**63). Anything at or beyond this must stay float64.
_INT64_LIMIT = float(2 ** 63)


def is_already_typed(dtype: pl.DataType) -> bool:
    """True when a column already carries a concrete non-string dtype.

    Numeric, boolean, and temporal columns are left untouched by inference:
    re-parsing them means a String -> Float64 round trip that loses precision on
    integers above 2**53 and gains nothing.
    """
    return dtype.is_numeric() or dtype.is_temporal() or dtype == pl.Boolean


def effective_non_null_expr(column: str) -> pl.Expr:
    """Boolean expression: value is non-null and not a blank/whitespace string.

    Blank strings count as null for coverage purposes, so a column containing
    empties can still win a numeric/date/bool dtype while preserving the
    empties as nulls.
    """
    original = pl.col(column)
    stripped = original.cast(pl.String, strict=False).str.strip_chars()
    return pl.when(stripped == "").then(None).otherwise(original).is_not_null()


def _probe_count_exprs(column: str, with_range: bool = False) -> List[pl.Expr]:
    """Aggregation expressions probing one column's bool/num/date parseability.

    with_range adds the parsed numeric min/max (<col>__min/__max) so an int64
    column can be narrowed (schema_tools._narrow_int_casts) without rescanning.
    That only pays off on a LazyFrame, where a rescan re-runs the parse from the
    source; the extra aggregations re-evaluate the parse (~20% probe cost), so
    eager callers narrow the already-cast columns with optimize_dtypes instead.
    """
    float_expr = PolarsParser.parse_float_expr(column)
    int_safe = (float_expr.round(0) == float_expr) & (float_expr.abs() < _INT64_LIMIT)
    exprs = [
        effective_non_null_expr(column).sum().alias(f"{column}__orig"),
        PolarsParser.parse_boolean_expr(column).is_not_null().sum().alias(f"{column}__bool"),
        float_expr.is_not_null().sum().alias(f"{column}__num"),
        PolarsParser.parse_date_expr_vectorized(column).is_not_null().sum().alias(f"{column}__date"),
        # int64 is only safe when every parsed value is whole AND in Int64 range
        int_safe.fill_null(True).all().alias(f"{column}__int_safe"),
    ]
    if with_range:
        exprs += [float_expr.min().alias(f"{column}__min"), float_expr.max().alias(f"{column}__max")]
    return exprs


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
        # int64 only if every parsed value is whole and fits Int64; else float64.
        if prefer_float or not counts[f"{column}__int_safe"]:
            return "float64"
        return "int64"
    return top_kind


def _infer_plan_with_counts(
    df: pl.DataFrame | pl.LazyFrame,
    columns: List[str],
    threshold: float,
    prefer_float: bool,
    extra_exprs: Sequence[pl.Expr] = (),
    with_ranges: bool = False,
) -> Tuple[Dict[str, str], Dict]:
    """Infer a plan in a single scan; also return the raw probe counts.

    Columns that already carry a numeric/boolean/temporal dtype are planned as
    "keep" without probing (see is_already_typed). `extra_exprs` are evaluated
    in the same scan and their results returned in the counts, so callers that
    need more stats (null counts, Int64 ranges) don't re-read the frame.
    `with_ranges` adds each probed column's numeric min/max (see _probe_count_exprs).
    """
    schema = df.collect_schema()
    plan: Dict[str, str] = {column: "keep" for column in columns if is_already_typed(schema[column])}
    # A Null-dtype column has nothing to parse, but probing it still pushes a
    # full column through every date format — plan it as "keep" from the schema.
    empty = [column for column in columns if schema[column] == pl.Null]
    plan.update({column: "keep" for column in empty})
    probe_columns = [column for column in columns if column not in plan]
    count_exprs = [expr for column in probe_columns for expr in _probe_count_exprs(column, with_ranges)]
    count_exprs.extend(extra_exprs)
    counts: Dict = collect_row(df, count_exprs) if count_exprs else {}
    counts.update({f"{column}__orig": 0 for column in empty})
    plan.update({column: _decide_kind(counts, column, threshold, prefer_float) for column in probe_columns})
    # Preserve caller order
    return {column: plan[column] for column in columns}, counts


def infer_cleaning_plan(
    df: pl.DataFrame | pl.LazyFrame,
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
    "keep" leaves the column's existing dtype untouched. Columns that are
    already numeric/boolean/temporal are always "keep" — they are not re-parsed.
    Numeric columns are "int64" only when every value is whole and inside the
    Int64 range; otherwise "float64".

    :param df: Polars DataFrame or LazyFrame to infer from (use a representative
        sample/superset). A LazyFrame is executed once, as a streaming
        aggregation — pass e.g. lf.head(n) to infer from a sample.
    :param columns: columns to plan (if None, plan all columns).
    :param threshold: fraction of meaningful values a parser must cover to win
        (1.0 == all-or-nothing; lower to be more aggressive).
    :param prefer_float: when True, numeric columns are always planned as float64
        (recommended for streaming — see the int64 caveat in apply_cleaning_plan).
    :return: dict mapping column name to target dtype kind.
    """
    if columns is None:
        columns = df.collect_schema().names()
    plan, _ = _infer_plan_with_counts(df, columns, threshold, prefer_float)
    return plan


def apply_cleaning_plan(df: FrameT, plan: Dict[str, str], optimize: bool = False) -> FrameT:
    """
    Apply a plan from infer_cleaning_plan() deterministically.

    Every cast is strict=False, so dirty values become null rather than raising
    — the same per-value resilience as the parser expressions. Because the plan is
    fixed up front, every batch lands on the same schema.

    int64 caveat: an int64-planned column casts via float -> Int64. If a *later*
    batch contains a fractional value in that column, the cast truncates it
    silently. For streaming where later batches may diverge, infer with
    prefer_float=True (or infer on a representative superset).

    :param df: Polars DataFrame or LazyFrame (typically one batch of a stream).
        A LazyFrame is not executed — the casts are added to its plan.
    :param plan: column -> dtype kind, from infer_cleaning_plan().
    :param optimize: when True, downcast Int64 columns via optimize_dtypes. Leave
        False for batch stability (optimize is per-frame and breaks schema parity).
        On a LazyFrame this runs one min/max aggregation query.
    :return: cleaned frame, the same kind as the input.
    """
    exprs = []
    schema = df.collect_schema()
    for column, kind in plan.items():
        if column not in schema or kind == "keep":
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
            exprs.append(pl.col(column).cast(pl.String, strict=False).alias(column))

    if exprs:
        df = df.with_columns(exprs)
    if optimize:
        df = optimize_dtypes(df)
    return df
