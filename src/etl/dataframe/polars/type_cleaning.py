# src/etl/dataframe/polars/type_cleaning.py
"""Value cleaning and one-shot type inference for Polars DataFrames.

The in-memory cleaning family: clean a single column family (numbers/dates/
bools), infer-and-cast every column in one pass (clean_all_types), and the
clean_df orchestrator that drops empty columns first. For schema-stable
batch/streaming cleaning use cleaning_plan.infer_cleaning_plan/apply_cleaning_plan.
"""
import time
from typing import Dict, List, Optional

import polars as pl

from .parser import PolarsParser
from .schema_tools import optimize_dtypes
from ...logger import Logger

logger = Logger().get_logger()


def clean_numbers(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Clean numeric columns by parsing floats and integers.
    :param df: Polars DataFrame
    :param columns: List of columns to clean (if None, clean all columns)
    :return: DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = df.columns

    for column in columns:
        try:
            # Use the parse_integer_expr from PolarsParser which handles
            # float cleaning and integer conversion for whole numbers
            df = df.with_columns(
                PolarsParser.parse_integer_expr(column).alias(column)
            )
        except Exception as e:
            logger.debug(f"Column {column} could not be cleaned as number: {e}")

    return df


def clean_dates(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Clean date columns by parsing various date formats.
    :param df: Polars DataFrame
    :param columns: List of columns to clean (if None, clean all columns)
    :return: DataFrame with cleaned date columns
    """
    if columns is None:
        columns = df.columns

    for column in columns:
        try:
            df = df.with_columns(
                PolarsParser.parse_date_expr(column)
                .alias(column)
            )
        except Exception as e:
            logger.debug(f"Column {column} could not be cleaned as date: {e}")

    return df


def clean_bools(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Clean boolean columns by parsing various truthy/falsy values.
    :param df: Polars DataFrame
    :param columns: List of columns to clean (if None, clean all columns)
    :return: DataFrame with cleaned boolean columns
    """
    if columns is None:
        columns = df.columns

    for column in columns:
        try:
            df = df.with_columns(
                PolarsParser.parse_boolean_expr(column)
                .alias(column)
            )
        except Exception as e:
            logger.debug(f"Column {column} could not be cleaned as boolean: {e}")

    return df


def clean_all_types(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    type_overrides: Optional[Dict[str, pl.DataType]] = None,
) -> pl.DataFrame:
    """
    Perform comprehensive cleaning on all columns by trying different parsing functions.
    Strategy:
    - Build safe expressions for bool, number, and date that return None for incompatible values.
    - Coalesce them in priority order, falling back to the original value to avoid data loss.
    - Apply for all target columns in a single pass for efficiency.
    :param df: Polars DataFrame
    :param columns: List of columns to clean (if None, clean all columns)
    :param type_overrides: Dict mapping column names to Polars DataTypes.
           Overridden columns skip inference and are cast directly.
    :return: DataFrame with all columns cleaned
    """
    if columns is None:
        columns = df.columns

    if type_overrides is None:
        type_overrides = {}

    # Filter out overrides for columns not present in the DataFrame
    overrides = {col: dtype for col, dtype in type_overrides.items() if col in df.columns}
    for col in type_overrides:
        if col not in df.columns:
            logger.warning(f"type_overrides column '{col}' not found in DataFrame, ignoring")

    if not columns and not overrides:
        return optimize_dtypes(df)

    row_count = len(df)
    col_count = len(columns)
    logger.debug(f"clean_all_types: {row_count:,} rows x {col_count} columns")

    # --- Phase 1: single scan to find empty columns ---
    phase_start = time.perf_counter()
    non_null_counts = df.select([
        pl.col(col).is_not_null().sum().alias(col) for col in columns
    ]).row(0, named=True)

    # Overridden columns skip inference entirely
    infer_candidates = [col for col in columns if col not in overrides]
    non_empty = [col for col in infer_candidates if non_null_counts[col] > 0]
    empty_cols = [col for col in infer_candidates if non_null_counts[col] == 0]
    for col in empty_cols:
        logger.info(f"{col} is empty, skipping cleaning")
    phase_duration = time.perf_counter() - phase_start
    logger.debug(
        f"  phase 1/4 null scan: {len(non_empty)} non-empty, "
        f"{len(empty_cols)} empty ({phase_duration:.2f}s)"
    )

    if not non_empty and not overrides:
        return optimize_dtypes(df)

    # --- Phase 2: build parse expressions for all non-empty columns ---
    phase_start = time.perf_counter()
    count_exprs = []
    column_info = {}  # col -> (bool_expr, num_expr, date_expr)
    all_counts = {}

    for column in non_empty:
        original = pl.col(column)
        # Treat empty or whitespace-only strings as nulls for the purpose of
        # determining if a parser covers all meaningful values. This allows
        # columns with empty strings to still be cast to their numeric/date/bool
        # dtypes while preserving empties as nulls.
        original_utf8 = original.cast(pl.Utf8, strict=False)
        original_stripped = original_utf8.str.strip_chars()
        effective_original = (
            pl.when(original_stripped == "")
            .then(None)
            .otherwise(original)
        )
        bool_expr = PolarsParser.parse_boolean_expr(column)
        num_expr = PolarsParser.parse_integer_expr(column)
        # Use fast vectorized date parsing for both counting and application.
        # Date only wins when date_cnt == original_count, meaning vectorized
        # already parsed everything — no need for the expensive dateutil fallback.
        date_expr = PolarsParser.parse_date_expr_vectorized(column)

        column_info[column] = (bool_expr, num_expr, date_expr)

        count_exprs.extend([
            effective_original.is_not_null().sum().alias(f"{column}__orig"),
            bool_expr.is_not_null().sum().alias(f"{column}__bool"),
            num_expr.is_not_null().sum().alias(f"{column}__num"),
            date_expr.is_not_null().sum().alias(f"{column}__date"),
        ])

    phase_duration = time.perf_counter() - phase_start
    logger.debug(
        f"  phase 2/4 build expressions: "
        f"{len(count_exprs)} aggregations for {len(non_empty)} columns ({phase_duration:.2f}s)"
    )

    # --- Phase 3: single scan to count parse successes ---
    phase_start = time.perf_counter()
    if count_exprs:
        try:
            all_counts = df.select(count_exprs).row(0, named=True)
        except Exception as e:
            logger.debug(f"Batched count evaluation failed: {e}")
            logger.error(f"  phase 3/4 type probing FAILED: {e}")
            all_counts = {}

    phase_duration = time.perf_counter() - phase_start
    logger.debug(f"  phase 3/4 type probing: ({phase_duration:.2f}s)")

    # --- Phase 4: choose best parser per column and apply in a single pass ---
    phase_start = time.perf_counter()
    cleaning_expressions = []
    type_decisions = {"bool": [], "num": [], "date": [], "string": [], "empty": [], "override": []}

    # Apply overrides first — direct cast, no inference
    for column, target_dtype in overrides.items():
        cleaning_expressions.append(pl.col(column).cast(target_dtype, strict=False).alias(column))
        type_decisions["override"].append(column)

    for column in columns:
        if column in overrides:
            continue
        if column not in column_info:
            # Empty column — keep as-is
            cleaning_expressions.append(pl.col(column))
            type_decisions["empty"].append(column)
            continue

        original_count = all_counts[f"{column}__orig"]
        bool_count = all_counts[f"{column}__bool"]
        numeric_count = all_counts[f"{column}__num"]
        date_count = all_counts[f"{column}__date"]

        bool_expr, num_expr, date_expr = column_info[column]

        # Choose the best parser among bool/num/date based on highest non-null count
        # Tie-breaker priority: bool > num > date
        candidates = [
            (bool_count, 'bool'),
            (numeric_count, 'num'),
            (date_count, 'date'),
        ]
        # Sort by count desc, then by priority order as listed
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_count, top_kind = candidates[0]
        # For now only convert if all elements are cleaned - potentially add in a threshold later
        if top_count == original_count and top_count > 0:
            if top_kind == 'bool':
                chosen = bool_expr
            elif top_kind == 'num':
                chosen = num_expr
            else:
                chosen = date_expr
            type_decisions[top_kind].append(column)
        else:
            chosen = pl.col(column)
            type_decisions["string"].append(column)

        cleaning_expressions.append(chosen.alias(column))

    df = df.with_columns(cleaning_expressions)
    phase_duration = time.perf_counter() - phase_start

    # Summarize decisions
    parts = []
    for kind, label in [("bool", "bool"), ("num", "numeric"), ("date", "date"),
                        ("string", "string"), ("empty", "empty"), ("override", "override")]:
        cols = type_decisions[kind]
        if cols:
            parts.append(f"{len(cols)} {label}")
    logger.debug(
        f"  phase 4/4 apply transforms: {', '.join(parts)} ({phase_duration:.2f}s)"
    )
    for kind, label in [("bool", "boolean"), ("num", "numeric"), ("date", "datetime"),
                        ("string", "string (unchanged)"), ("empty", "empty (skipped)"),
                        ("override", "override (user-specified)")]:
        cols = type_decisions[kind]
        if cols:
            names = ", ".join(cols[:10])
            suffix = f" ... +{len(cols) - 10} more" if len(cols) > 10 else ""
            logger.debug(f"    {label}: {names}{suffix}")

    phase_start = time.perf_counter()
    override_cols = set(overrides.keys())
    df = optimize_dtypes(df, skip_columns=override_cols)
    phase_duration = time.perf_counter() - phase_start
    if phase_duration > 0.01:
        logger.debug(f"  optimize_dtypes: ({phase_duration:.2f}s)")

    return df


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
