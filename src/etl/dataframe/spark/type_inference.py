# src/etl/dataframe/spark/type_inference.py
"""Type inference and retry logic for Spark DataFrame cleaning."""

import logging
from typing import Callable, Optional

from pyspark.sql import functions as spark_functions
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import StringType

from .config import SamplingConfig, TYPE_FALLBACK_HIERARCHY
from .type_checkers import is_boolean, is_integer, is_float, is_date
from .type_parsers import parse_boolean, parse_integer, parse_float, parse_date

logger = logging.getLogger(__name__)


def get_type_checks(source_timezone: str) -> list[dict]:
    """Get the type check definitions with a timezone-aware date parser."""
    def parse_date_with_tz(column: Column) -> Column:
        return parse_date(column, source_timezone)

    return [
        {'name': 'boolean', 'checker': is_boolean, 'parser': parse_boolean},
        {'name': 'integer', 'checker': is_integer, 'parser': parse_integer},
        {'name': 'float', 'checker': is_float, 'parser': parse_float},
        {'name': 'datetime', 'checker': is_date, 'parser': parse_date_with_tz},
    ]


def get_sample(dataframe: DataFrame, config: SamplingConfig, total_rows: int) -> DataFrame | None:
    """Get a representative sample of the DataFrame.

    Returns None if sampling should be skipped (dataset too small or disabled).
    """
    if not config.enabled or total_rows < config.min_rows:
        return None

    # Calculate an effective fraction to cap at max_rows
    effective_fraction = min(config.fraction, config.max_rows / total_rows)

    return dataframe.sample(withReplacement=False, fraction=effective_fraction, seed=config.seed)


def infer_types_from_dataframe(
    dataframe: DataFrame,
    type_checks: list[dict],
) -> dict[str, Optional[dict]]:
    """Infer the best type for each column using single-pass aggregation.

    Returns a dict mapping column name to the chosen type_check dict (or None).
    """
    # Build aggregation expressions to compute all stats in a single pass
    aggregation_expressions = []
    for column_name in dataframe.columns:
        column_ref = spark_functions.col(column_name)
        # Count non-null values (type checkers handle empty strings as "matching any type")
        aggregation_expressions.append(
            spark_functions.sum(
                spark_functions.when(column_ref.isNotNull(), 1).otherwise(0)
            ).alias(f"{column_name}__non_null")
        )
        # Count matches for each type using native SQL functions
        for type_check in type_checks:
            aggregation_expressions.append(
                spark_functions.sum(
                    spark_functions.when(type_check['checker'](column_ref), 1).otherwise(0)
                ).alias(f"{column_name}__{type_check['name']}")
            )

    # Execute a single aggregation to get all statistics
    statistics_row = dataframe.agg(*aggregation_expressions).first()

    # Determine the best type for each column based on collected stats
    column_type_mapping: dict[str, Optional[dict]] = {}
    for column_name in dataframe.columns:
        non_null_count = statistics_row[f"{column_name}__non_null"]

        if non_null_count == 0:
            logger.info(f"Column '{column_name}' is empty, skipping.")
            column_type_mapping[column_name] = None
            continue

        # Find the first type where ALL non-null values match
        chosen_type = None
        for type_check in type_checks:
            match_count = statistics_row[f"{column_name}__{type_check['name']}"]
            if match_count == non_null_count:
                chosen_type = type_check
                break

        column_type_mapping[column_name] = chosen_type

    return column_type_mapping


def detect_conversion_failures(
    original_df: DataFrame,
    converted_df: DataFrame,
    column_name: str,
) -> int:
    """Count values that became null after conversion (excluding empty strings).

    Returns the number of values that were non-null/non-empty before but null after.
    """
    # Count non-null, non-empty in the original
    original_non_null = original_df.select(
        spark_functions.sum(
            spark_functions.when(
                spark_functions.col(column_name).isNotNull() &
                (spark_functions.trim(spark_functions.col(column_name).cast(StringType())) != ''),
                1
            ).otherwise(0)
        )
    ).first()[0] or 0

    # Count non-null in converted
    converted_non_null = converted_df.select(
        spark_functions.sum(
            spark_functions.when(spark_functions.col(column_name).isNotNull(), 1).otherwise(0)
        )
    ).first()[0] or 0

    # Failures are values that were valid but became null
    return original_non_null - converted_non_null


def get_parser_for_type(type_name: str, type_checks: list[dict]) -> Callable[[Column], Column] | None:
    """Get the parser function for a given type name."""
    if type_name == 'string':
        return None  # No parser needed for string
    for type_check in type_checks:
        if type_check['name'] == type_name:
            return type_check['parser']
    return None


def apply_type_with_retry(
    dataframe: DataFrame,
    column_name: str,
    inferred_type: str,
    type_checks: list[dict],
) -> tuple[DataFrame, str]:
    """Apply type conversion with automatic fallback on failure.

    Returns (converted_dataframe, final_type_name).
    """
    current_type = inferred_type
    fallback_types = TYPE_FALLBACK_HIERARCHY.get(current_type, [])

    # Try the inferred type first
    parser = get_parser_for_type(current_type, type_checks)
    if parser is None:
        # String type or unknown - no conversion needed
        return dataframe, 'string'

    converted_df = dataframe.withColumn(
        column_name, parser(spark_functions.col(column_name))
    )

    # Check for conversion failures
    failures = detect_conversion_failures(dataframe, converted_df, column_name)

    if failures == 0:
        logger.info(f"Casting column '{column_name}' to {current_type}.")
        return converted_df, current_type

    # Try fallback types
    for fallback_type in fallback_types:
        logger.warning(
            f"Column '{column_name}': {failures} values failed conversion to {current_type}, "
            f"trying fallback to {fallback_type}."
        )

        if fallback_type == 'string':
            # String fallback - no conversion needed, return original
            logger.warning(f"Column '{column_name}' kept as string (fallback from {current_type}).")
            return dataframe, 'string'

        parser = get_parser_for_type(fallback_type, type_checks)
        if parser is None:
            continue

        converted_df = dataframe.withColumn(
            column_name, parser(spark_functions.col(column_name))
        )

        failures = detect_conversion_failures(dataframe, converted_df, column_name)

        if failures == 0:
            logger.info(f"Casting column '{column_name}' to {fallback_type} (fallback from {inferred_type}).")
            return converted_df, fallback_type

        current_type = fallback_type

    # All fallbacks failed, keep as string
    logger.warning(f"Column '{column_name}' kept as string (all type conversions failed).")
    return dataframe, 'string'
