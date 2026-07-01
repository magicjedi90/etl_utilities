# src/etl/dataframe/spark/diagnostics.py
"""Diagnostic utilities for analyzing DataFrame type conversions."""

from dataclasses import dataclass
from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as spark_functions
from pyspark.sql.types import StringType


@dataclass
class ColumnDiagnostics:
    """Diagnostic information for a single column's type conversion."""

    column_name: str
    original_type: str
    inferred_type: str
    total_values: int
    null_values: int
    successful_conversions: int
    failed_conversions: int
    success_rate: float
    sample_failed_values: list[Any]


def _non_null_non_empty(column_name: str) -> Column:
    """Condition: value is non-null and not an empty/whitespace-only string."""
    column = spark_functions.col(column_name)
    return column.isNotNull() & (
        spark_functions.trim(column.cast(StringType())) != ""
    )


def _count_where(df: DataFrame, condition: Column) -> int:
    """Count rows matching a condition in a single aggregation."""
    return df.select(
        spark_functions.sum(spark_functions.when(condition, 1).otherwise(0))
    ).first()[0] or 0


def count_conversion_failures(
    original_df: DataFrame,
    converted_df: DataFrame,
    column_name: str,
) -> int:
    """Count values that became null after conversion (excluding empty strings).

    Returns the number of values that were non-null/non-empty before but null after.
    """
    original_non_null = _count_where(original_df, _non_null_non_empty(column_name))
    converted_non_null = _count_where(
        converted_df, spark_functions.col(column_name).isNotNull()
    )
    return original_non_null - converted_non_null


def get_conversion_diagnostics(
    original_df: DataFrame,
    cleaned_df: DataFrame,
    sample_failed_values: int = 5,
) -> dict[str, ColumnDiagnostics]:
    """Analyze conversion results and return detailed diagnostics per column.

    Compares the original DataFrame with the cleaned DataFrame to identify
    values that became null during type conversion (conversion failures).

    Args:
        original_df: The original DataFrame before type cleaning.
        cleaned_df: The DataFrame after type cleaning.
        sample_failed_values: Maximum number of failed value examples to collect per column.

    Returns:
        Dictionary mapping column names to ColumnDiagnostics objects containing:
        - Original and inferred types
        - Counts of successful/failed conversions
        - Sample of values that failed conversion
        - Conversion success rate percentage
    """
    diagnostics: dict[str, ColumnDiagnostics] = {}

    for column_name in original_df.columns:
        if column_name not in cleaned_df.columns:
            continue

        original_type = original_df.schema[column_name].dataType.simpleString()
        inferred_type = cleaned_df.schema[column_name].dataType.simpleString()

        original_col = spark_functions.col(column_name)

        # Count total, null, and non-null/non-empty values in a single pass
        original_stats = original_df.select(
            spark_functions.count("*").alias("total"),
            spark_functions.sum(
                spark_functions.when(original_col.isNull(), 1).otherwise(0)
            ).alias("nulls"),
            spark_functions.sum(
                spark_functions.when(_non_null_non_empty(column_name), 1).otherwise(0)
            ).alias("non_null_non_empty"),
        ).first()

        total_values = original_stats["total"] or 0
        original_null_count = original_stats["nulls"] or 0
        non_null_non_empty = original_stats["non_null_non_empty"] or 0

        cleaned_non_null = _count_where(
            cleaned_df, spark_functions.col(column_name).isNotNull()
        )

        # Failed conversions: values that were non-null/non-empty but became null
        failed_conversions = non_null_non_empty - cleaned_non_null
        successful_conversions = non_null_non_empty - failed_conversions

        success_rate = (
            (successful_conversions / non_null_non_empty * 100)
            if non_null_non_empty > 0
            else 100.0
        )

        failed_samples: list[Any] = []
        if failed_conversions > 0 and sample_failed_values > 0:
            failed_samples = get_failed_value_samples(
                original_df, cleaned_df, column_name, sample_failed_values
            )

        diagnostics[column_name] = ColumnDiagnostics(
            column_name=column_name,
            original_type=original_type,
            inferred_type=inferred_type,
            total_values=total_values,
            null_values=original_null_count,
            successful_conversions=successful_conversions,
            failed_conversions=failed_conversions,
            success_rate=round(success_rate, 2),
            sample_failed_values=failed_samples,
        )

    return diagnostics


def get_failed_value_samples(
    original_df: DataFrame,
    cleaned_df: DataFrame,
    column_name: str,
    limit: int = 3,
) -> list[Any]:
    """Extract sample values that failed conversion (became null).

    Args:
        original_df: Original DataFrame with string values.
        cleaned_df: Cleaned DataFrame with converted types.
        column_name: Name of the column to analyze.
        limit: Maximum number of samples to return.

    Returns:
        List of original string values that became null after conversion.
    """
    # Add row index to join original and cleaned DataFrames
    original_indexed = original_df.select(column_name).withColumn(
        "_row_idx", spark_functions.monotonically_increasing_id()
    )
    cleaned_indexed = cleaned_df.select(column_name).withColumn(
        "_row_idx", spark_functions.monotonically_increasing_id()
    )

    # Join to find values that became null
    joined = original_indexed.alias("orig").join(
        cleaned_indexed.alias("clean"), "_row_idx"
    )

    # Find rows where original was non-null/non-empty but cleaned is null
    original_value = spark_functions.col(f"orig.{column_name}")
    failed_rows = joined.filter(
        original_value.isNotNull()
        & (spark_functions.trim(original_value.cast(StringType())) != "")
        & spark_functions.col(f"clean.{column_name}").isNull()
    ).select(original_value.alias("failed_value"))

    # Collect distinct failed values up to limit
    samples = (
        failed_rows.select("failed_value")
        .distinct()
        .limit(limit)
        .collect()
    )

    return [row["failed_value"] for row in samples]
