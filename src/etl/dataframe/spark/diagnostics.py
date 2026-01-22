# src/etl/dataframe/spark/diagnostics.py
"""Diagnostic utilities for analyzing DataFrame type conversions."""

from dataclasses import dataclass
from typing import Any

from pyspark.sql import DataFrame
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

        # Count total and null values in original
        original_stats = original_df.select(
            spark_functions.count("*").alias("total"),
            spark_functions.sum(
                spark_functions.when(original_col.isNull(), 1).otherwise(0)
            ).alias("nulls"),
            spark_functions.sum(
                spark_functions.when(
                    original_col.isNotNull()
                    & (spark_functions.trim(original_col.cast(StringType())) != ""),
                    1,
                ).otherwise(0)
            ).alias("non_null_non_empty"),
        ).first()

        total_values = original_stats["total"] or 0
        original_null_count = original_stats["nulls"] or 0
        non_null_non_empty = original_stats["non_null_non_empty"] or 0

        # Count non-null values in cleaned DataFrame
        cleaned_non_null = cleaned_df.select(
            spark_functions.sum(
                spark_functions.when(
                    spark_functions.col(column_name).isNotNull(), 1
                ).otherwise(0)
            )
        ).first()[0] or 0

        # Failed conversions: values that were non-null/non-empty but became null
        failed_conversions = non_null_non_empty - cleaned_non_null
        successful_conversions = non_null_non_empty - failed_conversions

        # Calculate success rate
        success_rate = (
            (successful_conversions / non_null_non_empty * 100)
            if non_null_non_empty > 0
            else 100.0
        )

        # Get sample of failed values
        failed_samples: list[Any] = []
        if failed_conversions > 0 and sample_failed_values > 0:
            failed_samples = _get_failed_value_samples(
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


def _get_failed_value_samples(
    original_df: DataFrame,
    cleaned_df: DataFrame,
    column_name: str,
    limit: int,
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
    original_col = spark_functions.col(column_name)

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
    failed_rows = joined.filter(
        spark_functions.col(f"orig.{column_name}").isNotNull()
        & (
            spark_functions.trim(
                spark_functions.col(f"orig.{column_name}").cast(StringType())
            )
            != ""
        )
        & spark_functions.col(f"clean.{column_name}").isNull()
    ).select(spark_functions.col(f"orig.{column_name}").alias("failed_value"))

    # Collect distinct failed values up to limit
    samples = (
        failed_rows.select("failed_value")
        .distinct()
        .limit(limit)
        .collect()
    )

    return [row["failed_value"] for row in samples]


def get_failed_value_samples(
    original_df: DataFrame,
    converted_df: DataFrame,
    column_name: str,
    limit: int = 3,
) -> list[Any]:
    """Get sample values that failed conversion for a single column.

    Public wrapper around _get_failed_value_samples for use in logging.

    Args:
        original_df: Original DataFrame with string values.
        converted_df: DataFrame after type conversion attempt.
        column_name: Name of the column to analyze.
        limit: Maximum number of samples to return.

    Returns:
        List of original string values that became null after conversion.
    """
    return _get_failed_value_samples(original_df, converted_df, column_name, limit)
