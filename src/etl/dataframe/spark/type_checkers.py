# src/etl/dataframe/spark/type_checkers.py
"""Type checking predicate functions for Spark columns."""

from pyspark.sql import functions as spark_functions
from pyspark.sql import Column

from .config import ALL_BOOLEAN_VALUES, DATE_FORMATS, NUMERIC_PATTERN


def _is_null_or_empty(column: Column) -> Column:
    """Check if a column value is null or empty/whitespace-only string."""
    return column.isNull() | (spark_functions.trim(column) == '')


def _clean_numeric_string(column: Column) -> Column:
    """Remove $, %, and , from a string column for numeric parsing."""
    without_dollar = spark_functions.regexp_replace(column, r'[\$]', '')
    without_percent = spark_functions.regexp_replace(without_dollar, r'[%]', '')
    without_comma = spark_functions.regexp_replace(without_percent, r'[,]', '')
    return without_comma


def _try_parse_date(column: Column, source_timezone: str = "UTC") -> Column:
    """Try parsing date with multiple formats, returning first successful parse.

    All parsed timestamps are converted to UTC for consistent timezone handling.

    Important: Session Timezone Requirement
        For correct behavior with timezone-naive strings, the Spark session timezone
        should be set to UTC: `spark.conf.set("spark.sql.session.timeZone", "UTC")`

        When Spark's `try_to_timestamp` parses a timezone-naive string like
        "2023-12-31 23:59:59", it interprets it in the session timezone. If the
        session timezone is not UTC (e.g., America/New_York), timestamps at day
        boundaries may appear to drift to the next day when converted to UTC.

        Example of potential drift with non-UTC session timezone:
        - Input: "2023-12-31 23:59:59" (naive)
        - Session TZ: America/New_York (EST = UTC-5)
        - Interpretation: 23:59:59 EST → stored as 2024-01-01 04:59:59 UTC
        - Result: Date appears to drift from Dec 31 to Jan 1

    Args:
        column: The column to parse
        source_timezone: The timezone to assume for timezone-naive datetime strings.
                        Defaults to "UTC". For timezone-aware strings (with offset),
                        the offset is respected and this parameter is ignored.
                        Note: This parameter applies the conversion AFTER Spark has
                        already interpreted the string in the session timezone.
    """
    from pyspark.sql.types import TimestampType

    parsed_result = spark_functions.lit(None).cast(TimestampType())
    for date_format in reversed(DATE_FORMATS):
        parsed_result = spark_functions.coalesce(
            spark_functions.try_to_timestamp(column, spark_functions.lit(date_format)),
            parsed_result
        )
    # Convert to UTC to ensure consistent timezone handling across all timestamps
    # For timezone-aware inputs, this normalizes to UTC
    # For timezone-naive inputs, assumes they are in source_timezone and converts to UTC
    return spark_functions.to_utc_timestamp(parsed_result, source_timezone)


def is_boolean(column: Column) -> Column:
    """Native Spark SQL check if value can be parsed as boolean."""
    lowercase_value = spark_functions.lower(spark_functions.trim(column))
    return spark_functions.when(
        column.isNull(), spark_functions.lit(False)
    ).when(
        spark_functions.trim(column) == '', spark_functions.lit(True)
    ).otherwise(
        lowercase_value.isin(list(ALL_BOOLEAN_VALUES))
    )


def is_integer(column: Column) -> Column:
    """Native Spark SQL check if value can be parsed as integer."""
    cleaned_value = _clean_numeric_string(spark_functions.trim(column))
    matches_numeric_pattern = cleaned_value.rlike(NUMERIC_PATTERN)
    value_as_double = spark_functions.when(matches_numeric_pattern, cleaned_value.cast('double'))
    is_whole_number = value_as_double.isNotNull() & (value_as_double == spark_functions.floor(value_as_double))
    return spark_functions.when(
        column.isNull(), spark_functions.lit(False)
    ).when(
        spark_functions.trim(column) == '', spark_functions.lit(True)
    ).otherwise(
        matches_numeric_pattern & is_whole_number
    )


def is_float(column: Column) -> Column:
    """Native Spark SQL check if value can be parsed as float."""
    cleaned_value = _clean_numeric_string(spark_functions.trim(column))
    matches_numeric_pattern = cleaned_value.rlike(NUMERIC_PATTERN)
    return spark_functions.when(
        column.isNull(), spark_functions.lit(False)
    ).when(
        spark_functions.trim(column) == '', spark_functions.lit(True)
    ).otherwise(
        matches_numeric_pattern
    )


def is_date(column: Column) -> Column:
    """Native Spark SQL check if value can be parsed as date."""
    return spark_functions.when(
        column.isNull(), spark_functions.lit(False)
    ).when(
        spark_functions.trim(column) == '', spark_functions.lit(True)
    ).otherwise(
        _try_parse_date(spark_functions.trim(column)).isNotNull()
    )
