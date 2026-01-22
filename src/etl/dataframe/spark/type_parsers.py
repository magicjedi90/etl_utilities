# src/etl/dataframe/spark/type_parsers.py
"""Type parsing/conversion functions for Spark columns."""

from pyspark.sql import functions as spark_functions
from pyspark.sql import Column
from pyspark.sql.types import BooleanType, LongType, DoubleType, TimestampType

from .config import TRUTHY_VALUES, FALSY_VALUES
from .type_checkers import _is_null_or_empty, _clean_numeric_string, _try_parse_date


def parse_boolean(column: Column) -> Column:
    """Native Spark SQL boolean parser."""
    lowercase_value = spark_functions.lower(spark_functions.trim(column))
    return spark_functions.when(
        _is_null_or_empty(column), spark_functions.lit(None).cast(BooleanType())
    ).when(
        lowercase_value.isin(list(TRUTHY_VALUES)), spark_functions.lit(True)
    ).when(
        lowercase_value.isin(list(FALSY_VALUES)), spark_functions.lit(False)
    ).otherwise(
        spark_functions.lit(None).cast(BooleanType())
    )


def parse_integer(column: Column) -> Column:
    """Native Spark SQL integer parser using LongType (64-bit) for large values."""
    cleaned_value = _clean_numeric_string(spark_functions.trim(column))
    # Use try_cast through double first to handle strings like "100.00"
    # try_cast returns null for invalid inputs instead of throwing CAST_INVALID_INPUT
    return spark_functions.when(
        _is_null_or_empty(column), spark_functions.lit(None).cast(LongType())
    ).otherwise(
        cleaned_value.try_cast('double').cast(LongType())
    )


def parse_float(column: Column) -> Column:
    """Native Spark SQL float parser using DoubleType (64-bit) for financial precision."""
    cleaned_value = _clean_numeric_string(spark_functions.trim(column))
    # Use try_cast to return null for invalid inputs instead of throwing CAST_INVALID_INPUT
    return spark_functions.when(
        _is_null_or_empty(column), spark_functions.lit(None).cast(DoubleType())
    ).otherwise(
        cleaned_value.try_cast(DoubleType())
    )


def parse_date(column: Column, source_timezone: str = "UTC") -> Column:
    """Native Spark SQL date/timestamp parser.

    All parsed timestamps are normalized to UTC for consistent serialization.

    Args:
        column: The column to parse
        source_timezone: The timezone to assume for timezone-naive datetime strings.
                        Defaults to "UTC". For timezone-aware strings (with offset),
                        the offset is respected and this parameter is ignored.
    """
    return spark_functions.when(
        _is_null_or_empty(column), spark_functions.lit(None).cast(TimestampType())
    ).otherwise(
        _try_parse_date(spark_functions.trim(column), source_timezone)
    )
