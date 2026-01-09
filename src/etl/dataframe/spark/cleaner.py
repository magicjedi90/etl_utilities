# src/etl/spark/cleaner.py

import logging
from typing import Optional

from pyspark.sql import functions as spark_functions
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import BooleanType, IntegerType, FloatType, TimestampType

from ..cleaner import standardize_column_name

logger = logging.getLogger(__name__)

# Boolean truthy/falsy values (lowercase)
TRUTHY_VALUES = ('y', 'yes', 't', 'true', 'on', '1')
FALSY_VALUES = ('n', 'no', 'f', 'false', 'off', '0')
ALL_BOOLEAN_VALUES = TRUTHY_VALUES + FALSY_VALUES

# Common date formats to try (ordered by specificity)
DATE_FORMATS = [
    "yyyy-MM-dd'T'HH:mm:ss.SSSXXX",
    "yyyy-MM-dd'T'HH:mm:ss.SSS",
    "yyyy-MM-dd'T'HH:mm:ssXXX",
    "yyyy-MM-dd'T'HH:mm:ss",
    "yyyy-MM-dd HH:mm:ss.SSS",
    "yyyy-MM-dd HH:mm:ss",
    "yyyy-MM-dd",
    "MM/dd/yyyy HH:mm:ss",
    "MM/dd/yyyy",
    "MM-dd-yyyy",
    "dd/MM/yyyy",
    "dd-MM-yyyy",
    "yyyy/MM/dd",
    "yyyyMMdd",
    "MMM dd, yyyy",
    "dd MMM yyyy",
    "MMMM dd, yyyy",
]


def _clean_numeric_string(column: Column) -> Column:
    """Remove $, %, and , from a string column for numeric parsing."""
    without_dollar = spark_functions.regexp_replace(column, r'[\$]', '')
    without_percent = spark_functions.regexp_replace(without_dollar, r'[%]', '')
    without_comma = spark_functions.regexp_replace(without_percent, r'[,]', '')
    return without_comma


def _is_null_or_empty(column: Column) -> Column:
    """Check if a column value is null or empty/whitespace-only string."""
    return column.isNull() | (spark_functions.trim(column) == '')


# Regex pattern for numeric values (integer or float, with optional sign)
_NUMERIC_PATTERN = r'^-?[0-9]+\.?[0-9]*$'


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


def is_integer(column: Column) -> Column:
    """Native Spark SQL check if value can be parsed as integer."""
    cleaned_value = _clean_numeric_string(spark_functions.trim(column))
    matches_numeric_pattern = cleaned_value.rlike(_NUMERIC_PATTERN)
    value_as_double = spark_functions.when(matches_numeric_pattern, cleaned_value.cast('double'))
    is_whole_number = value_as_double.isNotNull() & (value_as_double == spark_functions.floor(value_as_double))
    return spark_functions.when(
        column.isNull(), spark_functions.lit(False)
    ).when(
        spark_functions.trim(column) == '', spark_functions.lit(True)
    ).otherwise(
        matches_numeric_pattern & is_whole_number
    )


def parse_integer(column: Column) -> Column:
    """Native Spark SQL integer parser."""
    cleaned_value = _clean_numeric_string(spark_functions.trim(column))
    # Cast through double first to handle strings like "100.00"
    return spark_functions.when(
        _is_null_or_empty(column), spark_functions.lit(None).cast(IntegerType())
    ).otherwise(
        cleaned_value.cast('double').cast(IntegerType())
    )


def is_float(column: Column) -> Column:
    """Native Spark SQL check if value can be parsed as float."""
    cleaned_value = _clean_numeric_string(spark_functions.trim(column))
    matches_numeric_pattern = cleaned_value.rlike(_NUMERIC_PATTERN)
    return spark_functions.when(
        column.isNull(), spark_functions.lit(False)
    ).when(
        spark_functions.trim(column) == '', spark_functions.lit(True)
    ).otherwise(
        matches_numeric_pattern
    )


def parse_float(column: Column) -> Column:
    """Native Spark SQL float parser."""
    cleaned_value = _clean_numeric_string(spark_functions.trim(column))
    return spark_functions.when(
        _is_null_or_empty(column), spark_functions.lit(None).cast(FloatType())
    ).otherwise(
        cleaned_value.cast(FloatType())
    )


def _try_parse_date(column: Column, source_timezone: str = "UTC") -> Column:
    """Try parsing date with multiple formats, returning first successful parse.

    All parsed timestamps are converted to UTC for consistent timezone handling.

    Args:
        column: The column to parse
        source_timezone: The timezone to assume for timezone-naive datetime strings.
                        Defaults to "UTC". For timezone-aware strings (with offset),
                        the offset is respected and this parameter is ignored.
    """
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


def is_date(column: Column) -> Column:
    """Native Spark SQL check if value can be parsed as date."""
    return spark_functions.when(
        column.isNull(), spark_functions.lit(False)
    ).when(
        spark_functions.trim(column) == '', spark_functions.lit(True)
    ).otherwise(
        _try_parse_date(spark_functions.trim(column)).isNotNull()
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


class SparkCleaner:
    @staticmethod
    def column_names_to_snake_case(dataframe: DataFrame) -> DataFrame:
        """Converts DataFrame column names to snake_case for Spark."""
        result_dataframe = dataframe
        for column_name in result_dataframe.columns:
            result_dataframe = result_dataframe.withColumnRenamed(
                column_name, standardize_column_name(column_name)
            )
        return result_dataframe

    @staticmethod
    def clean_all_types(dataframe: DataFrame, source_timezone: str = "UTC") -> DataFrame:
        """
        Cleans and casts all columns in a Spark DataFrame to their most appropriate type.

        Only casts a column if ALL non-null values can be successfully parsed.
        Uses native Spark SQL operations for optimal performance (no Python UDFs).
        All datetime columns are normalized to UTC for consistent serialization.

        Args:
            dataframe: The DataFrame to clean
            source_timezone: The timezone to assume for timezone-naive datetime strings.
                           Defaults to "UTC". Timezone-aware strings are handled correctly
                           regardless of this setting.
        """
        # Create a date parser that uses the specified source timezone
        def parse_date_with_tz(column: Column) -> Column:
            return parse_date(column, source_timezone)

        type_checks = [
            {'name': 'boolean', 'checker': is_boolean, 'parser': parse_boolean},
            {'name': 'integer', 'checker': is_integer, 'parser': parse_integer},
            {'name': 'float', 'checker': is_float, 'parser': parse_float},
            {'name': 'datetime', 'checker': is_date, 'parser': parse_date_with_tz},
        ]

        # Cache the DataFrame to avoid recomputation
        cached_dataframe = dataframe.cache()

        # Build aggregation expressions to compute all stats in a single pass
        aggregation_expressions = []
        for column_name in cached_dataframe.columns:
            column_ref = spark_functions.col(column_name)
            # Count non-null values
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

        # Execute single aggregation to get all statistics
        statistics_row = cached_dataframe.agg(*aggregation_expressions).first()

        # Determine best type for each column based on collected stats
        column_type_mapping: dict[str, Optional[dict]] = {}
        for column_name in cached_dataframe.columns:
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

        # Apply transformations using native Spark SQL functions
        cleaned_dataframe = cached_dataframe
        for column_name, chosen_type in column_type_mapping.items():
            if chosen_type is None:
                non_null_count = statistics_row[f"{column_name}__non_null"]
                if non_null_count > 0:
                    logger.debug(f"Column '{column_name}' kept as original type (no full type match).")
            else:
                logger.info(f"Casting column '{column_name}' to {chosen_type['name']}.")
                cleaned_dataframe = cleaned_dataframe.withColumn(
                    column_name, chosen_type['parser'](spark_functions.col(column_name))
                )

        # Unpersist the cached DataFrame
        cached_dataframe.unpersist()

        return cleaned_dataframe

    @staticmethod
    def clean_df(dataframe: DataFrame, source_timezone: str = "UTC") -> DataFrame:
        """
        Drops fully empty rows and columns, then cleans the remaining data.

        All datetime columns are normalized to UTC for consistent serialization.

        Args:
            dataframe: The DataFrame to clean
            source_timezone: The timezone to assume for timezone-naive datetime strings.
                           Defaults to "UTC". Timezone-aware strings are handled correctly
                           regardless of this setting.
        """
        # 1. Drop rows where all values are null
        cleaned_dataframe = dataframe.na.drop(how='all')

        # 2. Identify and drop columns where all values are null
        null_count_expressions = [
            spark_functions.count(
                spark_functions.when(spark_functions.col(column_name).isNull(), column_name)
            ).alias(column_name)
            for column_name in cleaned_dataframe.columns
        ]
        null_counts = cleaned_dataframe.select(null_count_expressions).first()

        total_rows = cleaned_dataframe.count()
        columns_to_drop = [
            column_name for column_name in cleaned_dataframe.columns
            if null_counts[column_name] == total_rows
        ]

        if columns_to_drop:
            logger.info(f"Dropping all-null columns: {columns_to_drop}")
            cleaned_dataframe = cleaned_dataframe.drop(*columns_to_drop)

        # 3. Clean the types of the remaining columns
        return SparkCleaner.clean_all_types(cleaned_dataframe, source_timezone)