# src/etl/spark/cleaner.py

import dataclasses
import logging
from typing import Callable, Optional

from pyspark.sql import functions as spark_functions
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import BooleanType, LongType, FloatType, TimestampType, StringType

from ..cleaner import standardize_column_name

logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True, frozen=True)
class SamplingConfig:
    """Configuration for sampling-based type inference."""
    enabled: bool = True
    fraction: float = 0.1          # 10% sample
    min_rows: int = 1000           # Skip sampling if fewer rows
    max_rows: int = 100_000        # Cap sample size
    seed: int | None = None        # For reproducibility


# Type fallback hierarchy for retry logic when sampled type fails on full data
TYPE_FALLBACK_HIERARCHY: dict[str, list[str]] = {
    'boolean': ['integer', 'float', 'string'],
    'integer': ['float', 'string'],
    'float': ['string'],
    'datetime': ['string'],
    'string': [],
}

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
    """Native Spark SQL integer parser using LongType (64-bit) for large values."""
    cleaned_value = _clean_numeric_string(spark_functions.trim(column))
    # Cast through double first to handle strings like "100.00"
    return spark_functions.when(
        _is_null_or_empty(column), spark_functions.lit(None).cast(LongType())
    ).otherwise(
        cleaned_value.cast('double').cast(LongType())
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
    def _get_type_checks(source_timezone: str) -> list[dict]:
        """Get the type check definitions with timezone-aware date parser."""
        def parse_date_with_tz(column: Column) -> Column:
            return parse_date(column, source_timezone)

        return [
            {'name': 'boolean', 'checker': is_boolean, 'parser': parse_boolean},
            {'name': 'integer', 'checker': is_integer, 'parser': parse_integer},
            {'name': 'float', 'checker': is_float, 'parser': parse_float},
            {'name': 'datetime', 'checker': is_date, 'parser': parse_date_with_tz},
        ]

    @staticmethod
    def _get_sample(dataframe: DataFrame, config: SamplingConfig, total_rows: int) -> DataFrame | None:
        """Get a representative sample of the DataFrame.

        Returns None if sampling should be skipped (dataset too small or disabled).
        """
        if not config.enabled or total_rows < config.min_rows:
            return None

        # Calculate effective fraction to cap at max_rows
        effective_fraction = min(config.fraction, config.max_rows / total_rows)

        return dataframe.sample(withReplacement=False, fraction=effective_fraction, seed=config.seed)

    @staticmethod
    def _infer_types_from_dataframe(
        dataframe: DataFrame,
        type_checks: list[dict],
    ) -> dict[str, Optional[dict]]:
        """Infer best type for each column using single-pass aggregation.

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

        # Execute single aggregation to get all statistics
        statistics_row = dataframe.agg(*aggregation_expressions).first()

        # Determine best type for each column based on collected stats
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

    @staticmethod
    def _detect_conversion_failures(
        original_df: DataFrame,
        converted_df: DataFrame,
        column_name: str,
    ) -> int:
        """Count values that became null after conversion (excluding empty strings).

        Returns the number of values that were non-null/non-empty before but null after.
        """
        # Count non-null, non-empty in original
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

    @staticmethod
    def _get_parser_for_type(type_name: str, type_checks: list[dict]) -> Callable[[Column], Column] | None:
        """Get the parser function for a given type name."""
        if type_name == 'string':
            return None  # No parser needed for string
        for type_check in type_checks:
            if type_check['name'] == type_name:
                return type_check['parser']
        return None

    @staticmethod
    def _apply_type_with_retry(
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
        parser = SparkCleaner._get_parser_for_type(current_type, type_checks)
        if parser is None:
            # String type or unknown - no conversion needed
            return dataframe, 'string'

        converted_df = dataframe.withColumn(
            column_name, parser(spark_functions.col(column_name))
        )

        # Check for conversion failures
        failures = SparkCleaner._detect_conversion_failures(dataframe, converted_df, column_name)

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

            parser = SparkCleaner._get_parser_for_type(fallback_type, type_checks)
            if parser is None:
                continue

            converted_df = dataframe.withColumn(
                column_name, parser(spark_functions.col(column_name))
            )

            failures = SparkCleaner._detect_conversion_failures(dataframe, converted_df, column_name)

            if failures == 0:
                logger.info(f"Casting column '{column_name}' to {fallback_type} (fallback from {inferred_type}).")
                return converted_df, fallback_type

            current_type = fallback_type

        # All fallbacks failed, keep as string
        logger.warning(f"Column '{column_name}' kept as string (all type conversions failed).")
        return dataframe, 'string'

    @staticmethod
    def clean_all_types(
        dataframe: DataFrame,
        source_timezone: str = "UTC",
        sampling_config: SamplingConfig | None = None,
    ) -> DataFrame:
        """
        Cleans and casts all columns in a Spark DataFrame to their most appropriate type.

        When sampling is enabled, types are inferred from a sample and then validated
        against the full data. If validation fails, the type is automatically broadened
        using the fallback hierarchy (e.g., integer -> float -> string).

        Uses native Spark SQL operations for optimal performance (no Python UDFs).
        All datetime columns are normalized to UTC for consistent serialization.

        Args:
            dataframe: The DataFrame to clean
            source_timezone: The timezone to assume for timezone-naive datetime strings.
                           Defaults to "UTC". Timezone-aware strings are handled correctly
                           regardless of this setting.
            sampling_config: Configuration for sampling-based type inference.
                           Defaults to SamplingConfig() which enables sampling.
                           Pass SamplingConfig(enabled=False) to disable sampling.
        """
        if sampling_config is None:
            sampling_config = SamplingConfig()

        type_checks = SparkCleaner._get_type_checks(source_timezone)

        # Cache the original DataFrame - we'll need it for retry logic
        cached_dataframe = dataframe.cache()
        total_rows = cached_dataframe.count()

        # Get sample or use full data
        sample_df = SparkCleaner._get_sample(cached_dataframe, sampling_config, total_rows)
        use_sampling = sample_df is not None

        if use_sampling:
            logger.info(f"Using sampling for type inference (config: fraction={sampling_config.fraction}, "
                       f"min_rows={sampling_config.min_rows}, max_rows={sampling_config.max_rows})")
            # Cache sample for efficient reuse
            sample_df = sample_df.cache()
            column_type_mapping = SparkCleaner._infer_types_from_dataframe(sample_df, type_checks)
            sample_df.unpersist()
        else:
            if sampling_config.enabled:
                logger.info(f"Skipping sampling (total_rows={total_rows} < min_rows={sampling_config.min_rows})")
            column_type_mapping = SparkCleaner._infer_types_from_dataframe(cached_dataframe, type_checks)

        # Apply transformations with retry logic if sampling was used
        cleaned_dataframe = cached_dataframe
        for column_name, chosen_type in column_type_mapping.items():
            if chosen_type is None:
                logger.debug(f"Column '{column_name}' kept as original type (no type match).")
                continue

            if use_sampling:
                # Use retry logic when sampling was used
                cleaned_dataframe, final_type = SparkCleaner._apply_type_with_retry(
                    cleaned_dataframe, column_name, chosen_type['name'], type_checks
                )
            else:
                # Direct conversion when full data was used for inference
                logger.info(f"Casting column '{column_name}' to {chosen_type['name']}.")
                cleaned_dataframe = cleaned_dataframe.withColumn(
                    column_name, chosen_type['parser'](spark_functions.col(column_name))
                )

        # Unpersist the cached DataFrame
        cached_dataframe.unpersist()

        return cleaned_dataframe

    @staticmethod
    def clean_df(
        dataframe: DataFrame,
        source_timezone: str = "UTC",
        sampling_config: SamplingConfig | None = None,
    ) -> DataFrame:
        """
        Drops fully empty rows and columns, then cleans the remaining data.

        All datetime columns are normalized to UTC for consistent serialization.

        Args:
            dataframe: The DataFrame to clean
            source_timezone: The timezone to assume for timezone-naive datetime strings.
                           Defaults to "UTC". Timezone-aware strings are handled correctly
                           regardless of this setting.
            sampling_config: Configuration for sampling-based type inference.
                           Defaults to SamplingConfig() which enables sampling.
                           Pass SamplingConfig(enabled=False) to disable sampling.
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
        return SparkCleaner.clean_all_types(cleaned_dataframe, source_timezone, sampling_config)