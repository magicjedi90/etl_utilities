# src/etl/dataframe/spark/cleaner.py
"""SparkCleaner - orchestrator for Spark DataFrame cleaning operations."""

import logging

from pyspark.sql import functions as spark_functions
from pyspark.sql import DataFrame

from ..cleaner import standardize_column_name
from .config import SamplingConfig
from .type_inference import (
    get_type_checks,
    get_sample,
    infer_types_from_dataframe,
    apply_type_with_retry,
)

logger = logging.getLogger(__name__)


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

        type_checks = get_type_checks(source_timezone)

        # Cache the original DataFrame - we'll need it for retry logic
        cached_dataframe = dataframe.cache()
        total_rows = cached_dataframe.count()

        # Get sample or use full data
        sample_df = get_sample(cached_dataframe, sampling_config, total_rows)
        use_sampling = sample_df is not None

        if use_sampling:
            logger.info(f"Using sampling for type inference (config: fraction={sampling_config.fraction}, "
                       f"min_rows={sampling_config.min_rows}, max_rows={sampling_config.max_rows})")
            # Cache sample for efficient reuse
            sample_df = sample_df.cache()
            column_type_mapping = infer_types_from_dataframe(sample_df, type_checks)
            sample_df.unpersist()
        else:
            if sampling_config.enabled:
                logger.info(f"Skipping sampling (total_rows={total_rows} < min_rows={sampling_config.min_rows})")
            column_type_mapping = infer_types_from_dataframe(cached_dataframe, type_checks)

        # Apply transformations with retry logic if sampling was used
        cleaned_dataframe = cached_dataframe
        for column_name, chosen_type in column_type_mapping.items():
            if chosen_type is None:
                logger.debug(f"Column '{column_name}' kept as original type (no type match).")
                continue

            if use_sampling:
                # Use retry logic when sampling was used
                cleaned_dataframe, final_type = apply_type_with_retry(
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
