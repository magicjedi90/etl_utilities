# src/etl/spark/cleaner.py

import logging

import pyspark.sql.functions as functions
from pyspark.sql import DataFrame

from ..cleaner import standardize_column_name
from .udfs import (
    is_boolean_udf, is_integer_udf, is_float_udf, is_date_udf,
    parse_boolean_udf, parse_integer_udf, parse_float_udf, parse_date_udf
)

logger = logging.getLogger(__name__)


class SparkCleaner:
    @staticmethod
    def column_names_to_snake_case(df: DataFrame) -> DataFrame:
        """Converts DataFrame column names to snake_case for Spark."""
        new_df = df
        for column in new_df.columns:
            new_df = new_df.withColumnRenamed(column, standardize_column_name(column))
        return new_df

    @staticmethod
    def clean_all_types(df: DataFrame) -> DataFrame:
        """
        Cleans and casts all columns in a Spark DataFrame to their most appropriate type.

        Only casts a column if ALL non-null values can be successfully parsed.
        Optimized to compute all type statistics in a single pass to minimize Spark jobs.
        """
        # Define the order of type checks, from most specific to most general.
        type_checks = [
            {'name': 'boolean', 'udf': is_boolean_udf, 'parser': parse_boolean_udf},
            {'name': 'integer', 'udf': is_integer_udf, 'parser': parse_integer_udf},
            {'name': 'float', 'udf': is_float_udf, 'parser': parse_float_udf},
            {'name': 'datetime', 'udf': is_date_udf, 'parser': parse_date_udf},
        ]

        # Cache the DataFrame to avoid recomputation
        df = df.cache()

        # Build aggregation expressions to compute all stats in a single pass
        agg_exprs = []
        for col_name in df.columns:
            # Count non-null values
            agg_exprs.append(
                functions.sum(functions.when(functions.col(col_name).isNotNull(), 1).otherwise(0))
                .alias(f"{col_name}__non_null")
            )
            # Count matches for each type
            for check in type_checks:
                agg_exprs.append(
                    functions.sum(
                        functions.when(check['udf'](functions.col(col_name)) == True, 1).otherwise(0)
                    ).alias(f"{col_name}__{check['name']}")
                )

        # Execute single aggregation to get all statistics
        stats_row = df.agg(*agg_exprs).first()

        # Determine best type for each column based on collected stats
        column_types = {}
        for col_name in df.columns:
            non_null_count = stats_row[f"{col_name}__non_null"]

            if non_null_count == 0:
                logger.info(f"Column '{col_name}' is empty, skipping.")
                column_types[col_name] = None
                continue

            # Find the first type where ALL non-null values match
            chosen_type = None
            for check in type_checks:
                match_count = stats_row[f"{col_name}__{check['name']}"]
                if match_count == non_null_count:
                    chosen_type = check
                    break

            column_types[col_name] = chosen_type

        # Apply transformations
        cleaned_df = df
        for col_name, chosen_type in column_types.items():
            if chosen_type is None:
                non_null_count = stats_row[f"{col_name}__non_null"]
                if non_null_count > 0:
                    logger.debug(f"Column '{col_name}' kept as original type (no full type match).")
            else:
                logger.info(f"Casting column '{col_name}' to {chosen_type['name']}.")
                cleaned_df = cleaned_df.withColumn(col_name, chosen_type['parser'](functions.col(col_name)))

        # Unpersist the cached DataFrame
        df.unpersist()

        return cleaned_df

    @staticmethod
    def clean_df(df: DataFrame) -> DataFrame:
        """
        Drops fully empty rows and columns, then cleans the remaining data.
        """
        # 1. Drop rows where all values are null
        cleaned_df = df.na.drop(how='all')

        # 2. Identify and drop columns where all values are null
        null_counts = cleaned_df.select(
            [functions.count(functions.when(functions.col(c).isNull(), c)).alias(c) for c in cleaned_df.columns]).first()

        total_rows = cleaned_df.count()
        cols_to_drop = [c for c in cleaned_df.columns if null_counts[c] == total_rows]

        if cols_to_drop:
            logger.info(f"Dropping all-null columns: {cols_to_drop}")
            cleaned_df = cleaned_df.drop(*cols_to_drop)

        # 3. Clean the types of the remaining columns
        return SparkCleaner.clean_all_types(cleaned_df)
