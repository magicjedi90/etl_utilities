# src/etl/spark/cleaner.py

import pyspark.sql.functions as functions
from src.etl.dataframe.cleaner import standardize_column_name
from pyspark.sql import DataFrame

from .udfs import (
    is_boolean_udf, is_integer_udf, is_float_udf, is_date_udf,
    parse_boolean_udf, parse_integer_udf, parse_float_udf, parse_date_udf
)


class SparkCleaner:
    @staticmethod
    def column_names_to_snake_case(df: DataFrame) -> DataFrame:
        """Converts DataFrame column names to snake_case for Spark."""
        new_df = df
        for column in new_df.columns:
            new_df = new_df.withColumnRenamed(column, standardize_column_name(column))
        return new_df

    @staticmethod
    def clean_all_types(df: DataFrame, threshold: float = 0.95) -> DataFrame:
        """
        Cleans and casts all columns in a Spark DataFrame to their most appropriate type.
        """

        # Define the order of type checks, from most specific to most general.
        type_checks = [
            {'name': 'boolean', 'udf': is_boolean_udf, 'parser': parse_boolean_udf},
            {'name': 'integer', 'udf': is_integer_udf, 'parser': parse_integer_udf},
            {'name': 'float', 'udf': is_float_udf, 'parser': parse_float_udf},
            {'name': 'datetime', 'udf': is_date_udf, 'parser': parse_date_udf},
        ]

        cleaned_df = df

        for col_name in df.columns:
            non_null_count = df.filter(functions.col(col_name).isNotNull()).count()
            if non_null_count == 0:
                print(f"Column '{col_name}' is empty, skipping.")
                continue

            for check in type_checks:
                # Check how many non-null values conform to the type
                matches = df.withColumn(
                    "is_type", check['udf'](functions.col(col_name))
                ).agg(functions.sum(functions.when(functions.col("is_type") == True, 1).otherwise(0)).alias(
                    "match_count")).first().match_count

                # If a high percentage of values match, cast the column and stop checking
                if matches and (matches / non_null_count) >= threshold:
                    print(f"Casting column '{col_name}' to {check['name']}.")
                    cleaned_df = cleaned_df.withColumn(col_name, check['parser'](functions.col(col_name)))
                    break  # Move to the next column
            else:
                # If no type matches, default to string
                print(f"Column '{col_name}' could not be cast to a specific type, defaulting to string.")
                cleaned_df = cleaned_df.withColumn(col_name, functions.col(col_name).cast("string"))

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
            print(f"Dropping all-null columns: {cols_to_drop}")
            cleaned_df = cleaned_df.drop(*cols_to_drop)

        # 3. Clean the types of the remaining columns
        return SparkCleaner.clean_all_types(cleaned_df)
