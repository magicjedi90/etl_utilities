# src/etl/dataframe/pandas/analyzer.py
"""DataFrame analysis utilities for Pandas."""

import itertools
import math
from typing import Hashable

import pandas as pd

from .parser import Parser


class Analyzer:
    """Provides static methods for analyzing pandas DataFrames."""

    @staticmethod
    def find_unique_columns(df: pd.DataFrame) -> list[Hashable]:
        """
        Find columns where all values are unique.

        Args:
            df: The DataFrame to analyze.

        Returns:
            List of column names with all unique values.
        """
        total_records = df.shape[0]
        unique_columns = []
        for column, series in df.items():
            column_unique = series.unique()
            column_unique_count = column_unique.size
            if column_unique_count == total_records:
                unique_columns.append(column)
        return unique_columns

    @staticmethod
    def find_unique_column_pairs(df: pd.DataFrame) -> list[tuple[Hashable, Hashable]]:
        """
        Find pairs of columns that together form unique combinations.

        Args:
            df: The DataFrame to analyze.

        Returns:
            List of tuples, each containing two column names that together are unique.
        """
        total_records = df.shape[0]
        column_list = df.columns
        unique_columns = Analyzer.find_unique_columns(df)
        unique_column_pairs = []
        combo_df = pd.DataFrame()
        for column_set in itertools.combinations(column_list, 2):
            if column_set is None:
                continue
            first_column = column_set[0]
            second_column = column_set[1]
            if first_column in unique_columns or second_column in unique_columns:
                continue
            combo_df["combo"] = df[first_column].astype(str) + df[second_column].astype(str)
            combined_unique = combo_df["combo"].unique()
            combined_unique_count = combined_unique.size
            if combined_unique_count == total_records:
                unique_column_pairs.append(column_set)
        return unique_column_pairs

    @staticmethod
    def find_empty_columns(df: pd.DataFrame) -> list[str]:
        """
        Find columns where all values are null.

        Args:
            df: The DataFrame to analyze.

        Returns:
            List of column names that are entirely null.
        """
        empty_columns = []
        for column, series in df.items():
            if series.dropna().empty:
                empty_columns.append(column.__str__())
        return empty_columns

    @staticmethod
    def generate_column_metadata(
        df: pd.DataFrame,
        primary_key: str | None,
        unique_columns: list[str] | None,
        decimal_places: int
    ) -> list[dict]:
        """
        Generate metadata for each column in the DataFrame.

        Args:
            df: The DataFrame to analyze.
            primary_key: The name of the primary key column, if any.
            unique_columns: List of columns known to have unique values.
            decimal_places: Number of decimal places for float precision calculation.

        Returns:
            List of dictionaries containing metadata for each column.
        """
        column_metadata_list = []
        if df.empty:
            return []
        for column, series in df.items():
            column_metadata = {
                'column_name': column,
                'data_type': None,
                'is_id': column == primary_key,
                'is_unique': unique_columns and column in unique_columns,
                'is_empty': False,
                'max_str_size': None,
                'float_precision': None,
                'decimal_places': None,
                'biggest_num': None,
                'smallest_num': None
            }
            if series.dropna().empty:
                column_metadata['is_empty'] = True
                column_metadata_list.append(column_metadata)
                continue
            try:
                series.apply(Parser.parse_float)
                no_null_series = series.dropna()
                if not no_null_series.eq(0).all():
                    left_digits = int(math.log10(abs(series.max()))) + 1
                    float_precision = left_digits + decimal_places
                    column_metadata['data_type'] = 'float'
                    column_metadata['float_precision'] = float_precision
                    column_metadata['decimal_places'] = decimal_places
                series.apply(Parser.parse_integer)
                biggest_num = series.max()
                smallest_num = series.min()
                column_metadata['data_type'] = 'integer'
                column_metadata['biggest_num'] = biggest_num
                column_metadata['smallest_num'] = smallest_num
                column_metadata['float_precision'] -= decimal_places
            except (ValueError, TypeError):
                pass
            try:
                series.apply(Parser.parse_boolean)
                column_metadata['data_type'] = 'boolean'
            except ValueError:
                pass
            if column_metadata['data_type'] is None:
                try:
                    series.apply(Parser.parse_date)
                    column_metadata['data_type'] = 'datetime'
                except (ValueError, TypeError, OverflowError):
                    pass
            if column_metadata['data_type'] is None:
                str_series = series.apply(str)
                largest_string_size = str_series.str.len().max()
                column_metadata['data_type'] = 'string'
                column_metadata['max_str_size'] = largest_string_size
            column_metadata_list.append(column_metadata)
        return column_metadata_list

    @staticmethod
    def find_categorical_columns(df: pd.DataFrame, unique_threshold: float = 1) -> list[Hashable]:
        """
        Find columns that are likely categorical based on unique value ratio.

        Args:
            df: The DataFrame to analyze.
            unique_threshold: Maximum ratio of unique values to total values (0-1).

        Returns:
            List of column names that are likely categorical.

        Raises:
            ValueError: If unique_threshold is not between 0 and 1.
        """
        if unique_threshold < 0 or unique_threshold > 1:
            raise ValueError('Unique threshold must be between 0 and 1')
        categorical_columns = []
        for column, series in df.items():
            no_null_series = series.dropna()
            if no_null_series.empty:
                continue
            column_count = no_null_series.size
            column_unique_count = no_null_series.unique().size
            unique_pct = column_unique_count / column_count
            if unique_pct <= unique_threshold:
                categorical_columns.append(column)
        return categorical_columns
