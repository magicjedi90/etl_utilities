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
        unique_column_pairs: list[tuple[Hashable, Hashable]] = []
        combo_df = pd.DataFrame()
        for first_column, second_column in itertools.combinations(column_list, 2):
            if first_column in unique_columns or second_column in unique_columns:
                continue
            combo_df["combo"] = df[first_column].astype(str) + df[second_column].astype(str)
            combined_unique = combo_df["combo"].unique()
            combined_unique_count = combined_unique.size
            if combined_unique_count == total_records:
                unique_column_pairs.append((first_column, second_column))
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
            column_metadata.update(Analyzer._probe_numeric(series, decimal_places))
            # Boolean outranks numeric: a column of all 0/1/true/false is boolean.
            if Analyzer._parses_as(series, Parser.parse_boolean):
                column_metadata['data_type'] = 'boolean'
            if column_metadata['data_type'] is None and Analyzer._parses_as(series, Parser.parse_date):
                column_metadata['data_type'] = 'datetime'
            if column_metadata['data_type'] is None:
                column_metadata['data_type'] = 'string'
                column_metadata['max_str_size'] = series.apply(str).str.len().max()
            column_metadata_list.append(column_metadata)
        return column_metadata_list

    @staticmethod
    def _parses_as(series: pd.Series, parse_function) -> bool:
        """Return True if every value in the series parses with parse_function."""
        try:
            series.apply(parse_function)
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    @staticmethod
    def _probe_numeric(series: pd.Series, decimal_places: int) -> dict:
        """
        Probe a series as float, then refine to integer if every value is whole.

        Returns the metadata fields to merge ({} when the series is not numeric).
        float_precision is left-of-decimal digits plus decimal_places for floats,
        and just the left-of-decimal digits for integers.
        """
        try:
            parsed = series.apply(Parser.parse_float).dropna()
        except (ValueError, TypeError, OverflowError):
            return {}
        magnitude = max(abs(parsed.max()), abs(parsed.min())) if not parsed.empty else 0
        left_digits = int(math.log10(magnitude)) + 1 if magnitude > 0 else 1
        metadata = {
            'data_type': 'float',
            'float_precision': left_digits + decimal_places,
            'decimal_places': decimal_places,
        }
        if Analyzer._parses_as(series, Parser.parse_integer):
            metadata.update({
                'data_type': 'integer',
                'biggest_num': series.max(),
                'smallest_num': series.min(),
                'float_precision': left_digits,
            })
        return metadata

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
