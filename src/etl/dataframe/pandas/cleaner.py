# src/etl/dataframe/pandas/cleaner.py
"""DataFrame cleaning operations for Pandas."""

import logging

import pandas as pd
from dateutil import parser as dateutil_parser

from .parser import Parser
from ..common.utils import standardize_column_name, compute_hash

logger = logging.getLogger(__name__)


class Cleaner:
    """
    Provides static methods for data cleaning operations on a pandas DataFrame.

    Methods include column name standardization, type cleaning (numbers, dates,
    booleans), hash column generation, and column coalescing.
    """

    @staticmethod
    def column_names_to_snake_case(df: pd.DataFrame) -> None:
        """Convert DataFrame column names to snake_case in place."""
        df.columns = [standardize_column_name(name) for name in df.columns]

    @staticmethod
    def column_names_to_pascal_case(df: pd.DataFrame) -> None:
        """Convert DataFrame column names to PascalCase in place."""
        df.columns = ["".join(standardize_column_name(name).title().split('_')) for name in df.columns]

    @staticmethod
    def clean_series(series: pd.Series, clean_function) -> pd.Series:
        """
        Apply a cleaning function to a series.

        Args:
            series: The pandas Series to clean.
            clean_function: The function to apply to each element.

        Returns:
            The cleaned Series with appropriate dtype.

        Raises:
            ValueError, TypeError, ParserError, OverflowError: If cleaning fails.
        """
        try:
            cleaned_series = series.apply(clean_function)
            series_dtype = clean_function.__annotations__.get('return', None)
            if series_dtype:
                cleaned_series = cleaned_series.astype(series_dtype)
            return cleaned_series
        except (ValueError, TypeError, dateutil_parser.ParserError, OverflowError):
            raise

    @staticmethod
    def clean_numbers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean all numeric columns by parsing floats and integers.

        Args:
            df: The DataFrame to clean.

        Returns:
            The DataFrame with cleaned numeric columns.
        """
        for column, series in df.items():
            df[column] = Cleaner.clean_series(series, Parser.parse_float)
            try:
                df[column] = Cleaner.clean_series(df[column], Parser.parse_integer)
            except ValueError:
                pass
        return df

    @staticmethod
    def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean all date columns by parsing date formats.

        Args:
            df: The DataFrame to clean.

        Returns:
            The DataFrame with cleaned date columns.
        """
        for column, series in df.items():
            df[column] = Cleaner.clean_series(series, Parser.parse_date)
        return df

    @staticmethod
    def clean_bools(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean all boolean columns by parsing boolean values.

        Args:
            df: The DataFrame to clean.

        Returns:
            The DataFrame with cleaned boolean columns.
        """
        for column, series in df.items():
            df[column] = Cleaner.clean_series(series, Parser.parse_boolean)
        return df

    @staticmethod
    def clean_all_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive type cleaning on all columns.

        Tries multiple parsing functions (float, integer, boolean, date) on each
        column and applies the first successful conversion.

        Args:
            df: The DataFrame to clean.

        Returns:
            The DataFrame with all columns cleaned and converted to appropriate dtypes.
        """
        try_functions = [Parser.parse_float, Parser.parse_integer, Parser.parse_boolean, Parser.parse_date]
        for column, series in df.items():
            if series.dropna().empty:
                logger.info(f'{column} is empty skipping cleaning')
                df[column] = df[column].astype(str)
                continue
            is_column_clean = False
            for func in try_functions:
                if is_column_clean and func == Parser.parse_date:
                    continue
                try:
                    series = Cleaner.clean_series(series, func)
                    df[column] = series
                    is_column_clean = True
                    logger.info(f'{column} was cleaned with {func.__name__}')
                except (ValueError, TypeError, dateutil_parser.ParserError, OverflowError) as error:
                    logger.debug(f'{column} failed cleaning with {func.__name__}: {error}')
        df = df.convert_dtypes()
        return df

    @staticmethod
    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop empty rows/columns and clean all types.

        Args:
            df: The DataFrame to clean.

        Returns:
            The cleaned DataFrame.
        """
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
        return Cleaner.clean_all_types(df)

    @staticmethod
    def generate_hash_column(df: pd.DataFrame, columns_to_hash, new_column_name) -> pd.DataFrame:
        """
        Generate a hash column based on specified columns.

        Args:
            df: The DataFrame to modify.
            columns_to_hash: List of column names to include in hash.
            new_column_name: Name for the new hash column.

        Returns:
            The DataFrame with the added hash column.
        """
        df[new_column_name] = df[columns_to_hash].astype(str).sum(axis=1).apply(compute_hash)
        return df

    @staticmethod
    def coalesce_columns(df: pd.DataFrame, columns_to_coalesce, target_column, drop=False) -> pd.DataFrame:
        """
        Coalesce multiple columns into one, taking the first non-null value.

        Args:
            df: The DataFrame to modify.
            columns_to_coalesce: List of column names to coalesce.
            target_column: Name for the coalesced column.
            drop: Whether to drop the original columns.

        Returns:
            The DataFrame with the coalesced column.
        """
        df[target_column] = df[columns_to_coalesce].bfill(axis=1).iloc[:, 0]
        if drop:
            if target_column in columns_to_coalesce:
                columns_to_coalesce.remove(target_column)
            df = df.drop(columns=columns_to_coalesce)
        return df
