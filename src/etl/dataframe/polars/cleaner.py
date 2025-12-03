import hashlib
import re
import polars as pl
from typing import Union, List, Optional, Callable, Any
import logging

from .parser import PolarsParser
from ..cleaner import standardize_column_name, compute_hash

# Set up logging
logger = logging.getLogger(__name__)


class PolarsCleaner:
    """
    This class provides static methods for data cleaning operations on a Polars DataFrame.
    Optimized for Polars' lazy evaluation and expression system.
    """
    
    @staticmethod
    def column_names_to_snake_case(df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert column names to snake_case format.
        :param df: Polars DataFrame
        :return: DataFrame with standardized column names
        """
        new_columns = [standardize_column_name(name) for name in df.columns]
        return df.rename(dict(zip(df.columns, new_columns)))
    
    @staticmethod
    def column_names_to_pascal_case(df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert column names to PascalCase format.
        :param df: Polars DataFrame
        :return: DataFrame with PascalCase column names
        """
        new_columns = ["".join(standardize_column_name(name).title().split('_')) 
                      for name in df.columns]
        return df.rename(dict(zip(df.columns, new_columns)))


    @staticmethod
    def clean_numbers(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Clean numeric columns by parsing floats and integers.
        :param df: Polars DataFrame
        :param columns: List of columns to clean (if None, clean all columns)
        :return: DataFrame with cleaned numeric columns
        """
        if columns is None:
            columns = df.columns

        for column in columns:
            try:
                # Use the parse_integer_expr from PolarsParser which handles
                # float cleaning and integer conversion for whole numbers
                df = df.with_columns(
                    PolarsParser.parse_integer_expr(column).alias(column)
                )
            except Exception as e:
                logger.debug(f"Column {column} could not be cleaned as number: {e}")

        return df
    
    @staticmethod
    def clean_dates(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Clean date columns by parsing various date formats.
        :param df: Polars DataFrame
        :param columns: List of columns to clean (if None, clean all columns)
        :return: DataFrame with cleaned date columns
        """
        if columns is None:
            columns = df.columns
        
        for column in columns:
            try:
                df = df.with_columns(
                    PolarsParser.parse_date_expr(column)
                    .alias(column)
                )
            except Exception as e:
                logger.debug(f"Column {column} could not be cleaned as date: {e}")
        
        return df
    
    @staticmethod
    def clean_bools(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Clean boolean columns by parsing various truthy/falsy values.
        :param df: Polars DataFrame
        :param columns: List of columns to clean (if None, clean all columns)
        :return: DataFrame with cleaned boolean columns
        """
        if columns is None:
            columns = df.columns
        
        for column in columns:
            try:
                df = df.with_columns(
                    PolarsParser.parse_boolean_expr(column)
                    .alias(column)
                )
            except Exception as e:
                logger.debug(f"Column {column} could not be cleaned as boolean: {e}")
        
        return df

    @staticmethod
    def clean_all_types(df: pl.DataFrame, columns: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Perform comprehensive cleaning on all columns by trying different parsing functions.
        Strategy:
        - Build safe expressions for bool, number, and date that return None for incompatible values.
        - Coalesce them in priority order, falling back to the original value to avoid data loss.
        - Apply for all target columns in a single pass for efficiency.
        :param df: Polars DataFrame
        :param columns: List of columns to clean (if None, clean all columns)
        :return: DataFrame with all columns cleaned
        """
        if columns is None:
            columns = df.columns

        cleaning_expressions = []

        for column in columns:
            # Skip if column has no non-null values
            if df.select(pl.col(column).drop_nulls()).height == 0:
                logger.info(f"{column} is empty, skipping cleaning")
                cleaning_expressions.append(pl.col(column))
                continue

            original = pl.col(column)
            # Treat empty or whitespace-only strings as nulls for the purpose of
            # determining if a parser covers all meaningful values. This allows
            # columns with empty strings to still be cast to their numeric/date/bool
            # dtypes while preserving empties as nulls.
            original_utf8 = original.cast(pl.Utf8, strict=False)
            original_stripped = original_utf8.str.strip_chars()
            effective_original = (
                pl.when(original_stripped == "")
                .then(None)
                .otherwise(original)
            )
            bool_expr = PolarsParser.parse_boolean_expr(column)
            num_expr = PolarsParser.parse_integer_expr(column)
            # Use fast vectorized date parsing for counting to avoid expensive/python fallback affecting selection
            date_expr_count = PolarsParser.parse_date_expr_vectorized(column)
            # But keep a tolerant, full date parser for final application if date wins
            date_expr_full = PolarsParser.parse_date_expr(column)

            # Evaluate non-null counts for each candidate expression
            try:
                counts = df.select([
                    effective_original.is_not_null().sum().alias("orig"),
                    bool_expr.is_not_null().sum().alias("bool"),
                    num_expr.is_not_null().sum().alias("num"),
                    date_expr_count.is_not_null().sum().alias("date"),
                ]).row(0)
                original_count, bool_count, num_cnt, date_cnt = counts
            except Exception as e:
                logger.debug(f"Could not evaluate counts for {column}: {e}")
                cleaning_expressions.append(original)
                continue

            # Choose the best parser among bool/num/date based on highest non-null count
            # Tie-breaker priority: bool > num > date
            candidates = [
                (bool_count, 'bool'),
                (num_cnt, 'num'),
                (date_cnt, 'date'),
            ]
            # Sort by count desc, then by priority order as listed
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_count, top_kind = candidates[0]
            # For now only convert if all elements are cleaned - potentially add in a threshold later
            if top_count == original_count and top_count > 0:
                if top_kind == 'bool':
                    chosen = bool_expr
                elif top_kind == 'num':
                    chosen = num_expr
                else:
                    chosen = date_expr_full
            else:
                chosen = original

            cleaning_expressions.append(chosen.alias(column))

        # Apply all chosen expressions in a single pass
        df = df.with_columns(cleaning_expressions)
        return PolarsCleaner.optimize_dtypes(df)


    @staticmethod
    def clean_df(df: pl.DataFrame) -> pl.DataFrame:
        """
        Comprehensive DataFrame cleaning - removes empty rows/columns and cleans all types.
        :param df: Polars DataFrame
        :return: Cleaned DataFrame
        """
        # Remove columns that are all null
        df = df.select([col for col in df.columns if df.select(pl.col(col).is_not_null().any()).item()])
        
        return PolarsCleaner.clean_all_types(df)
    
    @staticmethod
    def generate_hash_column(df: pl.DataFrame, columns_to_hash: List[str], new_column_name: str) -> pl.DataFrame:
        """
        Generate a hash column based on specified columns.
        :param df: Polars DataFrame
        :param columns_to_hash: List of column names to include in hash
        :param new_column_name: Name for the new hash column
        :return: DataFrame with added hash column
        """
        # Validate inputs
        if not columns_to_hash:
            raise ValueError("columns_to_hash cannot be empty")
        
        missing_cols = [col for col in columns_to_hash if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        if new_column_name in df.columns:
            raise ValueError(f"Column '{new_column_name}' already exists in DataFrame")
        
        # Use map_elements instead of deprecated apply
        hash_expr = pl.concat_str([pl.col(col).cast(pl.Utf8) for col in columns_to_hash]).map_elements(
            compute_hash, 
            return_dtype=pl.Utf8
        )
        return df.with_columns(hash_expr.alias(new_column_name))
    
    @staticmethod
    def coalesce_columns(df: pl.DataFrame, columns_to_coalesce: List[str], target_column: str, drop: bool = False) -> pl.DataFrame:
        """
        Coalesce multiple columns into one, taking the first non-null value.
        :param df: Polars DataFrame
        :param columns_to_coalesce: List of column names to coalesce
        :param target_column: Name for the coalesced column
        :param drop: Whether to drop the original columns
        :return: DataFrame with coalesced column
        """
        # Use coalesce function
        coalesce_expr = pl.coalesce([pl.col(col) for col in columns_to_coalesce])
        df = df.with_columns(coalesce_expr.alias(target_column))
        
        if drop:
            cols_to_drop = [col for col in columns_to_coalesce if col != target_column]
            df = df.drop(cols_to_drop)
        
        return df
    
    @staticmethod
    def optimize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
        """
        Optimize data types for memory efficiency.
        :param df: Polars DataFrame
        :return: DataFrame with optimized data types
        """
        # Convert integer columns to smallest possible type
        for col in df.columns:
            if df[col].dtype == pl.Int64:
                max_val = df[col].max()
                min_val = df[col].min()
                if min_val is not None and max_val is not None:
                    if min_val >= 0:
                        if max_val < 255:
                            df = df.with_columns(pl.col(col).cast(pl.UInt8))
                        elif max_val < 65535:
                            df = df.with_columns(pl.col(col).cast(pl.UInt16))
                        elif max_val < 4294967295:
                            df = df.with_columns(pl.col(col).cast(pl.UInt32))
                    else:
                        if min_val >= -128 and max_val < 127:
                            df = df.with_columns(pl.col(col).cast(pl.Int8))
                        elif min_val >= -32768 and max_val < 32767:
                            df = df.with_columns(pl.col(col).cast(pl.Int16))
                        elif min_val >= -2147483648 and max_val < 2147483647:
                            df = df.with_columns(pl.col(col).cast(pl.Int32))
        
        return df
    