import hashlib
import re
import polars as pl
from dateutil import parser
from typing import Union, List, Optional, Callable, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)


class PolarsParser:
    TRUTHY_VALUES = ['y', 'yes', 't', 'true', 'on', '1']
    FALSY_VALUES = ['n', 'no', 'f', 'false', 'off', '0']

    # Cleaning patterns for numeric values
    NUMERIC_CLEANUP_PATTERNS = [
        (',', ''),
        ('\\$', ''),
        ('%', '')
    ]

    @staticmethod
    def _is_null_or_nan(value: Any) -> bool:
        """
        Check if a value is None or NaN.
        :param value: The value to check
        :return: True if value is None or NaN, False otherwise
        """
        return value is None or (isinstance(value, float) and str(value) == 'nan')
    """
    Parser class with static methods for parsing different data types in Polars.
    These methods are designed to work with Polars' expression system.
    """
    @staticmethod
    def parse_boolean_expr(column: str) -> pl.Expr:
        """
        Create a Polars expression for parsing boolean values.
        Returns an expression that converts various truthy/falsy strings to boolean.
        """
        return pl.col(column).map_elements(PolarsParser.parse_bool_value, return_dtype=pl.Boolean)

    @staticmethod
    def parse_bool_value(val: Any) -> Optional[bool]:
        if val is None:
            return None
        val_lower = str(val).lower()
        if val_lower in PolarsParser.TRUTHY_VALUES:
            return True
        elif val_lower in PolarsParser.FALSY_VALUES:
            return False
        else:
            # Return None for non-boolean-like values instead of raising to keep pipelines resilient
            return None


    @staticmethod
    def parse_float_expr(column: str) -> pl.Expr:
        """
        Create a Polars expression for parsing float values.
        Returns an expression that cleans and converts strings to float.
        """
        expr = pl.col(column).cast(pl.Utf8, strict=False)
        for pattern, replacement in PolarsParser.NUMERIC_CLEANUP_PATTERNS:
            expr = expr.str.replace_all(pattern, replacement)

        return expr.cast(pl.Float64, strict=False)


    @staticmethod
    def parse_date_expr(column: str) -> pl.Expr:
        """
        Create a Polars expression for parsing date values.
        Strategy: try fast vectorized strptime with common formats, then fall back to
        python-side dateutil.parser.parse for anything that didn't match. This keeps
        performance reasonable while being highly tolerant on messy inputs.
        All mismatches yield nulls (no exceptions).
        """
        expr_utf8 = pl.col(column).cast(pl.Utf8, strict=False)
        vectorized = PolarsParser.parse_date_expr_vectorized(column)
        # Fallback to dateutil for anything the vectorized passes couldn't parse
        dateutil_fallback = expr_utf8.map_elements(
            PolarsParser.parse_date,
            return_dtype=pl.Datetime,
            skip_nulls=True,
        )
        return pl.coalesce([vectorized, dateutil_fallback])

    @staticmethod
    def parse_date_expr_vectorized(column: str) -> pl.Expr:
        """
        Fast, vectorized date parsing using a handful of common formats.
        Helpful for counting and for the first pass in hybrid parsing.
        """
        expr_utf8 = pl.col(column).cast(pl.Utf8, strict=False)
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%b %d, %Y",
            "%B %d, %Y",
            "%m/%d/%Y",
        ]
        parsed_candidates = [expr_utf8.str.strptime(pl.Datetime, fmt, strict=False) for fmt in formats]
        return pl.coalesce(parsed_candidates)

    @staticmethod
    def parse_date(value: Any) -> Optional[Any]:
        """
        This function is used to parse a date value.
        :param value: The value to be parsed as a date.
        :return: The parsed date value.
        """
        if value is None or (isinstance(value, float) and str(value) == 'nan'):
            return None
        try:
            return parser.parse(str(value).strip())
        except Exception:
            # Return None when parsing fails to avoid raising during vectorized operations
            return None


    @staticmethod
    def parse_integer_expr(column: str) -> pl.Expr:
        """
        Create a Polars expression for parsing integer values.
        First cleans the value as a float, then converts to integer if it's a whole number.
        """
        # First, get the cleaned float value using parse_float_expr logic
        original = pl.col(column)
        # Cast to Utf8 safely to allow string replacements even if the column is numeric
        cleaned_utf8 = original.cast(pl.Utf8, strict=False)
        for pattern, replacement in PolarsParser.NUMERIC_CLEANUP_PATTERNS:
            cleaned_utf8 = cleaned_utf8.str.replace_all(pattern, replacement)
        cleaned_float = cleaned_utf8.cast(pl.Float64, strict=False)

        # Then check if it's a whole number and cast to integer if so
        return (pl.when(cleaned_float.is_null())
                .then(None)
                .when(cleaned_float == cleaned_float.round(0))
                .then(cleaned_float.cast(pl.Int64))
                .otherwise(cleaned_float))
