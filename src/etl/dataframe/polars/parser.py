import logging
import math
from typing import Optional, Any

import polars as pl
from dateutil import parser

from ..common.constants import TRUTHY_VALUES, FALSY_VALUES

# Set up logging
logger = logging.getLogger(__name__)


class PolarsParser:
    """
    Parser class with static methods for parsing different data types in Polars.
    These methods are designed to work with Polars' expression system.
    """

    # Re-export for backwards compatibility
    TRUTHY_VALUES = list(TRUTHY_VALUES)
    FALSY_VALUES = list(FALSY_VALUES)

    # Cleaning patterns for numeric values
    NUMERIC_CLEANUP_PATTERNS = [
        (',', ''),
        ('\\$', ''),
        ('%', '')
    ]

    @staticmethod
    def parse_boolean_expr(column: str) -> pl.Expr:
        """
        Create a Polars expression for parsing boolean values.
        Returns an expression that converts various truthy/falsy strings to boolean.
        Handles empty strings and whitespace by converting them to null before boolean conversion.
        """
        # Normalize input: cast to Utf8 safely, handle nulls, strip whitespace
        original = pl.col(column)
        cleaned_utf8 = original.cast(pl.Utf8, strict=False)
        cleaned_utf8 = cleaned_utf8.str.strip_chars()
        
        # Handle nulls and empty strings after stripping
        is_null_or_empty = (cleaned_utf8.is_null() | (cleaned_utf8 == ""))
        
        # Apply boolean parsing with proper null handling
        return pl.when(is_null_or_empty)\
            .then(None)\
            .otherwise(cleaned_utf8.map_elements(PolarsParser.parse_bool_value, return_dtype=pl.Boolean))

    @staticmethod
    def parse_bool_value(val: Any) -> Optional[bool]:
        if val is None:
            return None
        val_lower = str(val).strip().lower()
        if val_lower == "":
            return None
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
        Handles empty strings and whitespace by converting them to null before numeric conversion.
        Also handles cases where numeric cleanup results in an empty string.
        """
        # Normalize input to string and trim whitespace
        expr = pl.col(column).cast(pl.Utf8, strict=False)
        expr = expr.str.strip_chars()
        
        # Handle empty strings and whitespace by converting to null
        is_null_or_empty = (expr.is_null() | (expr == ""))
        
        # Apply numeric cleanup patterns
        for pattern, replacement in PolarsParser.NUMERIC_CLEANUP_PATTERNS:
            expr = expr.str.replace_all(pattern, replacement)
            
        # After cleanup, check again for empty strings (e.g., "N/A" -> "" after cleanup)
        is_null_or_empty = is_null_or_empty | (expr == "")

        # Convert to float, handling nulls and empty strings properly
        return pl.when(is_null_or_empty).then(None).otherwise(expr.cast(pl.Float64, strict=False))


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
        Parse a date value using dateutil.
        :param value: The value to be parsed as a date.
        :return: The parsed date value, or None if parsing fails.
        """
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        try:
            return parser.parse(str(value).strip())
        except Exception:
            return None


    @staticmethod
    def parse_integer_expr(column: str) -> pl.Expr:
        """
        Create a Polars expression for parsing integer values.
        First cleans the value as a float, then converts to integer if it's a whole number.
        Handles empty strings and whitespace by converting them to null before numeric conversion.
        Also handles cases where numeric cleanup results in an empty string.
        """
        # First, get the cleaned float value using parse_float_expr logic
        original = pl.col(column)
        # Cast to Utf8 safely to allow string operations even if the column is numeric
        cleaned_utf8 = original.cast(pl.Utf8, strict=False)
        # Trim whitespace and treat empty strings as null so downstream numeric casts won't error
        cleaned_utf8 = cleaned_utf8.str.strip_chars()
        
        # Handle empty strings and whitespace by converting to null
        is_null_or_empty = (cleaned_utf8.is_null() | (cleaned_utf8 == ""))
        
        # Apply numeric cleanup patterns
        for pattern, replacement in PolarsParser.NUMERIC_CLEANUP_PATTERNS:
            cleaned_utf8 = cleaned_utf8.str.replace_all(pattern, replacement)
            
        # After cleanup, check again for empty strings (e.g., "N/A" -> "" after cleanup)
        is_null_or_empty = is_null_or_empty | (cleaned_utf8 == "")
            
        # Convert to float, handling nulls and empty strings properly
        cleaned_float = pl.when(is_null_or_empty).then(None).otherwise(cleaned_utf8.cast(pl.Float64, strict=False))

        # Then check if it's a whole number and cast to integer if so
        return (pl.when(cleaned_float.is_null())
                .then(None)
                .when(cleaned_float == cleaned_float.round(0))
                .then(cleaned_float.cast(pl.Int64))
                .otherwise(cleaned_float))
