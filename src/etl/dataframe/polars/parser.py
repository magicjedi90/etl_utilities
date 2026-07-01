import logging
import math
import re
from typing import Optional, Any

import polars as pl
from dateutil import parser

from ..common.constants import TRUTHY_VALUES, FALSY_VALUES, NUMERIC_CLEANUP_CHARS, POLARS_DATE_FORMATS

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

    # Cleaning patterns for numeric values (regex, replacement) — derived from
    # the shared cleanup characters. Kept as an attribute for backwards compat.
    NUMERIC_CLEANUP_PATTERNS = [(re.escape(char), '') for char in NUMERIC_CLEANUP_CHARS]

    @staticmethod
    def parse_boolean_expr(column: str) -> pl.Expr:
        """
        Create a Polars expression for parsing boolean values.
        Returns an expression that converts various truthy/falsy strings to boolean.
        Handles empty strings and whitespace by converting them to null before boolean conversion.
        Fully vectorized — no Python-level row iteration.
        """
        cleaned_utf8 = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()
        is_null_or_empty = cleaned_utf8.is_null() | (cleaned_utf8 == "")
        lowered = cleaned_utf8.str.to_lowercase()

        return (
            pl.when(is_null_or_empty)
            .then(pl.lit(None, dtype=pl.Boolean))
            .when(lowered.is_in(PolarsParser.TRUTHY_VALUES))
            .then(pl.lit(True))
            .when(lowered.is_in(PolarsParser.FALSY_VALUES))
            .then(pl.lit(False))
            .otherwise(pl.lit(None, dtype=pl.Boolean))
        )

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
        Normalizes to string, trims whitespace, strips numeric cleanup characters
        ($, %, ,), and converts to Float64. Empty strings (before or after
        cleanup) become null, and inf/nan are rejected — they parse as valid
        Float64 but are not meaningful numeric data.
        """
        expr = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()

        # Handle empty strings and whitespace by converting to null
        is_null_or_empty = expr.is_null() | (expr == "")

        # Apply numeric cleanup patterns
        for pattern, replacement in PolarsParser.NUMERIC_CLEANUP_PATTERNS:
            expr = expr.str.replace_all(pattern, replacement)

        # After cleanup, check again for empty strings (e.g., "$" -> "" after cleanup)
        is_null_or_empty = is_null_or_empty | (expr == "")

        raw_float = expr.cast(pl.Float64, strict=False)
        return pl.when(is_null_or_empty | raw_float.is_nan() | raw_float.is_infinite()).then(None).otherwise(raw_float)

    @staticmethod
    def parse_integer_expr(column: str) -> pl.Expr:
        """
        Create a Polars expression for parsing integer values.
        Builds on parse_float_expr, then casts whole numbers to Int64 while
        leaving fractional values as floats. Note the expression's static dtype
        is Float64 (the when/then/otherwise supertype); use apply_cleaning_plan's
        "int64" kind for a strict integer column.
        """
        cleaned_float = PolarsParser.parse_float_expr(column)
        return (pl.when(cleaned_float.is_null())
                .then(None)
                .when(cleaned_float == cleaned_float.round(0))
                .then(cleaned_float.cast(pl.Int64, strict=False))
                .otherwise(cleaned_float))

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
        # Fallback to dateutil, but only for values the vectorized pass missed —
        # values it already parsed are masked to null and skipped, so the
        # Python-level fallback costs nothing when the fast path covers everything.
        unmatched = (
            pl.when(vectorized.is_null())
            .then(expr_utf8)
            .otherwise(pl.lit(None, dtype=pl.Utf8))
        )
        dateutil_fallback = unmatched.map_elements(
            PolarsParser.parse_date,
            return_dtype=pl.Datetime,
            skip_nulls=True,
        )
        return pl.coalesce([vectorized, dateutil_fallback])

    @staticmethod
    def parse_date_expr_vectorized(column: str) -> pl.Expr:
        """
        Fast, vectorized date parsing using the shared date formats
        (first matching format wins per value).
        """
        expr_utf8 = pl.col(column).cast(pl.Utf8, strict=False)
        parsed_candidates = [
            expr_utf8.str.strptime(pl.Datetime, fmt, strict=False)
            for fmt in POLARS_DATE_FORMATS
        ]
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
