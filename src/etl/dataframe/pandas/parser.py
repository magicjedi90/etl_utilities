# src/etl/dataframe/pandas/parser.py
"""Type parsing functions for Pandas DataFrames."""

import pandas as pd
from dateutil import parser

from ..common.constants import TRUTHY_VALUES, FALSY_VALUES


class Parser:
    """Parser class with static methods for parsing different data types."""

    @staticmethod
    def parse_boolean(value):
        """
        Parse a boolean value from a given input.

        Args:
            value: The value to be parsed as a boolean.

        Returns:
            The parsed boolean value, or None if the value is null.

        Raises:
            ValueError: If the value is not a recognized boolean string.
        """
        if pd.isnull(value):
            return None
        value = str(value).lower()
        if value in TRUTHY_VALUES:
            return True
        elif value in FALSY_VALUES:
            return False
        else:
            raise ValueError(f"Invalid truth value: {value}")

    @staticmethod
    def parse_float(value):
        """
        Parse a given value as a float.

        Args:
            value: The value to parse as a float.

        Returns:
            The parsed float value, or None if the value is null.
        """
        if pd.isnull(value):
            return None
        cleaned_value = str(value).replace(',', '').replace('$', '').replace('%', '')
        return float(cleaned_value)

    @staticmethod
    def parse_date(value):
        """
        Parse a date value using dateutil.

        Args:
            value: The value to be parsed as a date.

        Returns:
            The parsed date value, or None if the value is null.
        """
        if pd.isnull(value):
            return None
        return parser.parse(str(value).strip())

    @staticmethod
    def parse_integer(value):
        """
        Parse an input value to an integer.

        Args:
            value: The value to be parsed.

        Returns:
            The parsed integer value, or None if the value is null.

        Raises:
            ValueError: If the value is not a valid integer (has decimal part).
        """
        if pd.isnull(value):
            return None
        cleaned_value = str(value).replace(',', '').replace('$', '').replace('%', '')
        float_value = float(cleaned_value)
        int_value = int(float_value)
        if float_value == int_value:
            return int_value
        raise ValueError(f'Invalid integer value: {value}')
