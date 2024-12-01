import pandas as pd
from dateutil import parser


class Parser:
    @staticmethod
    def parse_boolean(value):
        """
        Function to parse a boolean value from a given input.
        :param value: The value to be parsed as a boolean.
        :return: The parsed boolean value.
        The function takes a value as an input and attempts to parse it as a boolean. If the value is `None`, it returns `None`. If the value is a case-insensitive match for any of the truthy values ('y', 'yes', 't', 'true', 'on', '1'), it returns `True`. If the value is a case-insensitive match for any of the falsy values ('n', 'no', 'f', 'false', 'off', '0'), it returns `False`. Otherwise, it raises a `ValueError` with an error message indicating the invalid truth value.
        """
        if pd.isnull(value):
            return
        value = str(value).lower()
        truthy_values = ('y', 'yes', 't', 'true', 'on', '1')
        falsy_values = ('n', 'no', 'f', 'false', 'off', '0')
        if value in truthy_values:
            return True
        elif value in falsy_values:
            return False
        else:
            raise ValueError(f"Invalid truth value: {value}")

    @staticmethod
    def parse_float(value):
        """
        Function to parse a given value as a float.
        :param value: The value to parse as a float.
        :return: The parsed float value.
        """
        if pd.isnull(value):
            return
        cleaned_value = str(value).replace(',', '').replace('$', '').replace('%', '')
        return float(cleaned_value)

    @staticmethod
    def parse_date(value):
        """
        This function is used to parse a date value.
        :param value: The value to be parsed as a date.
        :return: The parsed date value.
        """
        if pd.isnull(value):
            return
        return parser.parse(str(value).strip())

    @staticmethod
    def parse_integer(value):
        """
        Parses an input value to an integer.
        :param value: The value to be parsed.
        :return: The parsed integer value.
        :raises ValueError: If the value is not a valid integer.
        """
        if pd.isnull(value):
            return
        if value == int(value):
            return int(value)
        raise ValueError(f'Invalid integer value: {value}')

