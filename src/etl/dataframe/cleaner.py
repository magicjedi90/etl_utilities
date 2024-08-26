import hashlib
import re
import numpy as np
import pandas as pd
from dateutil import parser
from rich import print


def parse_boolean(value):
    """
    Function to parse a boolean value from a given input.
    :param value: The value to be parsed as a boolean.
    :return: The parsed boolean value.
    The function takes a value as an input and attempts to parse it as a boolean. If the value is `None`, it returns `None`. If the value is a case-insensitive match for any of the truthy values ('y', 'yes', 't', 'true', 'on', '1'), it returns `True`. If the value is a case-insensitive match for any of the falsy values ('n', 'no', 'f', 'false', 'off', '0'), it returns `False`. Otherwise, it raises a `ValueError` with an error message indicating the invalid truth value.
    """
    if value is None:
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


def parse_float(value):
    """
    Function to parse a given value as a float.
    :param value: The value to parse as a float.
    :return: The parsed float value.
    """
    if value is None:
        return
    cleaned_value = str(value).replace(',', '').replace('$', '').replace('%', '')
    return float(cleaned_value)


def parse_date(value):
    """
    This function is used to parse a date value.
    :param value: The value to be parsed as a date.
    :return: The parsed date value.
    """
    if value is None or value is np.nan:
        return
    return parser.parse(str(value).strip())


def parse_integer(value):
    """
    Parses an input value to an integer.
    :param value: The value to be parsed.
    :return: The parsed integer value.
    :raises ValueError: If the value is not a valid integer.
    """
    if value is None or np.isnan(value):
        return
    if value == int(value):
        return int(value)
    raise ValueError


def compute_hash(value):
    """
    Compute Hash
    Calculate the SHA-1 hash value of the given input value.
    :param value: The input value to be hashed.
    :return: The resulting hash value as a hexadecimal string.
    """
    return hashlib.sha1(str(value).encode()).hexdigest()


# Helper function to standardize column names
def standardize_column_name(name):
    """
    This function standardizes a given column name by removing special characters, replacing certain characters with new ones, and converting it to lowercase with underscores as separators.
    :param name: the column name to be standardized
    :return: the standardized column name
    """
    name = (str(name).strip()
            .replace('?', '').replace('(', '').replace(')', '')
            .replace('\\', '').replace(',', '').replace('#', 'Num')
            .replace('$', 'Dollars'))
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    return (name.replace('.', '_').replace(':', '_').replace(' ', '_')
            .replace('-', '_').replace('___', '_').replace('__', '_')
            .strip('_'))


class Cleaner:
    """
    This class provides static methods for data cleaning operations on a pandas DataFrame.
    The `column_names_to_snake_case` static method takes a DataFrame as input and converts the column names to snake case using the `standardize_column_name` function.
    The `clean_column` static method takes a DataFrame, a column name, and a clean function as input. It applies the clean function to the specified column in the DataFrame and returns the cleaned column. If any exceptions occur during the cleaning process, the method raises an exception.
    The `clean_numbers` static method takes a DataFrame as input and cleans all numeric columns by applying the `parse_float` function to each column. It also attempts to apply the `parse_integer` function to each column, but ignores any exceptions that occur.
    The `clean_dates` static method takes a DataFrame as input and cleans all date columns by applying the `parse_date` function to each column.
    The `clean_bools` static method takes a DataFrame as input and cleans all boolean columns by applying the `parse_boolean` function to each column.
    The `clean_all` static method takes a DataFrame as input and performs a comprehensive cleaning process by applying a set of cleaning functions, including `parse_boolean`, `parse_float`, `parse_integer`, and `parse_date`, to each column in the DataFrame. It handles exceptions that occur during the cleaning process and converts the DataFrame to the appropriate data types.
    The `generate_hash_column` static method takes a DataFrame, a list of column names to hash, and a new column name as input. It computes a hash value for each row based on the specified columns and adds a new column with the hash values to the DataFrame.
    The `coalesce_columns` static method takes a DataFrame, a list of columns to coalesce, a new column name, and an optional drop flag as input. It coalesces the specified columns by filling missing values with the previous non-null value in each row and creates a new column with the coalesced values. If the drop flag is True, the method drops the original columns from the DataFrame.
    """

    @staticmethod
    def column_names_to_snake_case(df: pd.DataFrame):
        df.columns = [standardize_column_name(name) for name in df.columns]

    @staticmethod
    def clean_series(series: pd.Series, clean_function):
        try:
            return series.apply(clean_function)
        except (ValueError, TypeError, parser.ParserError, OverflowError):
            raise

    @staticmethod
    def clean_numbers(df: pd.DataFrame):
        for column, series in df.items():
            df[column] = Cleaner.clean_series(series, parse_float)
            try:
                df[column] = Cleaner.clean_series(df[column], parse_integer)
            except ValueError:
                pass
        return df

    @staticmethod
    def clean_dates(df: pd.DataFrame):
        for column, series in df.items():
            df[column] = Cleaner.clean_series(series, parse_date)
        return df

    @staticmethod
    def clean_bools(df: pd.DataFrame):
        for column, series in df.items():
            df[column] = Cleaner.clean_series(series, parse_boolean)
        return df

    @staticmethod
    def clean_all(df: pd.DataFrame):
        try_functions = [parse_float, parse_integer, parse_boolean, parse_date]
        for column, series in df.items():
            is_column_clean = False
            for func in try_functions:
                # if the column has already been cleaned, make sure that it's either a boolean or the next cleaning call is parse_integer
                if is_column_clean and func == parse_date:
                    continue
                try:
                    series = Cleaner.clean_series(series, func)
                    df[column] = series
                    is_column_clean = True
                    print(f'{column} was cleaned with {func.__name__}')
                except (ValueError, TypeError, parser.ParserError, OverflowError):
                    pass
        df = df.convert_dtypes()
        return df

    @staticmethod
    def generate_hash_column(df: pd.DataFrame, columns_to_hash, new_column_name):
        df[new_column_name] = df[columns_to_hash].astype(str).sum(axis=1).apply(compute_hash)
        return df

    @staticmethod
    def coalesce_columns(df: pd.DataFrame, columns_to_coalesce, new_column_name, drop=False):
        df[new_column_name] = df[columns_to_coalesce].bfill(axis=1).iloc[:, 0]
        if drop:
            df = df.drop(columns=columns_to_coalesce)
        return df