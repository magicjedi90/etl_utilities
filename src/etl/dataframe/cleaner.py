import hashlib
import re
import pandas as pd
from rich import print
from dateutil import parser
from ..dataframe.parser import Parser

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
            .replace('\\', '').replace(',', '').replace('/','')
            .replace('\'','').replace('#', 'Num').replace('$', 'Dollars'))
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    return (name.replace('.', '_').replace(':', '_').replace(' ', '_')
            .replace('-', '_').replace('___', '_').replace('__', '_')
            .strip('_'))


class Cleaner:
    """
    This class provides static methods for data cleaning operations on a pandas DataFrame.
    The `column_names_to_snake_case` static method takes a DataFrame as input and converts the column names to snake case using the `standardize_column_name` function.
    The `clean_series` static method takes a series, and a clean function as input. It applies the clean function to the specified series and returns the cleaned series. If any exceptions occur during the cleaning process, the method raises an exception.
    The `clean_numbers` static method takes a DataFrame as input and cleans all numeric columns by applying the `parse_float` function to each column. It also attempts to apply the `parse_integer` function to each column, but ignores any exceptions that occur.
    The `clean_dates` static method takes a DataFrame as input and cleans all date columns by applying the `parse_date` function to each column.
    The `clean_bools` static method takes a DataFrame as input and cleans all boolean columns by applying the `parse_boolean` function to each column.
    The `clean_all` static method takes a DataFrame as input and performs a comprehensive cleaning process by applying a set of cleaning functions, including `parse_boolean`, `parse_float`, `parse_integer`, and `parse_date`, to each column in the DataFrame. It handles exceptions that occur during the cleaning process and converts the DataFrame to the appropriate data types.
    The `generate_hash_column` static method takes a DataFrame, a list of column names to hash, and a new column name as input. It computes a hash value for each row based on the specified columns and adds a new column with the hash values to the DataFrame.
    The `coalesce_columns` static method takes a DataFrame, a list of columns to coalesce, a target column name, and an optional drop flag as input. It coalesces the specified columns by filling missing values with the previous non-null value in each row and creates or consolidates the target column with the coalesced values. If the drop flag is True, the method drops the original columns from the DataFrame.
    """

    @staticmethod
    def column_names_to_snake_case(df: pd.DataFrame):
        df.columns = [standardize_column_name(name) for name in df.columns]

    @staticmethod
    def column_names_to_pascal_case(df: pd.DataFrame):
        df.columns = ["".join(standardize_column_name(name).title().split('_')) for name in df.columns]

    @staticmethod
    def clean_series(series: pd.Series, clean_function):
        try:
            cleaned_series = series.apply(clean_function)
            series_dtype = clean_function.__annotations__.get('return', None)
            if series_dtype:
                cleaned_series = cleaned_series.astype(series_dtype)
            return cleaned_series
        except (ValueError, TypeError, parser.ParserError, OverflowError):
            raise

    @staticmethod
    def clean_numbers(df: pd.DataFrame):
        for column, series in df.items():
            df[column] = Cleaner.clean_series(series, Parser.parse_float)
            try:
                df[column] = Cleaner.clean_series(df[column], Parser.parse_integer)
            except ValueError:
                pass
        return df

    @staticmethod
    def clean_dates(df: pd.DataFrame):
        for column, series in df.items():
            df[column] = Cleaner.clean_series(series, Parser.parse_date)
        return df

    @staticmethod
    def clean_bools(df: pd.DataFrame):
        for column, series in df.items():
            df[column] = Cleaner.clean_series(series, Parser.parse_boolean)
        return df

    @staticmethod
    def clean_all_types(df: pd.DataFrame):
        try_functions = [Parser.parse_float, Parser.parse_integer, Parser.parse_boolean, Parser.parse_date]
        for column, series in df.items():
            if series.dropna().empty:
                print(f'{column} is empty skipping cleaning')
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
                    print(f'{column} was cleaned with {func.__name__}')
                except (ValueError, TypeError, parser.ParserError, OverflowError) as error:
                    # print(f'{column} failed cleaning with {func.__name__}: {error}')
                    pass
        df = df.convert_dtypes()
        return df

    @staticmethod
    def clean_df(df: pd.DataFrame):
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
        return Cleaner.clean_all_types(df)

    @staticmethod
    def generate_hash_column(df: pd.DataFrame, columns_to_hash, new_column_name):
        df[new_column_name] = df[columns_to_hash].astype(str).sum(axis=1).apply(compute_hash)
        return df

    @staticmethod
    def coalesce_columns(df: pd.DataFrame, columns_to_coalesce, target_column, drop=False):
        df[target_column] = df[columns_to_coalesce].bfill(axis=1).iloc[:, 0]
        if drop:
            if target_column in columns_to_coalesce:
                columns_to_coalesce.remove(target_column)
            df = df.drop(columns=columns_to_coalesce)
        return df