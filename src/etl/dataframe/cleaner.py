import hashlib
import re
import numpy as np
import pandas as pd
from dateutil import parser
from rich import print


def clean_bool(dirty_bool):
    if dirty_bool is None:
        return
    dirty_bool = str(dirty_bool).lower()
    if dirty_bool in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif dirty_bool in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (dirty_bool,))


def clean_float(dirty_float):
    if dirty_float is None:
        return
    dirty_float = str(dirty_float)
    cleaned = dirty_float.replace(',', '').replace('$', '').replace('%', '')
    return float(cleaned)


def clean_date(dirty_date):
    if dirty_date is None or dirty_date is np.nan:
        return
    dirty_date = str(dirty_date).strip()
    return parser.parse(dirty_date)


def clean_int(dirty_int):
    if dirty_int is None or np.isnan(dirty_int):
        return
    if dirty_int == int(dirty_int):
        return int(dirty_int)
    raise ValueError


def hash_str(not_hashed):
    hashed = hashlib.sha1(str(not_hashed).encode())
    return hashed.hexdigest()


class Cleaner:

    @staticmethod
    def column_names_to_snake_case(df: pd.DataFrame):
        column_names = df.columns.to_list()
        clean_names = []
        for name in column_names:
            # Remove / Replace non-alpha-numeric characters
            name = (str(name)
                    .strip()
                    .replace('?', '')
                    .replace('(', '')
                    .replace(')', '')
                    .replace('\\', '')
                    .replace(',', '')
                    .replace('#', 'Num')
                    .replace('$', 'Dollars'))
            # add in underscore before a capital letter or number
            name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
            # replace other word-splitters with underscores
            name = (name
                    .replace('.', '_')
                    .replace(':', '_')
                    .replace(' ', '_')
                    .replace('-', '_')
                    .replace('___', '_')
                    .replace('__', '_')
                    .strip('_'))
            clean_names.append(name)
        df.columns = clean_names

    @staticmethod
    def clean_numbers(df: pd.DataFrame):
        for column, values in df.items():
            try:
                clean_floats = values.apply(clean_float)
                df[column] = clean_floats
                print(f'{column} has been cast to float')
                df[column] = clean_floats.apply(clean_int)
                print(f'{column} has been cast to int')
            except (ValueError, TypeError):
                continue
        return df

    @staticmethod
    def clean_dates(df: pd.DataFrame):
        for column, values in df.items():
            try:
                df[column] = values.apply(clean_date)
                print(f'{column} has been cast to datetime')
            except (parser.ParserError, OverflowError):
                continue
        return df

    @staticmethod
    def clean_bools(df: pd.DataFrame):
        for column, values in df.items():
            try:
                df[column] = values.apply(clean_bool)
                print(f'{column} has been cast to bool')
            except ValueError:
                continue
        return df

    @staticmethod
    def clean_all(df: pd.DataFrame):
        for column, values in df.items():
            try:
                clean_floats = values.apply(clean_float)
                df[column] = clean_floats
                print(f'{column} has been cast to float')
                df[column] = clean_floats.apply(clean_int)
                print(f'{column} has been cast to int')
            except (ValueError, TypeError):
                pass
            try:
                df[column] = values.apply(clean_date)
                print(f'{column} has been cast to datetime')
            except (parser.ParserError, OverflowError):
                pass
            try:
                df[column] = values.apply(clean_bool)
                print(f'{column} has been cast to bool')
            except ValueError:
                pass
        return df

    @staticmethod
    def generate_hash_column(df: pd.DataFrame, columns_to_hash: list[str], new_column_name: str):
        df[new_column_name] = ""
        for column in columns_to_hash:
            df[new_column_name] += df[column].apply(str)
        df[new_column_name] = df[new_column_name].apply(hash_str)
        return df

    @staticmethod
    def coalesce_columns(df: pd.DataFrame, columns_to_coalesce: list[str], new_column_name: str, drop: bool = False):
        df[new_column_name] = df[columns_to_coalesce[1]]
        for column in columns_to_coalesce:
            df[new_column_name] = df[new_column_name].combine_first(df[column])
        if drop:
            df = df.drop(columns=columns_to_coalesce)
        return df
