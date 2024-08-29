import math

import numpy as np
import pandas as pd
from ..dataframe.parser import Parser
from rich import print


class Creator:
    @staticmethod
    def make_mssql_table(df: pd.DataFrame, schema: str, table: str, primary_key: str = None, history: bool = False,
                         varchar_padding: int = 20, float_precision: int = 10, decimal_places: int = 2):
        df = df.replace({np.nan: None})
        location = f'{schema}.[{table}]'
        column_type_list = []
        for column, series in df.items():
            is_id = column.__str__() == primary_key
            column_string = None
            if series.dropna().empty:
                print(f"{column} is empty - setting to nvarchar(max)")
                column_string = f'[{column}] nvarchar(max)'
                column_type_list.append(column_string)
                continue
            try:
                series.apply(Parser.parse_date)
                column_string = f'[{column}] datetime2'
            except (ValueError, TypeError, OverflowError):
                pass
            try:
                series.apply(Parser.parse_float)
                left_digits = int(math.log10(series.max())) + 1
                if float_precision < left_digits + decimal_places:
                    float_precision = left_digits + decimal_places
                column_string = f'[{column}] decimal({float_precision}, {decimal_places})'
            except (ValueError, TypeError):
                pass
            try:
                series.apply(Parser.parse_integer)
                biggest_num = series.max()
                smallest_num = series.min()
                if smallest_num < -2147483648 or biggest_num > 2147483648:
                    column_string = f'[{column}] bigint'
                if smallest_num >= -2147483648 and biggest_num <= 2147483648:
                    column_string = f'[{column}] int'
                if smallest_num >= -32768 and biggest_num <= 32768:
                    column_string = f'[{column}] smallint'
                if smallest_num >= 0 and biggest_num <= 255:
                    column_string = f'[{column}] tinyint'
            except (ValueError, TypeError):
                pass
            try:
                series.apply(Parser.parse_boolean)
                column_string = f'[{column}] bit'
            except ValueError:
                pass
            if column_string is None:
                str_series = series.apply(str)
                largest_string_size = str_series.str.len().max()
                padded_length = int(largest_string_size + varchar_padding)
                if padded_length >= 4000:
                    column_string = f'[{column}] nvarchar(max)'
                else:
                    column_string = f'[{column}] nvarchar({padded_length})'
            if is_id:
                column_string += f' constraint pk_{table}_{column} primary key'
            column_type_list.append(column_string)

        column_type_string = ', '.join(column_type_list)
        create_query = f'create table {location} ({column_type_string});'
        if history:
            history_location = f'{schema}.[{table}_history]'
            history_insert = (
                f'{column_type_string}, '
                'system_record_start datetime2 generated always as row start '
                f'constraint df_{table}_system_record_start '
                'default sysutcdatetime() not null, '
                'system_record_end datetime2 generated always as row end '
                f'constraint df_{table}_system_record_end '
                'default sysutcdataetime() not null, '
                'period for system_time(system_record_start, system_record_end)'
            )
            create_query = (
                f'create table {location} ({history_insert}) with ('
                f'system_versioning = on (history_table = {history_location}));'
            )
        return create_query
