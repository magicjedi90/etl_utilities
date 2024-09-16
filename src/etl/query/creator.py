import pandas as pd
from ..dataframe.analyzer import Analyzer
from rich import print


class Creator:
    @staticmethod
    def create_mssql_table(df: pd.DataFrame, schema: str, table: str, primary_key: str = None,
                           unique_columns: list[str] = None, history: bool = False,
                           varchar_padding: int = 20, float_precision: int = 10, decimal_places: int = 2,
                           generate_id: bool = False):
        location = f'{schema}.[{table}]'
        column_metadata = Analyzer.generate_column_metadata(df, primary_key, unique_columns, decimal_places)
        column_type_list = []
        if generate_id:
            id_string = f'id int identity constraint pk_{table}_id primary key'
            column_type_list.append(id_string)
        for column in column_metadata:
            column_string = None
            column_name = column['column_name']
            if column['is_empty']:
                print(f"{column_name} is empty - setting to nvarchar(max)")
                column_string = f'[{column_name}] nvarchar(max)'
                column_type_list.append(column_string)
                continue
            if column['data_type'] == 'datetime':
                column_string = f'[{column_name}] datetime2'
            if column['data_type'] == 'float':
                if float_precision < column['float_precision']:
                    float_precision = column['float_precision']
                column_string = f'[{column_name}] decimal({float_precision}, {decimal_places})'
            if column['data_type'] == 'integer':
                if column['smallest_num'] < -2147483648 or column['biggest_num'] > 2147483648:
                    column_string = f'[{column_name}] bigint'
                if column['smallest_num'] >= -2147483648 and column['biggest_num'] <= 2147483648:
                    column_string = f'[{column_name}] int'
                if column['smallest_num'] >= -32768 and column['biggest_num'] <= 32768:
                    column_string = f'[{column_name}] smallint'
                if column['smallest_num'] >= 0 and column['biggest_num'] <= 255:
                    column_string = f'[{column_name}] tinyint'
            if column['data_type'] == 'boolean':
                column_string = f'[{column_name}] bit'
            if column['data_type'] == 'string':
                padded_length = int(column['max_str_size'] + varchar_padding)
                if padded_length >= 4000:
                    column_string = f'[{column_name}] nvarchar(max)'
                else:
                    column_string = f'[{column_name}] nvarchar({padded_length})'
            if column['is_id']:
                column_string += f' constraint pk_{table}_{column_name} primary key'
            if column['is_unique']:
                column_string += f' constraint ak_{table}_{column_name} unique'
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

    @staticmethod
    def create_mariadb_table(df: pd.DataFrame, schema: str, table: str, primary_key: str = None,
                           unique_columns: list[str] = None, history: bool = False,
                           varchar_padding: int = 20, float_precision: int = 10, decimal_places: int = 2,
                           generate_id: bool = False):
        location = f'{schema}.{table}'
        column_metadata = Analyzer.generate_column_metadata(df, primary_key, unique_columns, decimal_places)
        column_type_list = []
        if generate_id:
            id_string = f'id int auto_increment, constraint pk_{table}_id primary key (id)'
            column_type_list.append(id_string)
        for column in column_metadata:
            column_string = None
            column_name = column['column_name']
            if column['is_empty']:
                print(f"{column_name} is empty - skipping")
                continue
            if column['data_type'] == 'datetime':
                column_string = f'`{column_name}` datetime'
            if column['data_type'] == 'float':
                if float_precision < column['float_precision']:
                    float_precision = column['float_precision']
                column_string = f'`{column_name}` decimal({float_precision}, {decimal_places})'
            if column['data_type'] == 'integer':
                if column['smallest_num'] < -2147483648 or column['biggest_num'] > 2147483648:
                    column_string = f'`{column_name}` bigint'
                if column['smallest_num'] >= -2147483648 and column['biggest_num'] <= 2147483648:
                    column_string = f'`{column_name}` int'
                if column['smallest_num'] >= -32768 and column['biggest_num'] <= 32768:
                    column_string = f'`{column_name}` smallint'
                if column['smallest_num'] >= -128 and column['biggest_num'] <= 127:
                    column_string = f'`{column_name}` tinyint'
            if column['data_type'] == 'boolean':
                column_string = f'`{column_name}` bit'
            if column['data_type'] == 'string':
                padded_length = int(column['max_str_size'] + varchar_padding)
                if padded_length > 21844:
                    print(f"{column_name} has data too large for storing - skipping")
                    continue
                else:
                    column_string = f'`{column_name}` varchar({padded_length})'
            if column['is_id']:
                key_string = f'constraint pk_{table}_{column_name} primary key ({column_name})'
                column_type_list.append(key_string)
            if column['is_unique']:
                key_string = f'constraint ak_{table}_{column_name} unique ({column_name})'
                column_type_list.append(key_string)
            column_type_list.append(column_string)

        column_type_string = ', '.join(column_type_list)
        create_query = f'create table {location} ({column_type_string});'
        if history:
            create_query = (
                f'create table {location} ({column_type_string}) with ('
                f'system_versioning;'
            )
        return create_query