import itertools

from sqlalchemy import PoolProxiedConnection
import pandas as pd
from ..logger import Logger

logger = Logger().get_logger()


class Differentiator:
    @staticmethod
    def find_table_similarities(connection: PoolProxiedConnection, source_schema: str, target_schema: str,
                                source_table: str,
                                target_table: str, similarity_threshold: float = .8):
        get_target_column_info_query = (
            f'select COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION '
            f'from INFORMATION_SCHEMA.columns '
            f'where table_schema = \'{target_schema}\' and table_name = \'{target_table}\'')
        target_column_info_df = pd.read_sql(get_target_column_info_query, connection)
        get_source_column_info_query = (
            f'select COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION '
            f'from INFORMATION_SCHEMA.columns '
            f'where table_schema = \'{source_schema}\' and table_name = \'{source_table}\'')
        source_column_info_df = pd.read_sql(get_source_column_info_query, connection)
        source_columns = source_column_info_df['COLUMN_NAME'].tolist()
        target_columns = target_column_info_df['COLUMN_NAME'].tolist()
        source_column_list = []
        target_column_list = []
        logger.info("generating source column list")
        for column in source_columns:
            query = f'select distinct ([{column}]) from {source_schema}.{source_table}'
            column_series = pd.read_sql(query, connection).squeeze()
            column_series = column_series.dropna()
            column_dict = {"name": column, "data": column_series}
            source_column_list.append(column_dict)
        for column in target_columns:
            query = f'select distinct ([{column}]) from {target_schema}.{target_table}'
            column_series = pd.read_sql(query, connection).squeeze()
            column_series = column_series.dropna()
            column_dict = {"name": column, "data": column_series}
            target_column_list.append(column_dict)
        similar_columns = []
        unique_source_columns = []
        non_unique_target_columns = []
        same_name_columns = []
        for source_column in source_column_list:
            is_unique_source_column = True
            for target_column in target_column_list:
                if source_column['name'] == target_column['name']:
                    same_name_columns.append(source_column['name'])
                try:
                    similarity = (source_column['data'] == target_column['data']).mean()
                    if similarity >= similarity_threshold:
                        is_unique_source_column = False
                        column_dict = {
                            "source_column": source_column['name'],
                            "target_column": target_column['name'],
                            "similarity": similarity
                        }
                        similar_columns.append(column_dict)
                        is_unique_source_column = False
                        non_unique_target_columns.append(target_column['name'])
                except ValueError as e:
                    logger.debug(f'{source_column["name"]} and {target_column["name"]} are not comparable: {e}')
            if is_unique_source_column:
                unique_source_columns.append(source_column['name'])
        unique_target_columns = []
        if non_unique_target_columns.__len__() < target_columns.__len__():
            unique_target_columns = [column for column in target_columns if column not in non_unique_target_columns]
        message = (
            f'\n{"=" * 50}\ntable comparison between {source_table} and {target_table}\n'
            f'{"*" * 8} Columns with the same name:\n'
            f'{", ".join(same_name_columns)}\n'
            f'{"*" * 8} Columns with similar data:\n'
            f'{similar_columns}\n'
            f'{"*" * 8} Source Columns with unique data:\n'
            f'{", ".join(unique_source_columns)}\n'
            f'{"*" * 8} Target Columns with unique data:\n'
            f'{", ".join(unique_target_columns)}\n'
            f'\n{"=" * 50}\n'
        )
        logger.info(message)
        return same_name_columns, similar_columns, unique_source_columns, unique_target_columns

    @staticmethod
    def find_table_similarities_in_schema(connection: PoolProxiedConnection, schema: str,
                                          similarity_threshold: float = .8):
        get_table_info_query = (
            f'select TABLE_NAME from INFORMATION_SCHEMA.TABLES '
            f'where TABLE_SCHEMA = \'{schema}\''
        )
        table_info_df = pd.read_sql(get_table_info_query, connection)
        table_list = table_info_df['TABLE_NAME'].tolist()
        same_name_columns_list = []
        similar_columns_list = []
        unique_columns_list = []
        for table_set in itertools.combinations(table_list, 2):
            same_name_columns, similar_columns, unique_source_columns, unique_target_columns = Differentiator.find_table_similarities(
                connection, schema, schema, table_set[0], table_set[1], similarity_threshold)
            for column in same_name_columns:
                source_column = f'{table_set[0]}.{column}'
                target_column = f'{table_set[1]}.{column}'
                if source_column not in same_name_columns_list:
                    same_name_columns_list.append(source_column)
                if target_column not in same_name_columns_list:
                    same_name_columns_list.append(target_column)
            for column in similar_columns:
                source_column = f'{table_set[0]}.{column["source_column"]}'
                target_column = f'{table_set[1]}.{column["target_column"]}'
                if source_column not in similar_columns_list:
                    similar_columns_list.append(source_column)
                    if source_column in unique_columns_list:
                        unique_columns_list.remove(source_column)
                if target_column not in similar_columns_list:
                    similar_columns_list.append(target_column)
                    if target_column in unique_columns_list:
                        unique_columns_list.remove(target_column)
            for column in unique_source_columns:
                source_column = f'{table_set[0]}.{column}'
                if source_column not in unique_columns_list and source_column not in similar_columns_list:
                    unique_columns_list.append(source_column)
        message = (
            f'\n{"=" * 50}\n{schema} schema table differences\n'
            f'{"*" * 8} Columns with the same name:\n'
            f'{", ".join(same_name_columns_list)}\n'
            f'{"*" * 8} Columns with similar data:\n'
            f'{", ".join(similar_columns_list)}\n'
            f'{"*" * 8} Columns with unique data:\n'
            f'{", ".join(unique_columns_list)}\n'
            f'\n{"=" * 50}\n'
        )
        logger.info(message)
        return same_name_columns_list, similar_columns_list, unique_columns_list
