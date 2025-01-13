import itertools

from sqlalchemy import PoolProxiedConnection
import pandas as pd
from ..logger import Logger

logger = Logger().get_logger()


class Differentiator:
    @staticmethod
    def find_table_similarities(connection: PoolProxiedConnection, source_schema: str, source_table: str,
                                target_schema: str, target_table: str, similarity_threshold: float = .8):
        target_columns = Differentiator.get_column_name_list(connection, target_schema, target_table)
        source_columns = Differentiator.get_column_name_list(connection, source_schema, source_table)
        target_column_list = Differentiator.get_column_dict_list(connection, target_columns, target_schema,
                                                                 target_table)
        source_column_list = Differentiator.get_column_dict_list(connection, source_columns, source_schema,
                                                                 source_table)
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
                    similarity_source = source_column['data'].isin(target_column['data'])
                    similarity_target = target_column['data'].isin(source_column['data'])
                    similarity = max(similarity_source, similarity_target)
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
                except (ValueError, TypeError) as e:
                    logger.debug(f'{source_column["name"]} and {target_column["name"]} are not comparable: {e}')
            if is_unique_source_column:
                unique_source_columns.append(source_column['name'])
        unique_target_columns = []
        if non_unique_target_columns.__len__() < target_columns.__len__():
            unique_target_columns = [column for column in target_columns if column not in non_unique_target_columns]
        same_name_columns.sort()
        unique_source_columns.sort()
        unique_target_columns.sort()
        message = (
            f'\n{"=" * 50}\ntable comparison between {source_table} and {target_table}\n'
            f'{"*" * 8} Columns with the same name:\n'
            f'{", ".join(same_name_columns)}\n'
            f'{"*" * 8} Columns with similar data:\n'
            f'{"\n".join([f'{column}' for column in similar_columns])}\n'
            f'{"*" * 8} Source Columns with unique data:\n'
            f'{", ".join(unique_source_columns)}\n'
            f'{"*" * 8} Target Columns with unique data:\n'
            f'{", ".join(unique_target_columns)}\n'
            f'\n{"=" * 50}\n'
        )
        logger.info(message)
        return same_name_columns, similar_columns, unique_source_columns, unique_target_columns

    @staticmethod
    def get_column_dict_list(connection, column_names, schema, table):
        source_column_list = []
        for column in column_names:
            query = f'select distinct ([{column}]) from {schema}.{table}'
            column_series = pd.read_sql(query, connection).squeeze()
            column_series = column_series.dropna()
            column_dict = {"name": column, "data": column_series}
            source_column_list.append(column_dict)
        return source_column_list

    @staticmethod
    def get_column_name_list(connection, schema, table):
        get_target_column_info_query = (
            f'select COLUMN_NAME '
            f'from INFORMATION_SCHEMA.columns '
            f'where table_schema = \'{schema}\' and table_name = \'{table}\'')
        target_column_info_df = pd.read_sql(get_target_column_info_query, connection)
        target_columns = target_column_info_df['COLUMN_NAME'].tolist()
        return target_columns

    @staticmethod
    def find_table_similarities_in_schema(connection: PoolProxiedConnection, schema: str,
                                          similarity_threshold: float = .8):
        get_table_info_query = (
            f'select TABLE_NAME from INFORMATION_SCHEMA.TABLES '
            f'where TABLE_SCHEMA = \'{schema}\' and TABLE_TYPE = \'BASE TABLE\';'
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
        same_name_columns_list.sort()
        similar_columns_list.sort()
        unique_columns_list.sort()
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
