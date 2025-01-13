import itertools

from sqlalchemy import PoolProxiedConnection
import pandas as pd
from ..logger import Logger

logger = Logger().get_logger()


class Differentiator:
    def __init__(self, connection: PoolProxiedConnection, source_schema: str, target_schema: str, source_table: str,
                 target_table: str, similarity_threshold: float = .8):
        self.connection = connection
        self.source_schema = source_schema
        self.target_schema = target_schema
        self.source_table = source_table
        self.target_table = target_table
        self.similarity_threshold = similarity_threshold
        self.target_columns = []
        self.source_columns = []
        self.target_column_list = []
        self.source_column_list = []
        self.similar_columns = []
        self.unique_source_columns = []
        self.non_unique_target_columns = []
        self.same_name_columns = []
        self.unique_target_columns = []

    def find_table_similarities(self):
        self.get_column_data()
        self.get_column_dicts()
        self.compare_columns()
        self.log_results()
        return self.same_name_columns, self.similar_columns, self.unique_source_columns, self.unique_target_columns

    def get_column_dicts(self):
        self.target_column_list = self.get_column_dict_list(self.target_columns, self.target_schema,
                                                            self.target_table)
        self.source_column_list = self.get_column_dict_list(self.source_columns, self.source_schema,
                                                            self.source_table)

    def get_column_data(self):
        self.target_columns = self.get_column_name_list(self.target_schema, self.target_table)
        self.source_columns = self.get_column_name_list(self.source_schema, self.source_table)

    def log_results(self):
        self.same_name_columns.sort()
        self.unique_source_columns.sort()
        self.unique_target_columns.sort()
        message = (
            f'\n{"=" * 50}\ntable comparison between {self.source_table} and {self.target_table}\n'
            f'{"*" * 8} Columns with the same name:\n'
            f'{", ".join(self.same_name_columns)}\n'
            f'{"*" * 8} Columns with similar data:\n'
            f'{"\n".join([f'{column}' for column in self.similar_columns])}\n'
            f'{"*" * 8} Source Columns with unique data:\n'
            f'{", ".join(self.unique_source_columns)}\n'
            f'{"*" * 8} Target Columns with unique data:\n'
            f'{", ".join(self.unique_target_columns)}\n'
            f'\n{"=" * 50}\n'
        )
        logger.info(message)

    def compare_columns(self):
        for source_column in self.source_column_list:
            is_unique_source_column = True
            for target_column in self.target_column_list:
                if source_column['name'] == target_column['name']:
                    self.same_name_columns.append(source_column['name'])
                try:
                    similarity_source = source_column['data'].isin(target_column['data'])
                    similarity_target = target_column['data'].isin(source_column['data'])
                    similarity = max(similarity_source, similarity_target)
                    if similarity >= self.similarity_threshold:
                        is_unique_source_column = False
                        column_dict = {
                            "source_column": source_column['name'],
                            "target_column": target_column['name'],
                            "similarity": similarity
                        }
                        self.similar_columns.append(column_dict)
                        is_unique_source_column = False
                        self.non_unique_target_columns.append(target_column['name'])
                except (ValueError, TypeError) as e:
                    logger.debug(f'{source_column["name"]} and {target_column["name"]} are not comparable: {e}')
            if is_unique_source_column:
                self.unique_source_columns.append(source_column['name'])
        if self.non_unique_target_columns.__len__() < self.target_columns.__len__():
            self.unique_target_columns = [column for column in self.target_columns if
                                          column not in self.non_unique_target_columns]

    def get_column_dict_list(self, column_names, schema, table):
        column_list = []
        for column in column_names:
            query = f'select distinct ([{column}]) from {schema}.{table}'
            column_series = pd.read_sql(query, self.connection)[column]  # THIS IS FOR YOU JESSE
            column_series = column_series.dropna()
            column_dict = {"name": column, "data": column_series}
            column_list.append(column_dict)
        return column_list

    def get_column_name_list(self, schema, table):
        get_target_column_info_query = (
            f'select COLUMN_NAME '
            f'from INFORMATION_SCHEMA.columns '
            f'where table_schema = \'{schema}\' and table_name = \'{table}\'')
        target_column_info_df = pd.read_sql(get_target_column_info_query, self.connection)
        columns = target_column_info_df['COLUMN_NAME'].tolist()
        return columns

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
