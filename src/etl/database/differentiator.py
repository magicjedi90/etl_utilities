import itertools
from sqlalchemy import PoolProxiedConnection
import pandas as pd
from ..logger import Logger
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning, message='.*pandas only supports SQLAlchemy connectable.*')
logger = Logger().get_logger()


class Differentiator:
    def __init__(self, connection: PoolProxiedConnection, similarity_threshold: float = 0.8):
        self.connection = connection
        self.similarity_threshold = similarity_threshold

    def find_table_similarities(self, source_schema: str, source_table: str, target_schema: str, target_table: str):
        source_columns, target_columns = self.__get_column_names(source_schema, source_table, target_schema,
                                                                 target_table)
        source_column_list, target_column_list = self.__get_column_dicts(source_columns, source_schema, source_table,
                                                                         target_columns, target_schema, target_table)
        similar_columns = []
        unique_source_columns = []
        non_unique_target_columns = []
        same_name_columns = []

        for source_column in source_column_list:
            is_unique_source_column = True
            for target_column in target_column_list:
                self.__get_same_name_columns(same_name_columns, source_column, source_table, target_column,
                                             target_table)
                try:
                    is_unique_source_column = self.__check_similarity_columns(is_unique_source_column,
                                                                              non_unique_target_columns,
                                                                              similar_columns,
                                                                              source_column, source_table,
                                                                              target_column,
                                                                              target_table)
                except (ValueError, TypeError) as e:
                    logger.debug(f'{source_column["name"]} and {target_column["name"]} are not comparable: {e}')
            self.__get_unique_source_columns(is_unique_source_column, source_column, source_table,
                                             unique_source_columns)
        unique_target_columns = self.__get_unique_target_columns(non_unique_target_columns, target_columns,
                                                                 target_table)
        same_name_df, similarity_df, unique_df = self.__create_dataframes(same_name_columns, similar_columns,
                                                                          unique_source_columns, unique_target_columns)
        return similarity_df, same_name_df, unique_df

    def find_table_similarities_in_schema(self, schema: str):
        table_list = self.__get_table_list_for_schema(schema)
        same_name_list, similarity_list, unique_list = self.__get_comparison_for_table_list(schema, table_list)
        schema_similarities_df = pd.concat(similarity_list)
        schema_same_name_df = pd.concat(same_name_list)
        schema_unique_df = pd.concat(unique_list)
        schema_similarities_df, schema_unique_df = self.__get_combined_schema_comparisons(schema_similarities_df,
                                                                                          schema_unique_df)
        return schema_same_name_df, schema_similarities_df, schema_unique_df

    @staticmethod
    def __get_unique_target_columns(non_unique_target_columns, target_columns, target_table):
        unique_target_columns = []
        if len(non_unique_target_columns) < len(target_columns):
            unique_target_columns = [{"table_name": target_table, "column_name": column} for column in target_columns if
                                     column not in non_unique_target_columns]
        return unique_target_columns

    @staticmethod
    def __create_dataframes(same_name_columns, similar_columns, unique_source_columns, unique_target_columns):
        similarity_df = pd.DataFrame(similar_columns)
        same_name_df = pd.DataFrame(same_name_columns)
        unique_source_df = pd.DataFrame(unique_source_columns)
        unique_target_df = pd.DataFrame(unique_target_columns)
        unique_df = pd.concat([unique_source_df, unique_target_df])
        return same_name_df, similarity_df, unique_df

    @staticmethod
    def __get_unique_source_columns(is_unique_source_column, source_column, source_table, unique_source_columns):
        if is_unique_source_column:
            column_dict = {
                "table_name": source_table,
                "column_name": source_column['name']
            }
            unique_source_columns.append(column_dict)

    def __check_similarity_columns(self, is_unique_source_column, non_unique_target_columns, similar_columns,
                                   source_column, source_table, target_column, target_table):
        similarity_source = source_column['data'].isin(target_column['data']).mean()
        similarity_target = target_column['data'].isin(source_column['data']).mean()
        similarity = max(similarity_source, similarity_target)
        if similarity >= self.similarity_threshold:
            is_unique_source_column = False
            column_dict = {
                "source_table": source_table,
                "source_column": source_column['name'],
                "target_table": target_table,
                "target_column": target_column['name'],
                "similarity": similarity
            }
            similar_columns.append(column_dict)
            non_unique_target_columns.append(target_column['name'])
        return is_unique_source_column

    @staticmethod
    def __get_same_name_columns(same_name_columns, source_column, source_table, target_column, target_table):
        if source_column['name'] == target_column['name']:
            column_dict = {
                "source_table": source_table,
                "target_table": target_table,
                "column_name": source_column['name']
            }
            same_name_columns.append(column_dict)

    def __get_column_dicts(self, source_columns, source_schema, source_table, target_columns, target_schema,
                           target_table):
        target_column_list = self.__get_column_dict_list(target_columns, target_schema, target_table)
        source_column_list = self.__get_column_dict_list(source_columns, source_schema, source_table)
        return source_column_list, target_column_list

    def __get_column_names(self, source_schema, source_table, target_schema, target_table):
        target_columns = self.__get_column_name_list(target_schema, target_table)
        source_columns = self.__get_column_name_list(source_schema, source_table)
        return source_columns, target_columns

    def __get_column_dict_list(self, column_names, schema, table):
        column_list = []
        for column in column_names:
            query = f'select distinct ([{column}]) from {schema}.{table}'
            column_series = pd.read_sql(query, self.connection)[column]
            column_series = column_series.dropna()
            column_dict = {"name": column, "data": column_series}
            column_list.append(column_dict)
        return column_list

    def __get_column_name_list(self, schema, table):
        query = (
            f'select COLUMN_NAME '
            f'from INFORMATION_SCHEMA.columns '
            f'where table_schema = \'{schema}\' and table_name = \'{table}\'')
        column_info_df = pd.read_sql(query, self.connection)
        return column_info_df['COLUMN_NAME'].tolist()

    @staticmethod
    def __get_combined_schema_comparisons(schema_similarities_df, schema_unique_df):
        if not schema_similarities_df.empty:
            schema_unique_df['combined'] = schema_unique_df['table_name'] + '.' + schema_unique_df['column_name']
            schema_similarities_df['combined_source'] = schema_similarities_df['source_table'] + '.' + \
                                                        schema_similarities_df['source_column']
            schema_similarities_df['combined_target'] = schema_similarities_df['target_table'] + '.' + \
                                                        schema_similarities_df['target_column']
            similar_columns_combined = pd.concat([
                schema_similarities_df['combined_source'],
                schema_similarities_df['combined_target']
            ])
            schema_unique_df = schema_unique_df[~schema_unique_df['combined'].isin(similar_columns_combined)]
            schema_unique_df = schema_unique_df.drop(columns=['combined'])
            schema_similarities_df = schema_similarities_df.drop(columns=['combined_source', 'combined_target'])
        return schema_similarities_df, schema_unique_df

    def __get_comparison_for_table_list(self, schema, table_list):
        same_name_list = []
        similarity_list = []
        unique_list = []
        for table_set in itertools.combinations(table_list, 2):
            logger.info(f'comparing {table_set[0]} and {table_set[1]}')
            similarity_df, same_name_df, unique_df = self.find_table_similarities(
                schema, table_set[0], schema, table_set[1])
            same_name_list.append(same_name_df)
            similarity_list.append(similarity_df)
            unique_list.append(unique_df)
        return same_name_list, similarity_list, unique_list

    def __get_table_list_for_schema(self, schema):
        query = (
            f'select TABLE_NAME from INFORMATION_SCHEMA.TABLES '
            f'where TABLE_SCHEMA = \'{schema}\' and TABLE_TYPE = \'BASE TABLE\';'
        )
        table_info_df = pd.read_sql(query, self.connection)
        table_list = table_info_df['TABLE_NAME'].tolist()
        return table_list
