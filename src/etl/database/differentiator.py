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
                    column_dict = {
                        "source_table": source_table,
                        "target_table": target_table,
                        "column": source_column['name']
                    }
                    same_name_columns.append(column_dict)
                try:
                    similarity_source = source_column['data'].isin(target_column['data'])
                    similarity_target = target_column['data'].isin(source_column['data'])
                    similarity = max(similarity_source, similarity_target)
                    if similarity >= similarity_threshold:
                        is_unique_source_column = False
                        column_dict = {
                            "source_table": source_table,
                            "source_column": source_column['name'],
                            "target_table": target_table,
                            "target_column": target_column['name'],
                            "similarity": similarity
                        }
                        similar_columns.append(column_dict)
                        is_unique_source_column = False
                        non_unique_target_columns.append(target_column['name'])
                except (ValueError, TypeError) as e:
                    logger.debug(f'{source_column["name"]} and {target_column["name"]} are not comparable: {e}')
            if is_unique_source_column:
                column_dict = {
                    "table": source_table,
                    "column": source_column['name']
                }
                unique_source_columns.append(column_dict)
        unique_target_columns = []
        if non_unique_target_columns.__len__() < target_columns.__len__():
            unique_target_columns = [{"table": target_table, "column": column} for column in target_columns if
                                     column not in non_unique_target_columns]

        similarity_df = pd.DataFrame(source_column_list)
        same_name_df = pd.DataFrame(same_name_columns)
        unique_source_df = pd.DataFrame(unique_source_columns)
        unique_target_df = pd.DataFrame(unique_target_columns)
        unique_df = pd.concat([unique_source_df, unique_target_df])
        return similarity_df, same_name_df, unique_df

    @staticmethod
    def get_column_dict_list(connection, column_names, schema, table):
        source_column_list = []
        for column in column_names:
            query = f'select distinct ([{column}]) from {schema}.{table}'
            column_series = pd.read_sql(query, connection)[column]
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
        same_name_list = []
        similarity_list = []
        unique_list = []
        for table_set in itertools.combinations(table_list, 2):
            similarity_df, same_name_df, unique_df = Differentiator.find_table_similarities(
                connection, schema, schema, table_set[0], table_set[1], similarity_threshold)
            same_name_list.append(same_name_df)
            similarity_list.append(similarity_df)
            unique_list.append(unique_df)
        schema_similarities_df = pd.concat(similarity_list)
        schema_same_name_df = pd.concat(same_name_list)
        schema_unique_df = pd.concat(unique_list)

        # Combine table and column in both DataFrames for comparison
        schema_unique_df['combined'] = schema_unique_df['table'] + '.' + schema_unique_df['column']
        schema_similarities_df['combined_source'] = schema_similarities_df['source_table'] + '.' + \
                                                    schema_similarities_df[
                                                        'source_column']
        schema_similarities_df['combined_target'] = schema_similarities_df['target_table'] + '.' + \
                                                    schema_similarities_df[
                                                        'target_column']

        # Combine all "similar" columns into one series for exclusion
        similar_columns_combined = pd.concat([
            schema_similarities_df['combined_source'],
            schema_similarities_df['combined_target']
        ])

        # Filter out rows from schema_unique_df that match any in schema_similarities
        filtered_schema_unique_df = schema_unique_df[~schema_unique_df['combined'].isin(similar_columns_combined)]

        # drop the combined column not needed anymore
        filtered_schema_unique_df = filtered_schema_unique_df.drop(columns=['combined'])
        schema_similarities_df = schema_similarities_df.drop(columns=['combined_source', 'combined_target'])

        return schema_same_name_df, schema_similarities_df, filtered_schema_unique_df
