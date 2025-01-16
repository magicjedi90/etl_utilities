import itertools
import pandas as pd
from ..logger import Logger
from sqlalchemy import PoolProxiedConnection
from warnings import filterwarnings
from .utils import DatabaseUtils

filterwarnings("ignore", category=UserWarning, message='.*pandas only supports SQLAlchemy connectable.*')
logger = Logger().get_logger()


class Differentiator:
    """
    Compares tables and schemas for column similarities, same names, and unique columns.
    """

    def __init__(self, connection: PoolProxiedConnection, similarity_threshold: float = 0.8):
        self.db_utils = DatabaseUtils(connection)
        self.similarity_threshold = similarity_threshold

    def find_table_similarities(self, source_schema: str, source_table: str, target_schema: str, target_table: str):
        source_columns = self.db_utils.get_column_names_and_types(source_schema, source_table)
        target_columns = self.db_utils.get_column_names_and_types(target_schema, target_table)
        source_data, target_data = [], []
        for row in source_columns.itertuples(index=False):
            source_data.append({"name": row[0], "type": row[1],
                                "data": self.db_utils.get_column_data(source_schema, source_table, row[0])})
        for row in target_columns.itertuples(index=False):
            target_data.append({"name": row[0], "type": row[1],
                                "data": self.db_utils.get_column_data(target_schema, target_table, row[0])})

        similar_columns, same_name_columns, unique_source_columns, non_unique_target_columns, unique_target_columns = [], [], [], [], []
        # target_column_map = {col['name']: col['data'] for col in target_data}

        for source_col in source_data:
            source_name = source_col['name']
            source_type = source_col['type']
            source_datum = source_col['data']
            is_unique_source = True

            for target_col in target_data:
                target_name = target_col['name']
                target_type = target_col['type']
                target_datum = target_col['data']
                if source_name == target_name:
                    same_name_columns.append(
                        {"source_table": source_table, "target_table": target_table, "column_name": source_name})
                if source_type != target_type:
                    continue
                try:
                    similarity_source = source_datum.isin(target_datum).mean()
                    similarity_target = target_datum.isin(source_datum).mean()
                    similarity = max(similarity_source, similarity_target)

                    if similarity >= self.similarity_threshold:
                        similar_columns.append({
                            "source_table": source_table,
                            "source_column": source_name,
                            "target_table": target_table,
                            "target_column": target_name,
                            "data_type": source_type,
                            "similarity": similarity
                        })
                        is_unique_source = False
                        non_unique_target_columns.append(target_name)
                except (TypeError, ValueError) as e:
                    logger.debug(f'{source_name} and {target_name} are not comparable: {e}')

            if is_unique_source:
                unique_source_columns.append({"table_name": source_table, "column_name": source_name})

        unique_target_columns = [
            {"table_name": target_table, "column_name": col['name']}
            for col in target_data if col['name'] not in non_unique_target_columns
        ]
        same_name_df = pd.DataFrame(same_name_columns)
        similarity_df = pd.DataFrame(similar_columns)
        unique_df = pd.concat([pd.DataFrame(unique_source_columns), pd.DataFrame(unique_target_columns)],
                              ignore_index=True)
        return similarity_df, same_name_df, unique_df

    def find_schema_similarities(self, schema: str):
        table_list = self.db_utils.get_table_list(schema)
        similarity_list, same_name_list, unique_list = [], [], []

        for source_table, target_table in itertools.combinations(table_list, 2):
            logger.info(f"Comparing {source_table} and {target_table}")
            similarity_df, same_name_df, unique_df = self.find_table_similarities(schema, source_table, schema,
                                                                                  target_table)
            similarity_list.append(similarity_df)
            same_name_list.append(same_name_df)
            unique_list.append(unique_df)
        schema_same_name, schema_similarity, schema_unique = None, None, None
        if len(same_name_list) > 0:
            schema_same_name = pd.concat(same_name_list, ignore_index=True)
        if len(similarity_list) > 0:
            schema_similarity = pd.concat(similarity_list, ignore_index=True)
        if len(unique_list) > 0:
            schema_unique = pd.concat(unique_list, ignore_index=True)

        # Combine table and column in both DataFrames for comparison
        if schema_unique is not None and schema_similarity is not None and not schema_unique.empty and not schema_similarity.empty:
            schema_unique['combined'] = schema_unique['table_name'] + '.' + schema_unique['column_name']
            schema_similarity['combined_source'] = schema_similarity['source_table'] + '.' + \
                                                   schema_similarity[
                                                       'source_column']
            schema_similarity['combined_target'] = schema_similarity['target_table'] + '.' + \
                                                   schema_similarity[
                                                       'target_column']

            # Combine all "similar" columns into one series for exclusion
            similar_columns_combined = pd.concat([
                schema_similarity['combined_source'],
                schema_similarity['combined_target']
            ])

            # Filter out rows from schema_unique that match any in schema_similarity
            schema_unique = schema_unique[~schema_unique['combined'].isin(similar_columns_combined)]

            # drop the combined column, not needed anymore
            schema_unique = schema_unique.drop(columns=['combined'])
            schema_similarity = schema_similarity.drop(columns=['combined_source', 'combined_target'])
        return schema_same_name, schema_similarity, schema_unique
