import itertools
import pandas as pd
from ..logger import Logger
from sqlalchemy import PoolProxiedConnection
from warnings import filterwarnings
from .utils import DatabaseUtils

filterwarnings("ignore", category=UserWarning, message='.*pandas only supports SQLAlchemy connectable.*')
filterwarnings("ignore", category=FutureWarning)
logger = Logger().get_logger()


class Differentiator:
    """
    Compares tables and schemas for column similarities, same names, and unique columns.
    """

    def __init__(self, connection: PoolProxiedConnection, similarity_threshold: float = 0.8):
        self.db_utils = DatabaseUtils(connection)
        self.similarity_threshold = similarity_threshold

    def find_table_similarities(self, source_schema: str, source_table: str, target_schema: str, target_table: str):
        source_columns = self.db_utils.get_column_names(source_schema, source_table)
        target_columns = self.db_utils.get_column_names(target_schema, target_table)

        source_data = [
            {"name": col, "data": self.db_utils.get_column_data(source_schema, source_table, col)}
            for col in source_columns
        ]
        target_data = [
            {"name": col, "data": self.db_utils.get_column_data(target_schema, target_table, col)}
            for col in target_columns
        ]

        return self._compare_tables(source_data, target_data, source_table, target_table)

    def _compare_tables(self, source_data: list, target_data: list, source_table: str, target_table: str):
        similar_columns, same_name_columns, unique_source_columns, unique_target_columns = [], [], [], []
        target_column_map = {col['name']: col['data'] for col in target_data}

        for source_col in source_data:
            source_name = source_col['name']
            is_unique_source = True

            for target_name, target_datum in target_column_map.items():
                if source_name == target_name:
                    same_name_columns.append(
                        {"source_table": source_table, "target_table": target_table, "column_name": source_name})
                try:
                    similarity_source = source_col['data'].isin(target_datum).mean()
                    similarity_target = target_datum.isin(source_col['data']).mean()
                    similarity = max(similarity_source, similarity_target)
                    if similarity >= self.similarity_threshold:
                        similar_columns.append({
                            "source_table": source_table,
                            "source_column": source_name,
                            "target_table": target_table,
                            "target_column": target_name,
                            "similarity": similarity
                        })
                        is_unique_source = False
                except (TypeError, ValueError) as e:
                    logger.debug(f'{source_name} and {target_name} are not comparable: {e}')

            if is_unique_source:
                unique_source_columns.append({"table_name": source_table, "column_name": source_name})

        unique_target_columns = [
            {"table_name": target_table, "column_name": col['name']}
            for col in target_data if col['name'] not in [s['name'] for s in source_data]
        ]

        return self._create_dataframes(same_name_columns, similar_columns, unique_source_columns, unique_target_columns)

    @staticmethod
    def _create_dataframes(same_name_columns, similar_columns, unique_source_columns, unique_target_columns):
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

        return pd.concat(same_name_list, ignore_index=True), pd.concat(similarity_list, ignore_index=True), pd.concat(
            unique_list, ignore_index=True)
