import itertools
import pandas as pd
from ..logger import Logger
from sqlalchemy import PoolProxiedConnection
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning, message='.*pandas only supports SQLAlchemy connectable.*')
logger = Logger().get_logger()

class DatabaseUtils:
    """
    Utility class for database operations, such as fetching table and column data.
    """
    def __init__(self, connection: PoolProxiedConnection):
        self.connection = connection

    def get_table_list(self, schema: str) -> list:
        query = (
            f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE = 'BASE TABLE';"
        )
        return pd.read_sql(query, self.connection)['TABLE_NAME'].tolist()

    def get_column_names(self, schema: str, table: str) -> list:
        query = (
            f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}';"
        )
        return pd.read_sql(query, self.connection)['COLUMN_NAME'].tolist()

    def get_column_data(self, schema: str, table: str, column: str) -> pd.Series:
        query = f"SELECT DISTINCT [{column}] FROM {schema}.{table}"
        return pd.read_sql(query, self.connection)[column].dropna()


class ColumnComparator:
    """
    Handles column comparison logic between two tables.
    """
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def compare_columns(self, source_data: pd.Series, target_data: pd.Series) -> float:
        similarity_source = source_data.isin(target_data).mean()
        similarity_target = target_data.isin(source_data).mean()
        return max(similarity_source, similarity_target)


class Differentiator:
    """
    Compares tables and schemas for column similarities, same names, and unique columns.
    """
    def __init__(self, connection: PoolProxiedConnection, similarity_threshold: float = 0.8):
        self.db_utils = DatabaseUtils(connection)
        self.column_comparator = ColumnComparator(similarity_threshold)

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

            for target_name, target_data in target_column_map.items():
                if source_name == target_name:
                    same_name_columns.append({"source_table": source_table, "target_table": target_table, "column_name": source_name})

                similarity = self.column_comparator.compare_columns(source_col['data'], target_data)
                if similarity >= self.column_comparator.similarity_threshold:
                    similar_columns.append({
                        "source_table": source_table,
                        "source_column": source_name,
                        "target_table": target_table,
                        "target_column": target_name,
                        "similarity": similarity
                    })
                    is_unique_source = False

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
        unique_df = pd.concat([pd.DataFrame(unique_source_columns), pd.DataFrame(unique_target_columns)], ignore_index=True)
        return similarity_df, same_name_df, unique_df

    def find_schema_similarities(self, schema: str):
        table_list = self.db_utils.get_table_list(schema)
        similarity_list, same_name_list, unique_list = [], [], []

        for source_table, target_table in itertools.combinations(table_list, 2):
            logger.info(f"Comparing {source_table} and {target_table}")
            similarity_df, same_name_df, unique_df = self.find_table_similarities(schema, source_table, schema, target_table)
            similarity_list.append(similarity_df)
            same_name_list.append(same_name_df)
            unique_list.append(unique_df)

        return {
            "similarities": pd.concat(similarity_list, ignore_index=True),
            "same_names": pd.concat(same_name_list, ignore_index=True),
            "unique_columns": pd.concat(unique_list, ignore_index=True)
        }
