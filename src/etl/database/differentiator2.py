import itertools
import pandas as pd
from typing import List, Tuple, Set


class TableDataExtractor:
    """
    Responsible for fetching table and column data from the database.
    """

    def __init__(self, connection):
        self.connection = connection

    def get_table_names(self, schema: str) -> List[str]:
        """
        Fetch all table names in a schema.
        """
        query = (
            f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE = 'BASE TABLE';"
        )
        result_df = pd.read_sql(query, self.connection)
        return result_df["TABLE_NAME"].tolist()

    def get_column_names(self, schema: str, table: str) -> List[str]:
        """
        Fetch all column names for a given table.
        """
        query = (
            f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}';"
        )
        result_df = pd.read_sql(query, self.connection)
        return result_df["COLUMN_NAME"].tolist()

    def get_column_data(self, schema: str, table: str, column: str) -> pd.Series:
        """
        Fetch distinct, non-null data for a specific column in a table.
        """
        query = f"SELECT DISTINCT [{column}] FROM {schema}.{table}"
        column_data = pd.read_sql(query, self.connection)[column].dropna()
        return column_data


class Differentiator2:
    def __init__(
            self,
            connection,
            source_schema: str,
            target_schema: str,
            similarity_threshold: float = 0.8,
    ):
        self.connection = connection
        self.source_schema = source_schema
        self.target_schema = target_schema
        self.similarity_threshold = similarity_threshold
        self.data_extractor = TableDataExtractor(connection)

    def compare_columns(
            self, source_data: List[dict], target_data: List[dict]
    ) -> Tuple[List[str], List[dict], List[str], List[str]]:
        """
        Compare columns between two tables and identify similarities, differences, and unique columns.
        """
        same_name_columns = []
        similar_columns = []
        unique_source_columns = []

        target_column_names = {col["name"]: col for col in target_data}
        for source_column in source_data:
            if source_column["name"] in target_column_names:
                same_name_columns.append(source_column["name"])
            else:
                unique_source_columns.append(source_column["name"])

            for target_column in target_data:
                try:
                    similarity_source = source_column["data"].isin(target_column["data"])
                    similarity_target = target_column["data"].isin(source_column["data"])
                    similarity = max(similarity_source, similarity_target)

                    if similarity >= self.similarity_threshold:
                        similar_columns.append(
                            {
                                "source_column": source_column["name"],
                                "target_column": target_column["name"],
                                "similarity": similarity,
                            }
                        )
                except (ValueError, TypeError):
                    pass

        unique_target_columns = [col["name"] for col in target_data if col["name"] not in same_name_columns]

        return same_name_columns, similar_columns, unique_source_columns, unique_target_columns

    def find_table_similarities(
            self, source_table: str, target_table: str
    ) -> Tuple[List[str], List[dict], List[str], List[str]]:
        """
        Find similarities between two tables.
        """
        source_columns = self.data_extractor.get_column_names(self.source_schema, source_table)
        target_columns = self.data_extractor.get_column_names(self.target_schema, target_table)

        source_data = [
            {"name": col, "data": self.data_extractor.get_column_data(self.source_schema, source_table, col)}
            for col in source_columns
        ]
        target_data = [
            {"name": col, "data": self.data_extractor.get_column_data(self.target_schema, target_table, col)}
            for col in target_columns
        ]

        return self.compare_columns(source_data, target_data)

    def compare_schema_tables(self, schema: str):
        """
        Compare all tables in a given schema.
        """
        table_list = self.data_extractor.get_table_names(schema)
        results = {
            "same_name_columns": set(),
            "similar_columns": set(),
            "unique_columns": set(),
        }

        for source_table, target_table in itertools.combinations(table_list, 2):
            same_name_columns, similar_columns, unique_source_columns, unique_target_columns = self.find_table_similarities(
                source_table, target_table
            )

            results["same_name_columns"].update(
                f"{table}.{col}" for table in (source_table, target_table) for col in same_name_columns
            )
            results["similar_columns"].update(
                f"{table}.{col['source_column']}" for table in (source_table, target_table) for col in similar_columns
            )
            results["unique_columns"].update(
                f"{source_table}.{col}" for col in unique_source_columns
            )
            results["unique_columns"].update(
                f"{target_table}.{col}" for col in unique_target_columns
            )

        for key in results:
            results[key] = sorted(results[key])

        return results
