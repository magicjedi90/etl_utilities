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
        """
        Compares columns between two tables and identifies similarities, same names, and unique columns.

        Args:
            source_schema (str): The schema of the source table.
            source_table (str): The name of the source table.
            target_schema (str): The schema of the target table.
            target_table (str): The name of the target table.

        Returns:
            tuple: A tuple containing three pandas DataFrames:
                - similarity_df: DataFrame with columns that are similar between the source and target tables.
                - same_name_df: DataFrame with columns that have the same name in both tables.
                - unique_df: DataFrame with columns that are unique to either the source or target table.
        """
        source_columns, target_columns = self.get_column_names(source_schema, source_table, target_schema, target_table)
        source_column_list, target_column_list = self.get_column_dicts(source_columns, source_schema, source_table,
                                                                       target_columns, target_schema, target_table)
        similar_columns = []
        unique_source_columns = []
        non_unique_target_columns = []
        same_name_columns = []

        for source_column in source_column_list:
            is_unique_source_column = True
            for target_column in target_column_list:
                self.get_same_name_columns(same_name_columns, source_column, source_table, target_column, target_table)
                try:
                    is_unique_source_column = self.check_similarity_columns(is_unique_source_column,
                                                                            non_unique_target_columns, similar_columns,
                                                                            source_column, source_table, target_column,
                                                                            target_table)
                except (ValueError, TypeError) as e:
                    logger.debug(f'{source_column["name"]} and {target_column["name"]} are not comparable: {e}')
            self.get_unique_source_columns(is_unique_source_column, source_column, source_table, unique_source_columns)
        unique_target_columns = self.get_unique_target_columns(non_unique_target_columns, target_columns, target_table)
        same_name_df, similarity_df, unique_df = self.create_dataframes(same_name_columns, similar_columns,
                                                                        unique_source_columns, unique_target_columns)
        return similarity_df, same_name_df, unique_df

    @staticmethod
    def get_unique_target_columns(non_unique_target_columns, target_columns, target_table):
        """
        Identifies columns that are unique to the target table.

        Args:
            non_unique_target_columns (list): List of column names that are not unique to the target table.
            target_columns (list): List of all column names in the target table.
            target_table (str): The name of the target table.

        Returns:
            list: A list of dictionaries, each containing the table name and column name of unique columns in the target table.
        """
        unique_target_columns = []
        if len(non_unique_target_columns) < len(target_columns):
            unique_target_columns = [{"table_name": target_table, "column_name": column} for column in target_columns if
                                     column not in non_unique_target_columns]
        return unique_target_columns

    @staticmethod
    def create_dataframes(same_name_columns, similar_columns, unique_source_columns, unique_target_columns):
        """
        Creates pandas DataFrames from lists of column information.

        Args:
            same_name_columns (list): List of columns that have the same name in both tables.
            similar_columns (list): List of columns that are similar between the source and target tables.
            unique_source_columns (list): List of columns that are unique to the source table.
            unique_target_columns (list): List of columns that are unique to the target table.

        Returns:
            tuple: A tuple containing three pandas DataFrames:
                - same_name_df: DataFrame with columns that have the same name in both tables.
                - similarity_df: DataFrame with columns that are similar between the source and target tables.
                - unique_df: DataFrame with columns that are unique to either the source or target table.
        """
        similarity_df = pd.DataFrame(similar_columns)
        same_name_df = pd.DataFrame(same_name_columns)
        unique_source_df = pd.DataFrame(unique_source_columns)
        unique_target_df = pd.DataFrame(unique_target_columns)
        unique_df = pd.concat([unique_source_df, unique_target_df])
        return same_name_df, similarity_df, unique_df

    @staticmethod
    def get_unique_source_columns(is_unique_source_column, source_column, source_table, unique_source_columns):
        """
        Appends columns that are unique to the source table to the unique_source_columns list.

        Args:
            is_unique_source_column (bool): Flag indicating if the source column is unique.
            source_column (dict): Dictionary containing the source column information.
            source_table (str): The name of the source table.
            unique_source_columns (list): List to append the unique source columns to.

        Returns:
            None
        """
        if is_unique_source_column:
            column_dict = {
                "table_name": source_table,
                "column_name": source_column['name']
            }
            unique_source_columns.append(column_dict)

    def check_similarity_columns(self, is_unique_source_column, non_unique_target_columns, similar_columns,
                                 source_column, source_table, target_column, target_table):
        """
        Checks the similarity between columns from the source and target tables and updates the lists of similar columns,
        non-unique target columns, and the flag indicating if the source column is unique.

        Args:
            is_unique_source_column (bool): Flag indicating if the source column is unique.
            non_unique_target_columns (list): List of column names that are not unique to the target table.
            similar_columns (list): List to append the similar columns to.
            source_column (dict): Dictionary containing the source column information.
            source_table (str): The name of the source table.
            target_column (dict): Dictionary containing the target column information.
            target_table (str): The name of the target table.

        Returns:
            bool: Updated flag indicating if the source column is unique.
        """
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
    def get_same_name_columns(same_name_columns, source_column, source_table, target_column, target_table):
        """
        Appends columns that have the same name in both the source and target tables to the same_name_columns list.

        Args:
            same_name_columns (list): List to append the columns with the same name to.
            source_column (dict): Dictionary containing the source column information.
            source_table (str): The name of the source table.
            target_column (dict): Dictionary containing the target column information.
            target_table (str): The name of the target table.

        Returns:
            None
        """
        if source_column['name'] == target_column['name']:
            column_dict = {
                "source_table": source_table,
                "target_table": target_table,
                "column_name": source_column['name']
            }
            same_name_columns.append(column_dict)

    def get_column_dicts(self, source_columns, source_schema, source_table, target_columns, target_schema,
                         target_table):
        """
        Converts column names from source and target tables into lists of dictionaries containing column information.

        Args:
            source_columns (list): List of column names in the source table.
            source_schema (str): The schema of the source table.
            source_table (str): The name of the source table.
            target_columns (list): List of column names in the target table.
            target_schema (str): The schema of the target table.
            target_table (str): The name of the target table.

        Returns:
            tuple: A tuple containing two lists of dictionaries:
                - source_column_list: List of dictionaries with column information for the source table.
                - target_column_list: List of dictionaries with column information for the target table.
        """
        target_column_list = self.get_column_dict_list(target_columns, target_schema, target_table)
        source_column_list = self.get_column_dict_list(source_columns, source_schema, source_table)
        return source_column_list, target_column_list

    def get_column_names(self, source_schema, source_table, target_schema, target_table):
        """
        Retrieves the column names from the source and target tables.

        Args:
            source_schema (str): The schema of the source table.
            source_table (str): The name of the source table.
            target_schema (str): The schema of the target table.
            target_table (str): The name of the target table.

        Returns:
            tuple: A tuple containing two lists of column names:
                - source_columns: List of column names in the source table.
                - target_columns: List of column names in the target table.
        """
        target_columns = self.get_column_name_list(target_schema, target_table)
        source_columns = self.get_column_name_list(source_schema, source_table)
        return source_columns, target_columns

    def get_column_dict_list(self, column_names, schema, table):
        """
        Converts a list of column names into a list of dictionaries containing column information.

        Args:
            column_names (list): List of column names.
            schema (str): The schema of the table.
            table (str): The name of the table.

        Returns:
            list: A list of dictionaries, each containing the column name and its data.
        """
        column_list = []
        for column in column_names:
            query = f'select distinct ([{column}]) from {schema}.{table}'
            column_series = pd.read_sql(query, self.connection)[column]
            column_series = column_series.dropna()
            column_dict = {"name": column, "data": column_series}
            column_list.append(column_dict)
        return column_list

    def get_column_name_list(self, schema, table):
        """
        Retrieves the column names from a specified table within a schema.

        Args:
            schema (str): The schema of the table.
            table (str): The name of the table.

        Returns:
            list: A list of column names in the specified table.
        """
        query = (
            f'select COLUMN_NAME '
            f'from INFORMATION_SCHEMA.columns '
            f'where table_schema = \'{schema}\' and table_name = \'{table}\'')
        column_info_df = pd.read_sql(query, self.connection)
        return column_info_df['COLUMN_NAME'].tolist()

    def find_table_similarities_in_schema(self, schema: str):
        """
        Finds similarities, same names, and unique columns between tables within a schema.

        Args:
            schema (str): The schema to analyze.

        Returns:
            tuple: A tuple containing three pandas DataFrames:
                - schema_same_name_df: DataFrame with columns that have the same name in the tables.
                - schema_similarities_df: DataFrame with columns that are similar between the tables.
                - schema_unique_df: DataFrame with columns that are unique to each table.
        """
        table_list = self.get_table_list_for_schema(schema)
        same_name_list, similarity_list, unique_list = self.get_comparison_for_table_list(schema, table_list)
        schema_similarities_df = pd.concat(similarity_list)
        schema_same_name_df = pd.concat(same_name_list)
        schema_unique_df = pd.concat(unique_list)
        schema_similarities_df, schema_unique_df = self.get_combined_schema_comparisons(schema_similarities_df,
                                                                                        schema_unique_df)
        return schema_same_name_df, schema_similarities_df, schema_unique_df

    @staticmethod
    def get_combined_schema_comparisons(schema_similarities_df, schema_unique_df):
        """
        Combines and filters the similarities and unique columns DataFrames for a schema.

        Args:
            schema_similarities_df (pd.DataFrame): DataFrame with columns that are similar between tables.
            schema_unique_df (pd.DataFrame): DataFrame with columns that are unique to each table.

        Returns:
            tuple: A tuple containing two pandas DataFrames:
                - schema_similarities_df: Filtered DataFrame with columns that are similar between tables.
                - schema_unique_df: Filtered DataFrame with columns that are unique to each table.
        """
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

    def get_comparison_for_table_list(self, schema, table_list):
        """
        Compares tables within a schema and identifies similarities, same names, and unique columns.

        Args:
            schema (str): The schema containing the tables to compare.
            table_list (list): List of table names within the schema.

        Returns:
            tuple: A tuple containing three lists of pandas DataFrames:
                - same_name_list: List of DataFrames with columns that have the same name in the tables.
                - similarity_list: List of DataFrames with columns that are similar between the tables.
                - unique_list: List of DataFrames with columns that are unique to each table.
        """
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

    def get_table_list_for_schema(self, schema):
        """
        Retrieves the list of table names from a specified schema.

        Args:
            schema (str): The schema to retrieve the table names from.

        Returns:
            list: A list of table names in the specified schema.
        """
        query = (
            f'select TABLE_NAME from INFORMATION_SCHEMA.TABLES '
            f'where TABLE_SCHEMA = \'{schema}\' and TABLE_TYPE = \'BASE TABLE\';'
        )
        table_info_df = pd.read_sql(query, self.connection)
        table_list = table_info_df['TABLE_NAME'].tolist()
        return table_list
