import pyodbc
import pandas as pd
from ..logger import Logger


class FastLoader:
    def __init__(self, connection: pyodbc.Connection, schema_name: str, table_name: str):
        """
        Initializes the FastLoader object.

        Args:
            connection: The connection object to the database.
            schema_name: The name of the schema where the table is located.
            table_name: The name of the table where the data will be inserted.
        """
        self._connection = connection
        self.schema_name = schema_name
        self.table_name = table_name
        self.__logger = Logger().get_logger()

    def fast_insert(self, df: pd.DataFrame, batch_size: int = 1000) -> None:
        """
        Bulk inserts a DataFrame into a table in the database.

        Args:
            df: The DataFrame to be inserted.
            batch_size: The number of rows to insert in each batch.
        """

        # Check if the DataFrame is empty
        if df.empty:
            raise ValueError('DataFrame is empty.')

        # Convert data types to object and replace NaN with None
        df = df.astype(object).where(pd.notnull(df), None)

        # Prepare SQL query
        columns = ', '.join(df.columns)
        placeholders = ', '.join('?' * len(df.columns))
        query = f'INSERT INTO {self.schema_name}.{self.table_name} ({columns}) VALUES ({placeholders});'
        self.__logger.debug(f'Query: {query}')

        # Convert DataFrame to list of tuples
        data = [tuple(row) for row in df.itertuples(index=False, name=None)]

        # Perform the bulk insert
        cursor = self._connection.cursor()
        cursor.fast_executemany = True
        try:
            for i in range(0, len(data), batch_size):
                cursor.executemany(query, data[i:i + batch_size])
                self.__logger.info(
                    f'Inserted {min(i + batch_size, len(data))} rows into {self.schema_name}.{self.table_name}.')
            self.__logger.info(f'Inserted {len(data)} rows into {self.schema_name}.{self.table_name}.')
        except Exception as e:
            cursor.rollback()
            self.__logger.error(f'Error inserting data into {self.schema_name}.{self.table_name}: {str(e)}')
            raise RuntimeError(f'Error inserting data into {self.schema_name}.{self.table_name}: {str(e)}')
