import pyodbc
import pandas as pd

class BulkLoader:
    def __init__(self, connection: pyodbc.Connection, schema_name: str, table_name: str):
        """
        Initializes the BulkLoader object.

        Args:
            connection: The database connection object.
            schema_name: The schema of the destination table.
            table_name: The name of the destination table.
        """
        self.connection = connection
