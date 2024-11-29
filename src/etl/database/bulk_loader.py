import pyodbc
import pandas as pd


class BulkLoader:
    def __init__(self, connection: pyodbc.Connection):
        self.connection = connection