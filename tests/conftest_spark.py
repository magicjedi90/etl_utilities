"""Shared test utilities for Spark DataFrame tests."""
import datetime
import unittest
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from src.etl.dataframe.spark.cleaner import SparkCleaner


class SparkTestCase(unittest.TestCase):
    """Base class for Spark tests with shared session management and helpers."""

    spark: SparkSession = None

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .master("local[1]")
            .appName(cls.__name__)
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        if cls.spark:
            cls.spark.stop()

    def make_df(self, column_name: str, values: list[Any]) -> DataFrame:
        """Create a single-column string DataFrame."""
        data = [(v,) for v in values]
        schema = StructType([StructField(column_name, StringType(), True)])
        return self.spark.createDataFrame(data, schema)

    def make_multi_df(self, columns: dict[str, list[Any]], types: dict[str, type] | None = None) -> DataFrame:
        """Create a multi-column DataFrame. Default type is StringType."""
        types = types or {}
        col_names = list(columns.keys())
        vals = list(columns.values())
        data = [tuple(vals[i][j] for i in range(len(columns))) for j in range(len(vals[0]))]

        def get_spark_type(col_name):
            py_type = types.get(col_name, str)
            return IntegerType() if py_type == int else StringType()

        schema = StructType([StructField(n, get_spark_type(n), True) for n in col_names])
        return self.spark.createDataFrame(data, schema)

    def clean(self, column_name: str, values: list[Any], **kwargs) -> tuple[str, list[Any]]:
        """Create DataFrame, clean it, and return (dtype, values)."""
        df = self.make_df(column_name, values)
        result = SparkCleaner.clean_all_types(df, **kwargs)
        result_type = result.schema[column_name].dataType.simpleString()
        result_values = [row[column_name] for row in result.collect()]
        return result_type, result_values

    def assert_type(self, column_name: str, values: list[Any], expected_type: str, **kwargs):
        """Assert that a column is cleaned to the expected type."""
        result_type, _ = self.clean(column_name, values, **kwargs)
        self.assertEqual(result_type, expected_type)

    def assert_values(self, column_name: str, values: list[Any], expected_type: str, expected_values: list[Any], **kwargs):
        """Assert column type and exact values after cleaning."""
        result_type, result_values = self.clean(column_name, values, **kwargs)
        self.assertEqual(result_type, expected_type)
        self.assertEqual(result_values, expected_values)

    def assert_dates(self, column_name: str, values: list[Any], expected: list[tuple[int, int, int] | None]):
        """Assert date column values match expected (year, month, day) tuples. Allows ±1 day tolerance."""
        result_type, result_values = self.clean(column_name, values)
        self.assertEqual(result_type, "timestamp")

        for i, (actual, exp) in enumerate(zip(result_values, expected)):
            if exp is None:
                self.assertIsNone(actual, f"Value at index {i} should be None")
            else:
                self.assertIsInstance(actual, datetime.datetime)
                expected_date = datetime.date(*exp)
                day_diff = abs((actual.date() - expected_date).days)
                self.assertLessEqual(day_diff, 1, f"Date at index {i}: {actual.date()} not within 1 day of {expected_date}")

    def assert_utc_timestamp(self, result: DataFrame, column_name: str, row_index: int, expected: str):
        """Assert that a timestamp in Spark matches the expected UTC value."""
        formatted = result.select(
            F.date_format(F.col(column_name), "yyyy-MM-dd HH:mm:ss").alias("fmt")
        ).collect()
        actual = formatted[row_index]["fmt"]
        self.assertEqual(actual, expected, f"Row {row_index}: expected UTC '{expected}', got '{actual}'")
