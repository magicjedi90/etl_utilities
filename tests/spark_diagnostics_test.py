#!/usr/bin/env python3
"""Test suite for Spark DataFrame diagnostics utilities."""
import unittest
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, StringType

from src.etl.dataframe.spark.cleaner import SparkCleaner
from src.etl.dataframe.spark.diagnostics import (
    ColumnDiagnostics,
    get_conversion_diagnostics,
    get_failed_value_samples,
)


class SparkTestCase(unittest.TestCase):
    """Base class for Spark tests with shared session management."""

    spark: SparkSession = None

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .master("local[1]")
            .appName("SparkDiagnosticsTest")
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

    def make_multi_df(self, columns: dict[str, list[Any]]) -> DataFrame:
        """Create a multi-column string DataFrame."""
        col_names = list(columns.keys())
        vals = list(columns.values())
        data = [tuple(vals[i][j] for i in range(len(columns))) for j in range(len(vals[0]))]
        schema = StructType([StructField(n, StringType(), True) for n in col_names])
        return self.spark.createDataFrame(data, schema)


class TestConversionDiagnostics(SparkTestCase):
    """Test cases for get_conversion_diagnostics function."""

    def test_reports_conversion_counts(self):
        """Test that diagnostics correctly report successful and failed conversion counts."""
        original = self.make_df("int_col", ["100", "200", "abc", "300"])
        cleaned = SparkCleaner.clean_all_types(original)
        diag = get_conversion_diagnostics(original, cleaned)["int_col"]

        self.assertIsInstance(diag, ColumnDiagnostics)
        self.assertEqual(diag.total_values, 4)
        self.assertEqual(diag.null_values, 0)

    def test_clean_data_shows_zero_failures(self):
        """Test that clean data with no conversion failures shows zero failures."""
        original = self.make_df("int_col", ["100", "200", "300", "400"])
        cleaned = SparkCleaner.clean_all_types(original)
        diag = get_conversion_diagnostics(original, cleaned)["int_col"]

        self.assertEqual(diag.failed_conversions, 0)
        self.assertEqual(diag.successful_conversions, 4)
        self.assertEqual(diag.success_rate, 100.0)
        self.assertEqual(diag.sample_failed_values, [])

    def test_handles_nulls_correctly(self):
        """Test that null values in original data are counted separately from failures."""
        original = self.make_df("int_col", ["100", None, "200", None, "300"])
        cleaned = SparkCleaner.clean_all_types(original)
        diag = get_conversion_diagnostics(original, cleaned)["int_col"]

        self.assertEqual(diag.total_values, 5)
        self.assertEqual(diag.null_values, 2)
        self.assertEqual(diag.successful_conversions, 3)
        self.assertEqual(diag.failed_conversions, 0)

    def test_handles_empty_strings(self):
        """Test that empty strings are treated as null (not failures)."""
        original = self.make_df("int_col", ["100", "", "200", "   ", "300"])
        cleaned = SparkCleaner.clean_all_types(original)
        diag = get_conversion_diagnostics(original, cleaned)["int_col"]

        self.assertEqual(diag.failed_conversions, 0)
        self.assertEqual(diag.successful_conversions, 3)

    def test_reports_original_and_inferred_types(self):
        """Test that diagnostics correctly report original and inferred types."""
        original = self.make_df("bool_col", ["yes", "no", "true", "false"])
        cleaned = SparkCleaner.clean_all_types(original)
        diag = get_conversion_diagnostics(original, cleaned)["bool_col"]

        self.assertEqual(diag.original_type, "string")
        self.assertEqual(diag.inferred_type, "boolean")

    def test_reports_string_fallback_type(self):
        """Test that diagnostics reports string type when fallback occurs."""
        original = self.make_df("mixed_col", ["100", "abc", "200"])
        cleaned = SparkCleaner.clean_all_types(original)
        diag = get_conversion_diagnostics(original, cleaned)["mixed_col"]

        self.assertEqual(diag.original_type, "string")
        self.assertEqual(diag.inferred_type, "string")

    def test_limits_sample_size(self):
        """Test that sample_failed_values parameter limits the number of samples."""
        values = [str(i) for i in range(10)] + ["a", "b", "c", "d", "e"]
        original = self.make_df("col", values)
        cleaned = SparkCleaner.clean_all_types(original)
        diag = get_conversion_diagnostics(original, cleaned, sample_failed_values=2)["col"]

        self.assertLessEqual(len(diag.sample_failed_values), 2)

    def test_handles_multiple_columns(self):
        """Test that diagnostics correctly handles multiple columns."""
        original = self.make_multi_df({
            "int_col": ["100", "200", "300"],
            "bool_col": ["yes", "no", "true"],
            "str_col": ["abc", "def", "ghi"],
        })
        cleaned = SparkCleaner.clean_all_types(original)
        diag = get_conversion_diagnostics(original, cleaned)

        self.assertEqual(len(diag), 3)
        self.assertEqual(diag["int_col"].inferred_type, "bigint")
        self.assertEqual(diag["bool_col"].inferred_type, "boolean")
        self.assertEqual(diag["str_col"].inferred_type, "string")

    def test_with_mixed_column_quality(self):
        """Test diagnostics with some clean columns and some with issues."""
        original = self.make_multi_df({
            "clean_int": ["100", "200", "300"],
            "mixed_col": ["100", "abc", "200"],
        })
        cleaned = SparkCleaner.clean_all_types(original)
        diag = get_conversion_diagnostics(original, cleaned)

        self.assertEqual(diag["clean_int"].failed_conversions, 0)
        self.assertEqual(diag["clean_int"].success_rate, 100.0)
        self.assertEqual(diag["mixed_col"].inferred_type, "string")


class TestGetFailedValueSamples(SparkTestCase):
    """Test cases for get_failed_value_samples function."""

    def test_returns_empty_list_when_no_failures(self):
        """Test that empty list is returned when there are no conversion failures."""
        original = self.make_df("col", ["100", "200", "300"])
        cleaned = SparkCleaner.clean_all_types(original)
        samples = get_failed_value_samples(original, cleaned, "col", limit=5)
        self.assertEqual(samples, [])

    def test_respects_limit_parameter(self):
        """Test that the function respects the limit parameter."""
        original = self.make_df("col", ["100", "a", "200", "b", "300", "c", "400", "d"])
        cleaned = SparkCleaner.clean_all_types(original)
        samples = get_failed_value_samples(original, cleaned, "col", limit=2)
        self.assertLessEqual(len(samples), 2)

    def test_returns_distinct_values(self):
        """Test that duplicate failed values are not repeated in samples."""
        original = self.make_df("col", ["100", "abc", "200", "abc", "300", "abc"])
        cleaned = SparkCleaner.clean_all_types(original)
        samples = get_failed_value_samples(original, cleaned, "col", limit=10)
        self.assertEqual(len(samples), len(set(samples)))


class TestColumnDiagnosticsDataclass(unittest.TestCase):
    """Test cases for ColumnDiagnostics dataclass."""

    def test_dataclass_fields(self):
        """Test that ColumnDiagnostics has all expected fields."""
        diag = ColumnDiagnostics(
            column_name="test_col",
            original_type="string",
            inferred_type="bigint",
            total_values=100,
            null_values=5,
            successful_conversions=90,
            failed_conversions=5,
            success_rate=94.74,
            sample_failed_values=["abc", "def"],
        )

        self.assertEqual(diag.column_name, "test_col")
        self.assertEqual(diag.original_type, "string")
        self.assertEqual(diag.inferred_type, "bigint")
        self.assertEqual(diag.total_values, 100)
        self.assertEqual(diag.null_values, 5)
        self.assertEqual(diag.successful_conversions, 90)
        self.assertEqual(diag.failed_conversions, 5)
        self.assertEqual(diag.success_rate, 94.74)
        self.assertEqual(diag.sample_failed_values, ["abc", "def"])

    def test_dataclass_empty_samples(self):
        """Test ColumnDiagnostics with empty sample list."""
        diag = ColumnDiagnostics(
            column_name="clean_col",
            original_type="string",
            inferred_type="bigint",
            total_values=100,
            null_values=0,
            successful_conversions=100,
            failed_conversions=0,
            success_rate=100.0,
            sample_failed_values=[],
        )

        self.assertEqual(diag.failed_conversions, 0)
        self.assertEqual(diag.sample_failed_values, [])
        self.assertEqual(diag.success_rate, 100.0)


if __name__ == "__main__":
    unittest.main()
