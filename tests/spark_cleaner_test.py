#!/usr/bin/env python3
"""
Test suite for Spark DataFrame Cleaner
"""
import datetime
import unittest
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from src.etl.dataframe.spark.cleaner import SparkCleaner


class TestSparkCleaner(unittest.TestCase):
    """Test cases for SparkCleaner class"""

    @classmethod
    def setUpClass(cls):
        """Set up Spark session for all tests"""
        cls.spark = (
            SparkSession.builder
            .master("local[1]")
            .appName("SparkCleanerTest")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        """Stop Spark session after all tests"""
        cls.spark.stop()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def create_string_dataframe(self, column_name: str, values: list[Any]) -> DataFrame:
        """Create a single-column DataFrame with string type."""
        data = [(v,) for v in values]
        schema = StructType([StructField(column_name, StringType(), True)])
        return self.spark.createDataFrame(data, schema)

    def clean_and_get_values(self, column_name: str, values: list[Any]) -> tuple[str, list[Any]]:
        """Create DataFrame, clean it, and return (dtype, values)."""
        dataframe = self.create_string_dataframe(column_name, values)
        result = SparkCleaner.clean_all_types(dataframe)
        result_type = result.schema[column_name].dataType.simpleString()
        result_values = [row[column_name] for row in result.collect()]
        return result_type, result_values

    def assert_column_type(self, column_name: str, values: list[Any], expected_type: str):
        """Assert that a column is cleaned to the expected type."""
        result_type, _ = self.clean_and_get_values(column_name, values)
        self.assertEqual(result_type, expected_type)

    def assert_cleaned_values(
        self, column_name: str, values: list[Any], expected_type: str, expected_values: list[Any]
    ):
        """Assert column type and exact values after cleaning."""
        result_type, result_values = self.clean_and_get_values(column_name, values)
        self.assertEqual(result_type, expected_type)
        self.assertEqual(result_values, expected_values)

    def assert_date_values(
        self,
        column_name: str,
        string_values: list[Any],
        expected_dates: list[tuple[int, int, int] | None],
    ):
        """Assert date column values match expected (year, month, day) tuples.

        Allows ±1 day tolerance to handle timezone conversion differences between
        Spark (UTC) and local Python datetime display.
        None in expected_dates means the value should be null.
        """
        result_type, result_values = self.clean_and_get_values(column_name, string_values)
        self.assertEqual(result_type, "timestamp")

        for i, (actual, expected) in enumerate(zip(result_values, expected_dates)):
            if expected is None:
                self.assertIsNone(actual, f"Value at index {i} should be None")
            else:
                self.assertIsInstance(actual, datetime.datetime, f"Value at index {i} should be datetime")
                expected_year, expected_month, expected_day = expected
                expected_date = datetime.date(expected_year, expected_month, expected_day)
                actual_date = actual.date()
                # Allow ±1 day tolerance for timezone differences
                day_difference = abs((actual_date - expected_date).days)
                self.assertLessEqual(
                    day_difference, 1,
                    f"Date at index {i}: {actual_date} not within 1 day of {expected_date}"
                )

    # =========================================================================
    # Column Name Tests
    # =========================================================================

    def test_column_names_to_snake_case(self):
        """Test conversion of column names to snake_case"""
        data = [(1, "John Doe", "Value1"), (2, "Jane Doe", "Value2")]
        schema = StructType([
            StructField("UserID", IntegerType(), True),
            StructField("user name", StringType(), True),
            StructField("with$Special#Chars", StringType(), True)
        ])
        dataframe = self.spark.createDataFrame(data, schema)

        result = SparkCleaner.column_names_to_snake_case(dataframe)

        self.assertEqual(
            set(result.columns),
            {"user_id", "user_name", "with_dollars_special_num_chars"}
        )

    def test_column_names_to_snake_case_various_formats(self):
        """Test snake_case conversion with various input formats"""
        data = [("a", "b", "c", "d", "e")]
        schema = StructType([
            StructField("Customer Name", StringType(), True),
            StructField("First Name", StringType(), True),
            StructField("Total$Amount", StringType(), True),
            StructField("Is Active?", StringType(), True),
            StructField("User&ID", StringType(), True)
        ])
        dataframe = self.spark.createDataFrame(data, schema)

        result = SparkCleaner.column_names_to_snake_case(dataframe)

        expected_columns = ["customer_name", "first_name", "total_dollars_amount", "is_active", "user_and_id"]
        for expected in expected_columns:
            self.assertIn(expected, result.columns)

    # =========================================================================
    # Boolean Parsing Tests
    # =========================================================================

    def test_clean_booleans_truthy_values(self):
        """Test parsing of various truthy boolean values"""
        truthy_values = ["y", "yes", "t", "true", "on", "1", "Y", "YES", "T", "TRUE", "ON"]
        result_type, result_values = self.clean_and_get_values("bool_col", truthy_values)

        self.assertEqual(result_type, "boolean")
        self.assertTrue(all(v is True for v in result_values))

    def test_clean_booleans_falsy_values(self):
        """Test parsing of various falsy boolean values"""
        falsy_values = ["n", "no", "f", "false", "off", "0", "N", "NO", "F", "FALSE", "OFF"]
        result_type, result_values = self.clean_and_get_values("bool_col", falsy_values)

        self.assertEqual(result_type, "boolean")
        self.assertTrue(all(v is False for v in result_values))

    def test_clean_booleans_mixed_values(self):
        """Test parsing of mixed boolean values"""
        self.assert_cleaned_values(
            "bool_col",
            ["yes", "no", "true", "false", "1", "0"],
            "boolean",
            [True, False, True, False, True, False]
        )

    def test_clean_booleans_with_nulls(self):
        """Test that null values are preserved in boolean columns"""
        self.assert_cleaned_values(
            "bool_col",
            ["yes", None, "no", None],
            "boolean",
            [True, None, False, None]
        )

    # =========================================================================
    # Integer Parsing Tests
    # =========================================================================

    def test_clean_integers_basic(self):
        """Test parsing of basic integer values"""
        self.assert_cleaned_values(
            "int_col",
            ["123", "456", "789", "0", "-100"],
            "bigint",
            [123, 456, 789, 0, -100]
        )

    def test_clean_integers_with_formatting(self):
        """Test parsing of integers with $, %, and comma formatting"""
        self.assert_cleaned_values(
            "int_col",
            ["$1,000", "%200", "3,000,000", "$5"],
            "bigint",
            [1000, 200, 3000000, 5]
        )

    def test_clean_integers_with_decimal_zero(self):
        """Test that values like '100.00' are recognized as integers."""
        self.assert_cleaned_values(
            "int_col",
            ["100.00", "200.0", "300.000"],
            "bigint",
            [100, 200, 300]
        )

    def test_clean_integers_with_nulls(self):
        """Test that null values are preserved in integer columns"""
        self.assert_cleaned_values(
            "int_col",
            ["100", None, "200", None],
            "bigint",
            [100, None, 200, None]
        )

    # =========================================================================
    # Float Parsing Tests
    # =========================================================================

    def test_clean_floats_basic(self):
        """Test parsing of basic float values"""
        result_type, result_values = self.clean_and_get_values(
            "float_col", ["1.5", "2.75", "3.14159", "-0.5"]
        )

        self.assertEqual(result_type, "float")
        self.assertAlmostEqual(result_values[0], 1.5, places=2)
        self.assertAlmostEqual(result_values[1], 2.75, places=2)
        self.assertAlmostEqual(result_values[2], 3.14159, places=4)
        self.assertAlmostEqual(result_values[3], -0.5, places=2)

    def test_clean_floats_with_formatting(self):
        """Test parsing of floats with $, %, and comma formatting"""
        result_type, result_values = self.clean_and_get_values(
            "float_col", ["$1,000.50", "%99.9", "1,234.56"]
        )

        self.assertEqual(result_type, "float")
        self.assertAlmostEqual(result_values[0], 1000.50, places=2)
        self.assertAlmostEqual(result_values[1], 99.9, places=1)
        self.assertAlmostEqual(result_values[2], 1234.56, places=2)

    def test_clean_floats_with_nulls(self):
        """Test that null values are preserved in float columns"""
        result_type, result_values = self.clean_and_get_values(
            "float_col", ["1.5", None, "2.5", None]
        )

        self.assertEqual(result_type, "float")
        self.assertAlmostEqual(result_values[0], 1.5, places=2)
        self.assertIsNone(result_values[1])
        self.assertAlmostEqual(result_values[2], 2.5, places=2)
        self.assertIsNone(result_values[3])

    # =========================================================================
    # Date Parsing Tests
    # =========================================================================

    def test_clean_dates_iso_format(self):
        """Test parsing of ISO format dates"""
        self.assert_date_values(
            "date_col",
            ["2021-06-15", "2022-07-20", "2023-08-25"],
            [(2021, 6, 15), (2022, 7, 20), (2023, 8, 25)]
        )

    def test_clean_dates_us_format(self):
        """Test parsing of US format dates (MM/dd/yyyy)"""
        self.assert_date_values(
            "date_col",
            ["06/15/2021", "07/20/2022", "08/25/2023"],
            [(2021, 6, 15), (2022, 7, 20), (2023, 8, 25)]
        )

    def test_clean_dates_with_time(self):
        """Test parsing of dates with time components"""
        self.assert_date_values(
            "date_col",
            ["2021-06-15 10:30:00", "2022-07-20 14:45:30"],
            [(2021, 6, 15), (2022, 7, 20)]
        )

    def test_clean_dates_with_nulls(self):
        """Test that null values are preserved in date columns"""
        self.assert_date_values(
            "date_col",
            ["2021-06-15", None, "2022-07-20", None],
            [(2021, 6, 15), None, (2022, 7, 20), None]
        )

    # =========================================================================
    # Empty and Whitespace String Handling Tests
    # =========================================================================

    def test_empty_strings_treated_as_null_for_integers(self):
        """Test that empty strings are treated as null for integer columns"""
        self.assert_cleaned_values(
            "int_col",
            ["100", "", "200", "   ", None],
            "bigint",
            [100, None, 200, None, None]
        )

    def test_empty_strings_treated_as_null_for_floats(self):
        """Test that empty strings are treated as null for float columns"""
        result_type, result_values = self.clean_and_get_values(
            "float_col", ["1.5", "", "2.5", "   ", None]
        )

        self.assertEqual(result_type, "float")
        self.assertAlmostEqual(result_values[0], 1.5, places=2)
        self.assertIsNone(result_values[1])
        self.assertAlmostEqual(result_values[2], 2.5, places=2)
        self.assertIsNone(result_values[3])
        self.assertIsNone(result_values[4])

    def test_empty_strings_treated_as_null_for_booleans(self):
        """Test that empty strings are treated as null for boolean columns"""
        self.assert_cleaned_values(
            "bool_col",
            ["yes", "", "no", "   ", None],
            "boolean",
            [True, None, False, None, None]
        )

    def test_empty_strings_treated_as_null_for_dates(self):
        """Test that empty strings are treated as null for date columns"""
        self.assert_date_values(
            "date_col",
            ["2021-06-15", "", "2022-07-20", "   ", None],
            [(2021, 6, 15), None, (2022, 7, 20), None, None]
        )

    # =========================================================================
    # Mixed Type Column Tests
    # =========================================================================

    def test_clean_all_types_comprehensive(self):
        """Test comprehensive cleaning with multiple column types"""
        data = [
            ("y", "$1,000", "$45.67", "2021-06-15", "text1"),
            ("false", "2000", "%89.10", "2022-07-20", "text2"),
            ("no", "789", "50.5", "2022-08-25", "text3")
        ]
        schema = StructType([
            StructField("boolean_col", StringType(), True),
            StructField("integer_col", StringType(), True),
            StructField("float_col", StringType(), True),
            StructField("date_col", StringType(), True),
            StructField("string_col", StringType(), True)
        ])
        dataframe = self.spark.createDataFrame(data, schema)

        result = SparkCleaner.clean_all_types(dataframe)

        expected_types = {
            "boolean_col": "boolean",
            "integer_col": "bigint",
            "float_col": "float",
            "date_col": "timestamp",
            "string_col": "string"
        }
        for column_name, expected_type in expected_types.items():
            self.assertEqual(
                result.schema[column_name].dataType.simpleString(),
                expected_type,
                f"Column {column_name} should be {expected_type}"
            )

    def test_column_not_converted_if_mixed_types(self):
        """Test that columns with mixed types are not converted"""
        self.assert_column_type("mixed_col", ["100", "abc", "200"], "string")

    def test_type_priority_boolean_over_integer(self):
        """Test that boolean type takes priority over integer for '1' and '0'"""
        self.assert_cleaned_values(
            "col",
            ["1", "0", "1", "0"],
            "boolean",
            [True, False, True, False]
        )

    # =========================================================================
    # DataFrame Cleaning Tests
    # =========================================================================

    def test_clean_df_removes_all_null_rows(self):
        """Test that clean_df removes rows where all values are null"""
        data = [
            (1, "John", "A"),
            (None, None, None),
            (2, "Jane", "B"),
            (None, None, None)
        ]
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("grade", StringType(), True)
        ])
        dataframe = self.spark.createDataFrame(data, schema)

        result = SparkCleaner.clean_df(dataframe)

        self.assertEqual(result.count(), 2)

    def test_clean_df_removes_all_null_columns(self):
        """Test that clean_df removes columns where all values are null"""
        data = [(1, None, "Text1"), (2, None, "Text2"), (3, None, "Text3")]
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("empty_col", StringType(), True),
            StructField("text_col", StringType(), True)
        ])
        dataframe = self.spark.createDataFrame(data, schema)

        result = SparkCleaner.clean_df(dataframe)

        self.assertNotIn("empty_col", result.columns)
        self.assertIn("id", result.columns)
        self.assertIn("text_col", result.columns)

    def test_clean_df_preserves_partial_null_columns(self):
        """Test that clean_df preserves columns with some null values"""
        data = [(1, "Value1", "Text1"), (2, None, "Text2"), (3, "Value3", "Text3")]
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("partial_col", StringType(), True),
            StructField("text_col", StringType(), True)
        ])
        dataframe = self.spark.createDataFrame(data, schema)

        result = SparkCleaner.clean_df(dataframe)

        self.assertIn("partial_col", result.columns)
        self.assertEqual(result.count(), 3)

    # =========================================================================
    # Edge Case Tests
    # =========================================================================

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        schema = StructType([
            StructField("col1", StringType(), True),
            StructField("col2", StringType(), True)
        ])
        dataframe = self.spark.createDataFrame([], schema)

        result = SparkCleaner.clean_all_types(dataframe)

        self.assertEqual(result.count(), 0)
        self.assertEqual(len(result.columns), 2)

    def test_single_row_dataframe(self):
        """Test handling of single row DataFrame"""
        data = [("yes", "100", "1.5", "2021-06-15")]
        schema = StructType([
            StructField("bool_col", StringType(), True),
            StructField("int_col", StringType(), True),
            StructField("float_col", StringType(), True),
            StructField("date_col", StringType(), True)
        ])
        dataframe = self.spark.createDataFrame(data, schema)

        result = SparkCleaner.clean_all_types(dataframe)

        expected_types = {"bool_col": "boolean", "int_col": "bigint", "float_col": "float", "date_col": "timestamp"}
        for column_name, expected_type in expected_types.items():
            self.assertEqual(result.schema[column_name].dataType.simpleString(), expected_type)

    def test_single_column_dataframe(self):
        """Test handling of single column DataFrame"""
        self.assert_column_type("value", ["100", "200", "300"], "bigint")

    def test_all_null_column_in_clean_all_types(self):
        """Test that all-null columns are skipped in clean_all_types"""
        data = [("100", None), ("200", None), ("300", None)]
        schema = StructType([
            StructField("int_col", StringType(), True),
            StructField("null_col", StringType(), True)
        ])
        dataframe = self.spark.createDataFrame(data, schema)

        result = SparkCleaner.clean_all_types(dataframe)

        self.assertEqual(result.schema["int_col"].dataType.simpleString(), "bigint")
        self.assertEqual(result.schema["null_col"].dataType.simpleString(), "string")

    def test_whitespace_only_values(self):
        """Test handling of whitespace-only values"""
        self.assert_column_type("col", ["  ", "\t", "\n", "   "], "string")

    def test_preserves_non_string_columns(self):
        """Test that already-typed columns are handled correctly"""
        data = [(1, "100"), (2, "200"), (3, "300")]
        schema = StructType([
            StructField("int_col", IntegerType(), True),
            StructField("string_col", StringType(), True)
        ])
        dataframe = self.spark.createDataFrame(data, schema)

        result = SparkCleaner.clean_all_types(dataframe)

        self.assertEqual(result.schema["int_col"].dataType.simpleString(), "bigint")
        self.assertEqual(result.schema["string_col"].dataType.simpleString(), "bigint")

    def test_large_numbers(self):
        """Test handling of large numbers"""
        self.assert_cleaned_values(
            "large_int",
            ["1000000000", "2000000000", "999999999"],
            "bigint",
            [1000000000, 2000000000, 999999999]
        )

    def test_trillion_scale_numbers(self):
        """Test handling of trillion-scale numbers that exceed 32-bit int range"""
        self.assert_cleaned_values(
            "huge_int",
            ["2300000000000", "9000000000000", "100000000000"],
            "bigint",
            [2300000000000, 9000000000000, 100000000000]
        )

    def test_negative_numbers(self):
        """Test handling of negative numbers"""
        self.assert_column_type("negative", ["-100", "-200.5", "-0.001"], "float")


if __name__ == "__main__":
    unittest.main()
