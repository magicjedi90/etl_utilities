#!/usr/bin/env python3
"""Tests for SparkCleaner DataFrame operations (clean_df, column names, edge cases)."""
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from tests.conftest_spark import SparkTestCase
from src.etl.dataframe.spark.cleaner import SparkCleaner


class TestColumnNames(SparkTestCase):
    """Tests for column name conversion."""

    def test_to_snake_case(self):
        df = self.make_multi_df({
            "UserID": [1, 2],
            "user name": ["John", "Jane"],
            "with$Special#Chars": ["a", "b"],
        }, types={"UserID": int})
        result = SparkCleaner.column_names_to_snake_case(df)
        self.assertEqual(set(result.columns), {"user_id", "user_name", "with_dollars_special_num_chars"})

    def test_various_formats(self):
        schema = StructType([
            StructField("Customer Name", StringType(), True),
            StructField("Total$Amount", StringType(), True),
            StructField("Is Active?", StringType(), True),
            StructField("User&ID", StringType(), True),
        ])
        df = self.spark.createDataFrame([("a", "b", "c", "d")], schema)
        result = SparkCleaner.column_names_to_snake_case(df)
        expected = ["customer_name", "total_dollars_amount", "is_active", "user_and_id"]
        for col in expected:
            self.assertIn(col, result.columns)


class TestCleanDf(SparkTestCase):
    """Tests for clean_df method."""

    def test_removes_all_null_rows(self):
        df = self.make_multi_df({
            "id": [1, None, 2, None],
            "name": ["John", None, "Jane", None],
        }, types={"id": int})
        result = SparkCleaner.clean_df(df)
        self.assertEqual(result.count(), 2)

    def test_removes_all_null_columns(self):
        df = self.make_multi_df({
            "id": [1, 2, 3],
            "empty": [None, None, None],
            "text": ["a", "b", "c"],
        }, types={"id": int})
        result = SparkCleaner.clean_df(df)
        self.assertNotIn("empty", result.columns)
        self.assertIn("id", result.columns)
        self.assertIn("text", result.columns)

    def test_preserves_partial_null_columns(self):
        df = self.make_multi_df({
            "id": [1, 2, 3],
            "partial": ["a", None, "b"],
        }, types={"id": int})
        result = SparkCleaner.clean_df(df)
        self.assertIn("partial", result.columns)
        self.assertEqual(result.count(), 3)


class TestMixedTypes(SparkTestCase):
    """Tests for mixed type columns and comprehensive cleaning."""

    def test_comprehensive_cleaning(self):
        schema = StructType([
            StructField("bool_col", StringType(), True),
            StructField("int_col", StringType(), True),
            StructField("float_col", StringType(), True),
            StructField("date_col", StringType(), True),
            StructField("str_col", StringType(), True),
        ])
        data = [
            ("y", "$1,000", "$45.67", "2021-06-15", "text1"),
            ("false", "2000", "%89.10", "2022-07-20", "text2"),
            ("no", "789", "50.5", "2022-08-25", "text3"),
        ]
        df = self.spark.createDataFrame(data, schema)
        result = SparkCleaner.clean_all_types(df)

        expected = {"bool_col": "boolean", "int_col": "bigint", "float_col": "double", "date_col": "timestamp", "str_col": "string"}
        for col, exp_type in expected.items():
            self.assertEqual(result.schema[col].dataType.simpleString(), exp_type)

    def test_mixed_types_stay_string(self):
        self.assert_type("col", ["100", "abc", "200"], "string")


class TestEdgeCases(SparkTestCase):
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        schema = StructType([StructField("col1", StringType(), True), StructField("col2", StringType(), True)])
        df = self.spark.createDataFrame([], schema)
        result = SparkCleaner.clean_all_types(df)
        self.assertEqual(result.count(), 0)
        self.assertEqual(len(result.columns), 2)

    def test_single_row(self):
        schema = StructType([
            StructField("bool", StringType(), True),
            StructField("int", StringType(), True),
            StructField("float", StringType(), True),
            StructField("date", StringType(), True),
        ])
        df = self.spark.createDataFrame([("yes", "100", "1.5", "2021-06-15")], schema)
        result = SparkCleaner.clean_all_types(df)
        expected = {"bool": "boolean", "int": "bigint", "float": "double", "date": "timestamp"}
        for col, exp_type in expected.items():
            self.assertEqual(result.schema[col].dataType.simpleString(), exp_type)

    def test_single_column(self):
        self.assert_type("col", ["100", "200", "300"], "bigint")

    def test_all_null_column_skipped(self):
        df = self.make_multi_df({"int_col": ["100", "200", "300"], "null_col": [None, None, None]})
        result = SparkCleaner.clean_all_types(df)
        self.assertEqual(result.schema["int_col"].dataType.simpleString(), "bigint")
        self.assertEqual(result.schema["null_col"].dataType.simpleString(), "string")

    def test_whitespace_only(self):
        self.assert_type("col", ["  ", "\t", "\n", "   "], "string")

    def test_preserves_non_string_columns(self):
        schema = StructType([
            StructField("int_col", IntegerType(), True),
            StructField("str_col", StringType(), True),
        ])
        df = self.spark.createDataFrame([(1, "100"), (2, "200"), (3, "300")], schema)
        result = SparkCleaner.clean_all_types(df)
        self.assertEqual(result.schema["int_col"].dataType.simpleString(), "bigint")
        self.assertEqual(result.schema["str_col"].dataType.simpleString(), "bigint")


if __name__ == "__main__":
    import unittest
    unittest.main()
