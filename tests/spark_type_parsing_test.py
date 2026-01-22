#!/usr/bin/env python3
"""Tests for Spark DataFrame type parsing (boolean, integer, float, date)."""
from tests.conftest_spark import SparkTestCase
from src.etl.dataframe.spark.cleaner import SparkCleaner


class TestBooleanParsing(SparkTestCase):
    """Tests for boolean type parsing."""

    def test_truthy_values(self):
        truthy = ["y", "yes", "t", "true", "on", "1", "Y", "YES", "T", "TRUE", "ON"]
        result_type, result_values = self.clean("col", truthy)
        self.assertEqual(result_type, "boolean")
        self.assertTrue(all(v is True for v in result_values))

    def test_falsy_values(self):
        falsy = ["n", "no", "f", "false", "off", "0", "N", "NO", "F", "FALSE", "OFF"]
        result_type, result_values = self.clean("col", falsy)
        self.assertEqual(result_type, "boolean")
        self.assertTrue(all(v is False for v in result_values))

    def test_mixed_values(self):
        self.assert_values("col", ["yes", "no", "true", "false", "1", "0"], "boolean", [True, False, True, False, True, False])

    def test_with_nulls(self):
        self.assert_values("col", ["yes", None, "no", None], "boolean", [True, None, False, None])

    def test_empty_strings_as_null(self):
        self.assert_values("col", ["yes", "", "no", "   ", None], "boolean", [True, None, False, None, None])

    def test_priority_over_integer(self):
        self.assert_values("col", ["1", "0", "1", "0"], "boolean", [True, False, True, False])


class TestIntegerParsing(SparkTestCase):
    """Tests for integer type parsing."""

    def test_basic(self):
        self.assert_values("col", ["123", "456", "789", "0", "-100"], "bigint", [123, 456, 789, 0, -100])

    def test_with_formatting(self):
        self.assert_values("col", ["$1,000", "%200", "3,000,000", "$5"], "bigint", [1000, 200, 3000000, 5])

    def test_decimal_zero(self):
        self.assert_values("col", ["100.00", "200.0", "300.000"], "bigint", [100, 200, 300])

    def test_with_nulls(self):
        self.assert_values("col", ["100", None, "200", None], "bigint", [100, None, 200, None])

    def test_empty_strings_as_null(self):
        self.assert_values("col", ["100", "", "200", "   ", None], "bigint", [100, None, 200, None, None])

    def test_large_numbers(self):
        self.assert_values("col", ["1000000000", "2000000000", "999999999"], "bigint", [1000000000, 2000000000, 999999999])

    def test_trillion_scale(self):
        self.assert_values("col", ["2300000000000", "9000000000000", "100000000000"], "bigint", [2300000000000, 9000000000000, 100000000000])


class TestFloatParsing(SparkTestCase):
    """Tests for float type parsing."""

    def test_basic(self):
        result_type, vals = self.clean("col", ["1.5", "2.75", "3.14159", "-0.5"])
        self.assertEqual(result_type, "double")
        self.assertAlmostEqual(vals[0], 1.5, places=2)
        self.assertAlmostEqual(vals[1], 2.75, places=2)
        self.assertAlmostEqual(vals[2], 3.14159, places=4)
        self.assertAlmostEqual(vals[3], -0.5, places=2)

    def test_with_formatting(self):
        result_type, vals = self.clean("col", ["$1,000.50", "%99.9", "1,234.56"])
        self.assertEqual(result_type, "double")
        self.assertAlmostEqual(vals[0], 1000.50, places=2)
        self.assertAlmostEqual(vals[1], 99.9, places=1)
        self.assertAlmostEqual(vals[2], 1234.56, places=2)

    def test_with_nulls(self):
        result_type, vals = self.clean("col", ["1.5", None, "2.5", None])
        self.assertEqual(result_type, "double")
        self.assertAlmostEqual(vals[0], 1.5, places=2)
        self.assertIsNone(vals[1])
        self.assertAlmostEqual(vals[2], 2.5, places=2)
        self.assertIsNone(vals[3])

    def test_empty_strings_as_null(self):
        result_type, vals = self.clean("col", ["1.5", "", "2.5", "   ", None])
        self.assertEqual(result_type, "double")
        self.assertIsNone(vals[1])
        self.assertIsNone(vals[3])
        self.assertIsNone(vals[4])

    def test_negative_numbers(self):
        self.assert_type("col", ["-100", "-200.5", "-0.001"], "double")


class TestDateParsing(SparkTestCase):
    """Tests for date/timestamp type parsing."""

    def test_iso_format(self):
        self.assert_dates("col", ["2021-06-15", "2022-07-20", "2023-08-25"], [(2021, 6, 15), (2022, 7, 20), (2023, 8, 25)])

    def test_us_format(self):
        self.assert_dates("col", ["06/15/2021", "07/20/2022", "08/25/2023"], [(2021, 6, 15), (2022, 7, 20), (2023, 8, 25)])

    def test_with_time(self):
        self.assert_dates("col", ["2021-06-15 10:30:00", "2022-07-20 14:45:30"], [(2021, 6, 15), (2022, 7, 20)])

    def test_with_nulls(self):
        self.assert_dates("col", ["2021-06-15", None, "2022-07-20", None], [(2021, 6, 15), None, (2022, 7, 20), None])

    def test_empty_strings_as_null(self):
        self.assert_dates("col", ["2021-06-15", "", "2022-07-20", "   ", None], [(2021, 6, 15), None, (2022, 7, 20), None, None])

    def test_day_boundary_no_drift_utc(self):
        df = self.make_df("col", ["2023-12-31 23:59:59", "2023-06-30 23:59:59"])
        result = SparkCleaner.clean_all_types(df, source_timezone="UTC")
        self.assert_utc_timestamp(result, "col", 0, "2023-12-31 23:59:59")
        self.assert_utc_timestamp(result, "col", 1, "2023-06-30 23:59:59")

    def test_day_boundary_with_timezone_offset(self):
        df = self.make_df("col", ["2023-12-31T23:59:59-05:00"])
        result = SparkCleaner.clean_all_types(df)
        self.assert_utc_timestamp(result, "col", 0, "2024-01-01 04:59:59")

    def test_day_boundary_start_of_day(self):
        df = self.make_df("col", ["2023-01-01 00:00:00", "2023-06-15 00:00:01"])
        result = SparkCleaner.clean_all_types(df, source_timezone="UTC")
        self.assert_utc_timestamp(result, "col", 0, "2023-01-01 00:00:00")
        self.assert_utc_timestamp(result, "col", 1, "2023-06-15 00:00:01")


if __name__ == "__main__":
    import unittest
    unittest.main()
