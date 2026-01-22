#!/usr/bin/env python3
"""Tests for Spark type conversion fallback behavior."""
import logging

from tests.conftest_spark import SparkTestCase
from src.etl.dataframe.spark.cleaner import SparkCleaner, SamplingConfig


class TestFallbackBehavior(SparkTestCase):
    """Tests for type conversion fallback chains."""

    def test_invalid_boolean_to_string(self):
        result_type, vals = self.clean("col", ["yes", "no", "invalid"])
        self.assertEqual(result_type, "string")
        self.assertEqual(vals, ["yes", "no", "invalid"])

    def test_0_1_2_triggers_integer(self):
        result_type, vals = self.clean("col", ["0", "1", "2"])
        self.assertEqual(result_type, "bigint")
        self.assertEqual(vals, [0, 1, 2])

    def test_0_1_negative_triggers_integer(self):
        result_type, vals = self.clean("col", ["0", "1", "-1"])
        self.assertEqual(result_type, "bigint")
        self.assertEqual(vals, [0, 1, -1])

    def test_integer_decimal_triggers_float(self):
        result_type, vals = self.clean("col", ["1", "2", "3.5"])
        self.assertEqual(result_type, "double")
        self.assertAlmostEqual(vals[0], 1.0, places=2)
        self.assertAlmostEqual(vals[1], 2.0, places=2)
        self.assertAlmostEqual(vals[2], 3.5, places=2)

    def test_integer_text_triggers_string(self):
        result_type, vals = self.clean("col", ["100", "200", "abc"])
        self.assertEqual(result_type, "string")
        self.assertEqual(vals, ["100", "200", "abc"])

    def test_float_text_triggers_string(self):
        result_type, vals = self.clean("col", ["1.5", "2.7", "abc"])
        self.assertEqual(result_type, "string")
        self.assertEqual(vals, ["1.5", "2.7", "abc"])

    def test_special_float_as_string(self):
        result_type, _ = self.clean("col", ["1.5", "2.7", "inf"])
        self.assertEqual(result_type, "string")

    def test_full_chain_boolean_to_string(self):
        result_type, vals = self.clean("col", ["0", "1", "two"])
        self.assertEqual(result_type, "string")
        self.assertEqual(vals, ["0", "1", "two"])

    def test_partial_chain_integer_to_float(self):
        result_type, _ = self.clean("col", ["10", "20", "30.5", "40"])
        self.assertEqual(result_type, "double")

    def test_nulls_preserved(self):
        result_type, vals = self.clean("col", ["0", None, "1", "2"])
        self.assertEqual(result_type, "bigint")
        self.assertEqual(vals, [0, None, 1, 2])

    def test_empty_strings_as_null(self):
        result_type, vals = self.clean("col", ["0", "", "1", "2"])
        self.assertEqual(result_type, "bigint")
        self.assertEqual(vals, [0, None, 1, 2])


class TestFallbackLogging(SparkTestCase):
    """Tests for fallback warning logging."""

    def test_logs_warning_on_fallback(self):
        """Fallback should log warnings when type conversion fails."""
        log_capture = []
        handler = logging.Handler()
        handler.emit = lambda record: log_capture.append(record)

        logger = logging.getLogger("src.etl.dataframe.spark.type_inference")
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(handler)

        try:
            # Create data where sampling may infer boolean but full data has "2"
            values = ["0", "1", "0", "1", "0", "1", "0", "1", "0", "2"]
            df = self.make_df("col", values)
            config = SamplingConfig(enabled=True, min_rows=5, fraction=0.5, seed=123)
            SparkCleaner.clean_all_types(df, sampling_config=config)

            warnings = [r.message for r in log_capture if r.levelno == logging.WARNING]
            fallback_warnings = [m for m in warnings if "fallback" in m.lower()]
            self.assertGreater(len(fallback_warnings), 0)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)


if __name__ == "__main__":
    import unittest
    unittest.main()
