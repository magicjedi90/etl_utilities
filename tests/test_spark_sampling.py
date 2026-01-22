#!/usr/bin/env python3
"""Tests for Spark DataFrame sampling configuration and fallback behavior."""
from pyspark.sql.types import StructType, StructField, StringType

from tests.conftest_spark import SparkTestCase
from src.etl.dataframe.spark.cleaner import SparkCleaner, SamplingConfig


class TestSamplingConfig(SparkTestCase):
    """Tests for SamplingConfig dataclass."""

    def test_defaults(self):
        config = SamplingConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.fraction, 0.1)
        self.assertEqual(config.min_rows, 1000)
        self.assertEqual(config.max_rows, 100_000)
        self.assertIsNone(config.seed)

    def test_custom_values(self):
        config = SamplingConfig(enabled=False, fraction=0.5, min_rows=500, max_rows=50_000, seed=42)
        self.assertFalse(config.enabled)
        self.assertEqual(config.fraction, 0.5)
        self.assertEqual(config.min_rows, 500)
        self.assertEqual(config.max_rows, 50_000)
        self.assertEqual(config.seed, 42)


class TestSamplingBehavior(SparkTestCase):
    """Tests for sampling-based type inference."""

    def _make_values_df(self, values: list[str]) -> "DataFrame":
        data = [(v,) for v in values]
        schema = StructType([StructField("col", StringType(), True)])
        return self.spark.createDataFrame(data, schema)

    def test_skipped_for_small_dataframe(self):
        df = self._make_values_df([str(i) for i in range(100)])
        config = SamplingConfig(enabled=True, min_rows=1000)
        result = SparkCleaner.clean_all_types(df, sampling_config=config)
        self.assertEqual(result.schema["col"].dataType.simpleString(), "bigint")

    def test_disabled_uses_full_scan(self):
        df = self._make_values_df([str(i) for i in range(100)])
        config = SamplingConfig(enabled=False)
        result = SparkCleaner.clean_all_types(df, sampling_config=config)
        self.assertEqual(result.schema["col"].dataType.simpleString(), "bigint")

    def test_integer_fallback_to_float(self):
        values = [str(i) for i in range(99)] + ["99.5"]
        df = self._make_values_df(values)
        config = SamplingConfig(enabled=True, min_rows=10, fraction=0.5, seed=42)
        result = SparkCleaner.clean_all_types(df, sampling_config=config)
        self.assertEqual(result.schema["col"].dataType.simpleString(), "double")

    def test_boolean_fallback_to_integer(self):
        values = ["0", "1", "0", "1", "0", "1", "0", "1", "0", "2"]
        df = self._make_values_df(values)
        config = SamplingConfig(enabled=True, min_rows=5, fraction=0.5, seed=123)
        result = SparkCleaner.clean_all_types(df, sampling_config=config)
        self.assertIn(result.schema["col"].dataType.simpleString(), ["bigint", "double"])

    def test_multiple_fallback_levels(self):
        values = ["0", "1", "0", "1", "0", "1", "0", "1", "0", "abc"]
        df = self._make_values_df(values)
        config = SamplingConfig(enabled=True, min_rows=5, fraction=0.5, seed=999)
        result = SparkCleaner.clean_all_types(df, sampling_config=config)
        self.assertEqual(result.schema["col"].dataType.simpleString(), "string")

    def test_no_fallback_on_clean_data(self):
        df = self._make_values_df([str(i) for i in range(100)])
        config = SamplingConfig(enabled=True, min_rows=10, fraction=0.3, seed=42)
        result = SparkCleaner.clean_all_types(df, sampling_config=config)
        self.assertEqual(result.schema["col"].dataType.simpleString(), "bigint")

    def test_seed_reproducibility(self):
        values = [str(i) for i in range(100)]
        config = SamplingConfig(enabled=True, min_rows=10, fraction=0.3, seed=42)

        df1 = self._make_values_df(values)
        result1 = SparkCleaner.clean_all_types(df1, sampling_config=config)

        df2 = self._make_values_df(values)
        result2 = SparkCleaner.clean_all_types(df2, sampling_config=config)

        self.assertEqual(
            result1.schema["col"].dataType.simpleString(),
            result2.schema["col"].dataType.simpleString()
        )

    def test_datetime_fallback_to_string(self):
        values = ["2021-01-01", "2021-02-02", "2021-03-03", "not-a-date"]
        df = self._make_values_df(values)
        config = SamplingConfig(enabled=True, min_rows=2, fraction=0.5, seed=42)
        result = SparkCleaner.clean_all_types(df, sampling_config=config)
        self.assertEqual(result.schema["col"].dataType.simpleString(), "string")

    def test_clean_df_passes_config(self):
        df = self.make_multi_df({"int_col": ["100", "200", "300"], "str_col": ["a", "b", "c"]})
        config = SamplingConfig(enabled=False)
        result = SparkCleaner.clean_df(df, sampling_config=config)
        self.assertEqual(result.schema["int_col"].dataType.simpleString(), "bigint")
        self.assertEqual(result.schema["str_col"].dataType.simpleString(), "string")

    def test_max_rows_cap(self):
        df = self._make_values_df([str(i) for i in range(200)])
        config = SamplingConfig(enabled=True, min_rows=10, fraction=0.5, max_rows=50, seed=42)
        result = SparkCleaner.clean_all_types(df, sampling_config=config)
        self.assertEqual(result.schema["col"].dataType.simpleString(), "bigint")

    def test_invalid_cast_falls_back_gracefully(self):
        """Regression: invalid strings should fallback to string, not throw CAST_INVALID_INPUT."""
        values = ["100", "200", "300", "400", "500", "600", "700", "800", "900", "abc"]
        df = self._make_values_df(values)
        config = SamplingConfig(enabled=True, min_rows=5, fraction=0.5, seed=42)
        result = SparkCleaner.clean_all_types(df, sampling_config=config)
        self.assertEqual(result.schema["col"].dataType.simpleString(), "string")
        self.assertEqual(result.filter(result.col == "abc").count(), 1)


if __name__ == "__main__":
    import unittest
    unittest.main()
