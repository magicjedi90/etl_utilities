#!/usr/bin/env python3
"""
Tests for the data-platform porting additions to PolarsCleaner:
batch-safe plan/apply, collision-safe snake_case, float sanitization,
datetime localization, null-column casting, the clean_df keep-nulls flag,
and the xxhash option on generate_hash_column.
"""
import datetime

import pytest
import polars as pl

from src.etl.dataframe.polars.cleaner import PolarsCleaner


class TestInferApplyCleaningPlan:
    """A1: infer_cleaning_plan + apply_cleaning_plan are batch-stable."""

    def test_infer_apply_is_batch_stable(self):
        # Infer once on a representative frame, then apply to two divergent batches:
        # both must land on the SAME schema.
        sample = pl.DataFrame({"flag": ["1", "0", "1"], "n": ["1", "2", "3"]})
        plan = PolarsCleaner.infer_cleaning_plan(sample, prefer_float=True)
        assert plan == {"flag": "boolean", "n": "float64"}

        a = PolarsCleaner.apply_cleaning_plan(pl.DataFrame({"flag": ["1"], "n": ["5"]}), plan)
        b = PolarsCleaner.apply_cleaning_plan(pl.DataFrame({"flag": ["0"], "n": ["7"]}), plan)
        assert a.schema == b.schema
        assert a.schema["flag"] == pl.Boolean
        assert a.schema["n"] == pl.Float64
        assert a["n"].to_list() == [5.0]
        assert b["flag"].to_list() == [False]

    def test_infer_plans_int64_for_whole_numbers(self):
        plan = PolarsCleaner.infer_cleaning_plan(pl.DataFrame({"n": ["1", "2", "3"]}))
        assert plan["n"] == "int64"
        out = PolarsCleaner.apply_cleaning_plan(pl.DataFrame({"n": ["10", "20"]}), plan)
        assert out.schema["n"] == pl.Int64
        assert out["n"].to_list() == [10, 20]

    def test_infer_plans_float64_for_fractional_numbers(self):
        plan = PolarsCleaner.infer_cleaning_plan(pl.DataFrame({"n": ["1.5", "2", "3"]}))
        assert plan["n"] == "float64"

    def test_infer_keeps_uncoverable_and_empty_columns(self):
        df = pl.DataFrame({"mixed": ["1", "abc", "3"], "empty": [None, None, None]})
        plan = PolarsCleaner.infer_cleaning_plan(df)
        assert plan["mixed"] == "keep"   # parser does not cover all values
        assert plan["empty"] == "keep"   # all-null

    def test_apply_optimize_downcasts_only_when_requested(self):
        plan = {"n": "int64"}
        df = pl.DataFrame({"n": ["1", "2", "3"]})
        assert PolarsCleaner.apply_cleaning_plan(df, plan, optimize=False).schema["n"] == pl.Int64
        assert PolarsCleaner.apply_cleaning_plan(df, plan, optimize=True).schema["n"] == pl.UInt8

    def test_apply_ignores_columns_absent_from_batch(self):
        plan = {"a": "int64", "missing": "float64"}
        out = PolarsCleaner.apply_cleaning_plan(pl.DataFrame({"a": ["1"]}), plan)
        assert out.columns == ["a"]
        assert out.schema["a"] == pl.Int64


class TestSnakeCaseCollisions:
    """A2: collision-safe column_names_to_snake_case."""

    def test_no_collision_keeps_simple_rename(self):
        df = pl.DataFrame({"Customer Name": [1], "Age": [2]})
        out = PolarsCleaner.column_names_to_snake_case(df)
        assert out.columns == ["customer_name", "age"]

    def test_collision_coalesce(self):
        df = pl.DataFrame({"NUSC Delta": [None, 2], "NUSCDelta": [1, None]})
        out = PolarsCleaner.column_names_to_snake_case(df)  # default coalesce
        assert out.columns == ["nusc_delta"]
        assert out["nusc_delta"].to_list() == [1, 2]

    def test_collision_error_and_suffix(self):
        df = pl.DataFrame({"A B": [1], "a_b": [2]})
        with pytest.raises(ValueError):
            PolarsCleaner.column_names_to_snake_case(df, on_collision="error")
        suffixed = PolarsCleaner.column_names_to_snake_case(df, on_collision="suffix")
        assert set(suffixed.columns) == {"a_b", "a_b_2"}


class TestSanitizeFloatColumns:
    """A3: NaN/+-Infinity (native and string literal) become null."""

    def test_native_and_string_forms(self):
        df = pl.DataFrame({
            "f": [1.0, float("nan"), float("inf")],
            "s": ["NaN", "Infinity", "real text"],
        })
        out = PolarsCleaner.sanitize_float_columns(df)
        assert out["f"].to_list() == [1.0, None, None]
        assert out["s"].to_list() == [None, None, "real text"]

    def test_preserves_genuine_nulls_and_negatives(self):
        df = pl.DataFrame({"f": [-2.5, None, 3.0]})
        out = PolarsCleaner.sanitize_float_columns(df)
        assert out["f"].to_list() == [-2.5, None, 3.0]


class TestLocalizeNaiveDatetimes:
    """A4: naive Datetime columns become tz-aware; tz-aware untouched."""

    def test_localizes_naive_to_utc(self):
        df = pl.DataFrame({"ts": [datetime.datetime(2026, 1, 1)]})
        assert df.schema["ts"].time_zone is None
        out = PolarsCleaner.localize_naive_datetimes(df)
        assert out.schema["ts"].time_zone == "UTC"

    def test_custom_tz_and_alias(self):
        df = pl.DataFrame({"ts": [datetime.datetime(2026, 1, 1)]})
        out = PolarsCleaner.to_utc_datetimes(df, tz="America/New_York")
        assert out.schema["ts"].time_zone == "America/New_York"

    def test_leaves_non_datetime_columns_alone(self):
        df = pl.DataFrame({"x": [1, 2]})
        out = PolarsCleaner.localize_naive_datetimes(df)
        assert out.schema["x"] == pl.Int64


class TestCastNullColumns:
    """A5a: pl.Null dtype columns get a concrete dtype."""

    def test_casts_null_dtype_to_utf8(self):
        df = pl.DataFrame({"x": [None, None]})
        assert df.schema["x"] == pl.Null
        out = PolarsCleaner.cast_null_columns(df)
        assert out.schema["x"] == pl.Utf8

    def test_custom_target_dtype(self):
        df = pl.DataFrame({"x": [None, None]})
        out = PolarsCleaner.cast_null_columns(df, to=pl.Float64)
        assert out.schema["x"] == pl.Float64


class TestCleanDfKeepNulls:
    """A5b: clean_df(drop_all_null_columns=False) keeps all-null columns."""

    def test_keeps_all_null_columns_when_asked(self):
        df = pl.DataFrame({"keep_me": [None, None], "v": ["1", "2"]})
        dropped = PolarsCleaner.clean_df(df)                              # default drops
        kept = PolarsCleaner.clean_df(df, drop_all_null_columns=False)    # opt-out keeps
        assert "keep_me" not in dropped.columns
        assert "keep_me" in kept.columns


class TestGenerateHashAlgorithms:
    """A6: sha1 default preserved; xxhash opt-in is vectorized UInt64."""

    def test_sha1_default_unchanged(self):
        df = pl.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
        out = PolarsCleaner.generate_hash_column(df, ["a", "b"], "h")
        assert out.schema["h"] == pl.Utf8
        assert len(out["h"][0]) == 40
        assert out["h"][0] != out["h"][1]

    def test_xxhash_is_uint64(self):
        df = pl.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
        out = PolarsCleaner.generate_hash_column(df, ["a", "b"], "h", algorithm="xxhash")
        assert out.schema["h"] == pl.UInt64
        assert out["h"][0] != out["h"][1]

    def test_unknown_algorithm_raises(self):
        df = pl.DataFrame({"a": ["x"]})
        with pytest.raises(ValueError):
            PolarsCleaner.generate_hash_column(df, ["a"], "h", algorithm="md5")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
