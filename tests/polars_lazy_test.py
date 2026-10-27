#!/usr/bin/env python3
"""
LazyFrame support for the Polars cleaner: every method must accept a LazyFrame,
return a LazyFrame, and collect to exactly what the eager call returns — while
pure transforms run no query and data-driven methods run exactly one.
"""
import pytest
import polars as pl
from polars.testing import assert_frame_equal

from src.etl.dataframe.polars import cleaning_plan, schema_tools, type_cleaning
from src.etl.dataframe.polars.cleaner import PolarsCleaner

# Resolving a LazyFrame schema via .schema/.columns/.dtypes warns; the cleaner
# must only use collect_schema().
pytestmark = pytest.mark.filterwarnings("error::polars.exceptions.PerformanceWarning")


def _sample() -> pl.DataFrame:
    return pl.DataFrame({
        'Customer Name': ['John Doe', 'Jane Smith', 'Bob Johnson', None, 'Alice Brown', None],
        'Age': ['25.0', '30.0', '35', '', None, None],
        'Salary': ['$50,000', '65000', '75000.50', '$80,000', '', None],
        'Is Active': ['Yes', 'No', '1', '0', 'true', None],
        'Join Date': ['2023-01-15', '2023/02/20', 'March 15, 2023', '2023-04-10', None, None],
        'Performance Score': ['85%', '92.5%', '78', '88.5%', None, None],
        'Visits': [3, 40, 500, None, 7, None],          # already Int64 -> narrowed
        'Notes': [None, None, None, None, None, None],  # all-null column
        'Ratio': [0.5, float('nan'), float('inf'), 1.0, None, None],
        'Raw Ratio': ['nan', '1', 'Infinity', 'x', None, None],
    })  # last row is all-null


@pytest.fixture
def lazy_queries(monkeypatch):
    """Count LazyFrame executions made through collect_row (eager calls ignored)."""
    calls = []
    real = type_cleaning.collect_row

    def counting(frame, exprs):
        if isinstance(frame, pl.LazyFrame):
            calls.append(frame)
        return real(frame, exprs)

    for module in (cleaning_plan, schema_tools, type_cleaning):
        monkeypatch.setattr(module, "collect_row", counting)
    return calls


C = PolarsCleaner
CASES = [
    # (callable, LazyFrame queries it may run)
    pytest.param(lambda f: C.column_names_to_snake_case(f), 0, id="snake_case"),
    pytest.param(lambda f: C.column_names_to_pascal_case(f), 0, id="pascal_case"),
    pytest.param(lambda f: C.clean_numbers(f), 1, id="clean_numbers_auto"),
    pytest.param(lambda f: C.clean_numbers(f, ['Age', 'Salary']), 0, id="clean_numbers_named"),
    pytest.param(lambda f: C.clean_dates(f), 1, id="clean_dates_auto"),
    pytest.param(lambda f: C.clean_dates(f, ['Join Date']), 0, id="clean_dates_named"),
    pytest.param(lambda f: C.clean_bools(f), 1, id="clean_bools_auto"),
    pytest.param(lambda f: C.clean_bools(f, ['Is Active']), 0, id="clean_bools_named"),
    pytest.param(lambda f: C.clean_all_types(f), 1, id="clean_all_types"),
    pytest.param(lambda f: C.clean_all_types(f, columns=['Age', 'Visits']), 1, id="clean_all_types_subset"),
    pytest.param(
        lambda f: C.clean_all_types(f, type_overrides={'Age': pl.Int32, 'Visits': pl.Int64, 'Missing': pl.String}),
        1, id="clean_all_types_overrides",
    ),
    pytest.param(lambda f: C.clean_df(f), 1, id="clean_df"),
    pytest.param(lambda f: C.clean_df(f, drop_all_null_columns=False), 1, id="clean_df_keep_null_columns"),
    pytest.param(
        lambda f: C.clean_df(f, type_overrides={'Salary': pl.String, 'Notes': pl.Float64}),
        1, id="clean_df_overrides",
    ),
    pytest.param(lambda f: C.apply_cleaning_plan(f, C.infer_cleaning_plan(_sample())), 0, id="apply_plan"),
    pytest.param(
        lambda f: C.apply_cleaning_plan(f, C.infer_cleaning_plan(_sample()), optimize=True),
        1, id="apply_plan_optimize",
    ),
    pytest.param(lambda f: C.optimize_dtypes(f), 1, id="optimize_dtypes"),
    pytest.param(lambda f: C.sanitize_float_columns(f), 0, id="sanitize_float_columns"),
    pytest.param(lambda f: C.localize_naive_datetimes(C.clean_dates(f, ['Join Date'])), 0, id="localize_naive_datetimes"),
    pytest.param(lambda f: C.cast_null_columns(f), 0, id="cast_null_columns"),
    pytest.param(lambda f: C.coalesce_columns(f, ['Age', 'Salary'], 'Age', drop=True), 0, id="coalesce_columns"),
    pytest.param(lambda f: C.generate_hash_column(f, ['Customer Name', 'Age'], 'h'), 0, id="hash_sha1"),
    pytest.param(
        lambda f: C.generate_hash_column(f, ['Customer Name', 'Age'], 'h', algorithm="xxhash"),
        0, id="hash_xxhash",
    ),
]


class TestLazyParity:
    """Each method on a LazyFrame stays lazy and matches the eager result."""

    @pytest.mark.parametrize("fn, expected_queries", CASES)
    def test_lazy_matches_eager(self, fn, expected_queries, lazy_queries):
        eager = fn(_sample())
        lazy = fn(_sample().lazy())

        assert isinstance(lazy, pl.LazyFrame)
        assert len(lazy_queries) == expected_queries
        assert_frame_equal(lazy.collect(), eager)

    @pytest.mark.parametrize("on_collision", ["coalesce", "suffix"])
    def test_snake_case_collisions(self, on_collision, lazy_queries):
        df = pl.DataFrame({'NUSC Delta': [1, None], 'NUSCDelta': [None, 2], 'Other': ['a', 'b']})
        lazy = C.column_names_to_snake_case(df.lazy(), on_collision=on_collision)

        assert isinstance(lazy, pl.LazyFrame)
        assert not lazy_queries
        assert_frame_equal(lazy.collect(), C.column_names_to_snake_case(df, on_collision=on_collision))

    def test_snake_case_collision_error(self):
        df = pl.DataFrame({'NUSC Delta': [1], 'NUSCDelta': [2]})
        with pytest.raises(ValueError, match="collisions"):
            C.column_names_to_snake_case(df.lazy(), on_collision="error")

    def test_infer_cleaning_plan_matches_eager(self, lazy_queries):
        assert C.infer_cleaning_plan(_sample().lazy()) == C.infer_cleaning_plan(_sample())
        assert len(lazy_queries) == 1

    def test_clean_df_all_null_frame(self):
        df = pl.DataFrame({'a': [None, None], 'b': [None, None]})
        lazy = C.clean_df(df.lazy())
        assert isinstance(lazy, pl.LazyFrame)
        assert lazy.collect().shape == (0, 0)

    def test_hash_missing_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            C.generate_hash_column(_sample().lazy(), ['nope'], 'h')

    def test_named_missing_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            C.clean_numbers(_sample().lazy(), ['nope'])


class TestScanToSink:
    """The use case LazyFrame support exists for: scan -> clean -> sink."""

    def test_clean_df_scan_parquet_to_sink(self, tmp_path):
        source, target = tmp_path / "in.parquet", tmp_path / "out.parquet"
        _sample().write_parquet(source)

        lf = C.column_names_to_snake_case(pl.scan_parquet(source))
        C.clean_df(lf).sink_parquet(target)

        expected = C.clean_df(C.column_names_to_snake_case(pl.read_parquet(source)))
        assert_frame_equal(pl.read_parquet(target), expected)

    def test_plan_from_scan_applies_to_batches(self, tmp_path):
        """infer once over a scanned source, apply lazily to per-file scans."""
        df = _sample()
        paths = []
        for i, part in enumerate((df.slice(0, 3), df.slice(3, 3))):
            paths.append(tmp_path / f"part{i}.parquet")
            part.write_parquet(paths[-1])

        plan = C.infer_cleaning_plan(pl.scan_parquet(paths), prefer_float=True)
        batches = [C.apply_cleaning_plan(pl.scan_parquet(p), plan).collect() for p in paths]

        assert batches[0].schema == batches[1].schema
        assert_frame_equal(pl.concat(batches), C.apply_cleaning_plan(df, plan))
