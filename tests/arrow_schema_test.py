#!/usr/bin/env python3
"""
Tests for etl.io.arrow_schema.unify_parquet_schemas — cross-file Parquet
schema reconciliation (Module B of the data-platform port).
"""
from decimal import Decimal

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
ds = pytest.importorskip("pyarrow.dataset")

from src.etl.io.arrow_schema import unify_parquet_schemas


def _write_parquet(path, table):
    pq.write_table(table, path)


def test_unify_reconciles_decimal_and_conflicting_columns(tmp_path):
    # File 1: amount + mixed both decimal, with different decimal scales than file 2.
    t1 = pa.table({
        "amount": pa.array([Decimal("1.5")], type=pa.decimal128(38, 12)),
        "mixed": pa.array([Decimal("2.5")], type=pa.decimal128(38, 12)),
    })
    # File 2: amount is a differently-scaled decimal; mixed is a string.
    t2 = pa.table({
        "amount": pa.array([Decimal("9.25")], type=pa.decimal128(38, 13)),
        "mixed": pa.array(["hello"], type=pa.large_string()),
    })

    p1 = str(tmp_path / "f1.parquet")
    p2 = str(tmp_path / "f2.parquet")
    _write_parquet(p1, t1)
    _write_parquet(p2, t2)

    unified = unify_parquet_schemas([p1, p2])

    # decimal-only column -> float64; decimal-vs-string conflict -> large_string
    assert unified.field("amount").type == pa.float64()
    assert unified.field("mixed").type == pa.large_string()

    # And PyArrow actually casts on read with the unified schema (no errors).
    table = ds.dataset([p1, p2], schema=unified).to_table()
    assert table.num_rows == 2
    assert set(table.column_names) == {"amount", "mixed"}
    assert table.schema.field("amount").type == pa.float64()
    assert table.schema.field("mixed").type == pa.large_string()


def test_unify_preserves_single_consistent_type(tmp_path):
    t1 = pa.table({"id": pa.array([1, 2], type=pa.int64())})
    t2 = pa.table({"id": pa.array([3, 4], type=pa.int64())})
    p1 = str(tmp_path / "a.parquet")
    p2 = str(tmp_path / "b.parquet")
    _write_parquet(p1, t1)
    _write_parquet(p2, t2)

    unified = unify_parquet_schemas([p1, p2])
    assert unified.field("id").type == pa.int64()


def test_unify_skips_unreadable_files(tmp_path):
    good = str(tmp_path / "good.parquet")
    _write_parquet(good, pa.table({"x": pa.array([1], type=pa.int64())}))
    missing = str(tmp_path / "does_not_exist.parquet")

    unified = unify_parquet_schemas([good, missing])
    assert unified.field("x").type == pa.int64()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
