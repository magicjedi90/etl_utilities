# src/etl/io/arrow_schema.py
"""Reconcile incompatible Parquet schemas across many files into one castable schema.

A pile of Parquet files with the same logical columns but incompatible physical types
(``Decimal(38,12)`` vs ``Decimal(38,13)``, string-vs-decimal) defeats the usual readers.

THINGS THAT DO NOT WORK (learned the hard way; do not retry):
  - pl.scan_parquet([all_files]) ............ "Decimal(38,12) != Decimal(38,13)"
  - scan_parquet(..., schema=) .............. declares & rejects, does not cast
  - ScanCastOptions / cast_options= ......... int/float upcast only, not Decimal
  - per-schema-group scan + streaming collect  1000s of engine nodes -> OOM
  - pa.unify_schemas(promote="permissive") .. "large_string vs decimal128 incompatible"
  - ds.dataset(paths) without schema= ....... "Rescaling Decimal would cause data loss"
  - scan_pyarrow_dataset + collect(streaming)  materializes everything -> OOM

WHAT WORKS: read footers only (pq.read_schema, ~ms each, concurrent), reconcile to a
unified schema (any decimal -> float64; any cross-file type conflict -> large_string),
then ``ds.dataset(paths, schema=unified)`` — PyArrow *actually casts* on read. Iterate
fragments -> to_batches() for memory-safe streaming.
"""
import logging
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)


def unify_parquet_schemas(paths, filesystem=None, max_workers: int = 4) -> pa.Schema:
    """Build one PyArrow schema that every file in ``paths`` can be cast to.

    Rules: a column seen as decimal in any file (and never as string) -> float64;
    a column with conflicting types across files (e.g. string vs int vs decimal)
    -> large_string; otherwise its single observed type. Reads footers only.
    Pass the result as ``ds.dataset(paths, schema=...)`` to cast on read.

    :param paths: iterable of Parquet file paths.
    :param filesystem: optional PyArrow filesystem (e.g. for object storage).
    :param max_workers: thread pool size for concurrent footer reads.
    :return: a unified ``pa.Schema``.
    """
    def _read_schema(path):
        try:
            return pq.read_schema(path, filesystem=filesystem)
        except Exception as e:  # unreadable file: skip, don't fail the whole run
            logger.warning(f"  Skipping unreadable file {path}: {type(e).__name__}: {e}")
            return None

    column_types: dict[str, set] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for schema in ex.map(_read_schema, paths):
            if schema is None:
                continue
            for field in schema:
                column_types.setdefault(field.name, set()).add(field.type)

    fields = []
    for name, types in column_types.items():
        has_string = any(pa.types.is_string(t) or pa.types.is_large_string(t) for t in types)
        has_decimal = any(pa.types.is_decimal(t) for t in types)
        if len(types) == 1 and not has_decimal:
            fields.append(pa.field(name, next(iter(types))))
        elif has_decimal and not has_string:
            fields.append(pa.field(name, pa.float64()))
        else:
            fields.append(pa.field(name, pa.large_string()))
    return pa.schema(fields)
