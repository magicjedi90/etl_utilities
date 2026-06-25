# src/etl/io/__init__.py
"""I/O helpers for etl_utilities (PyArrow-based schema reconciliation, etc.).

Requires the optional `pyarrow` dependency. Import lazily:

    from etl.io.arrow_schema import unify_parquet_schemas
"""

from .arrow_schema import unify_parquet_schemas

__all__ = [
    "unify_parquet_schemas",
]
