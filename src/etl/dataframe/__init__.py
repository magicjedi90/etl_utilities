# src/etl/dataframe/__init__.py
"""DataFrame utilities for ETL operations.

This package provides DataFrame cleaning, parsing, and analysis utilities
for multiple DataFrame libraries (Pandas, Polars, Spark).

Subpackages:
    - common: Shared utilities and constants
    - pandas: Pandas-specific implementations
    - polars: Polars-specific implementations
    - spark: Spark-specific implementations

For backwards compatibility, the base-level modules (cleaner, parser, analyzer)
re-export from the pandas package.
"""

# Re-export for backwards compatibility
from .cleaner import Cleaner, standardize_column_name, compute_hash
from .parser import Parser
from .analyzer import Analyzer

__all__ = [
    "Cleaner",
    "Parser",
    "Analyzer",
    "standardize_column_name",
    "compute_hash",
]
