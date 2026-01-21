# src/etl/dataframe/cleaner.py
"""
Backwards compatibility module - imports from new locations.

This module re-exports the Cleaner class and utility functions from their
new locations in the pandas and common packages for backwards compatibility.
"""

# Re-export shared utilities from common
from .common.utils import standardize_column_name, compute_hash

# Re-export Cleaner and Parser from pandas package (Parser was historically accessible here)
from .pandas.cleaner import Cleaner
from .pandas.parser import Parser

__all__ = [
    "Cleaner",
    "Parser",
    "standardize_column_name",
    "compute_hash",
]
