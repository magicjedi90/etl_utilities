# src/etl/dataframe/parser.py
"""
Backwards compatibility module - imports from new location.

This module re-exports the Parser class from its new location
in the pandas package for backwards compatibility.
"""

from .pandas.parser import Parser

__all__ = ["Parser"]
