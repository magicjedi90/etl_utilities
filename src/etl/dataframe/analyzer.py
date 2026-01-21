# src/etl/dataframe/analyzer.py
"""
Backwards compatibility module - imports from new location.

This module re-exports the Analyzer class from its new location
in the pandas package for backwards compatibility.
"""

from .pandas.analyzer import Analyzer

__all__ = ["Analyzer"]
