# src/etl/database/extra_column_exception.py
"""Backwards-compatibility shim — the exception now lives in exceptions.py."""

from .exceptions import ExtraColumnsException

__all__ = ["ExtraColumnsException"]
