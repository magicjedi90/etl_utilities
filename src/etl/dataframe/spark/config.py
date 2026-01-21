# src/etl/dataframe/spark/constants.py
"""Configuration constants and dataclasses for Spark type inference and cleaning."""

import dataclasses

from ..common.constants import TRUTHY_VALUES, FALSY_VALUES, ALL_BOOLEAN_VALUES


@dataclasses.dataclass(slots=True, frozen=True)
class SamplingConfig:
    """Configuration for sampling-based type inference."""
    enabled: bool = True
    fraction: float = 0.1          # 10% sample
    min_rows: int = 1000           # Skip sampling if fewer rows
    max_rows: int = 100_000        # Cap sample size
    seed: int | None = None        # For reproducibility


# Type fallback hierarchy for retry logic when sampled type fails on full data
TYPE_FALLBACK_HIERARCHY: dict[str, list[str]] = {
    'boolean': ['integer', 'float', 'string'],
    'integer': ['float', 'string'],
    'float': ['string'],
    'datetime': ['string'],
    'string': [],
}

# Re-export for backwards compatibility
__all__ = [
    "SamplingConfig",
    "TYPE_FALLBACK_HIERARCHY",
    "TRUTHY_VALUES",
    "FALSY_VALUES",
    "ALL_BOOLEAN_VALUES",
]

# Common date formats to try (ordered by specificity)
DATE_FORMATS = [
    "yyyy-MM-dd'T'HH:mm:ss.SSSXXX",
    "yyyy-MM-dd'T'HH:mm:ss.SSS",
    "yyyy-MM-dd'T'HH:mm:ssXXX",
    "yyyy-MM-dd'T'HH:mm:ss",
    "yyyy-MM-dd HH:mm:ss.SSS",
    "yyyy-MM-dd HH:mm:ss",
    "yyyy-MM-dd",
    "MM/dd/yyyy HH:mm:ss",
    "MM/dd/yyyy",
    "MM-dd-yyyy",
    "dd/MM/yyyy",
    "dd-MM-yyyy",
    "yyyy/MM/dd",
    "yyyyMMdd",
    "MMM dd, yyyy",
    "dd MMM yyyy",
    "MMMM dd, yyyy",
]

# Regex pattern for numeric values (integer or float, with optional sign)
NUMERIC_PATTERN = r'^-?[0-9]+\.?[0-9]*$'
