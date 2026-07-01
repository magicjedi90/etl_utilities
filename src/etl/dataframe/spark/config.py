# src/etl/dataframe/spark/config.py
"""Configuration constants and dataclasses for Spark type inference and cleaning."""

import dataclasses

from ..common.constants import (
    TRUTHY_VALUES,
    FALSY_VALUES,
    ALL_BOOLEAN_VALUES,
    SPARK_DATE_FORMATS,
)


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

# Common date formats to try (ordered by specificity) — the canonical set
# lives in common/constants.py, shared with the Polars backend.
DATE_FORMATS = list(SPARK_DATE_FORMATS)

# Regex pattern for numeric values (integer or float, with optional sign)
NUMERIC_PATTERN = r'^-?[0-9]+\.?[0-9]*$'
