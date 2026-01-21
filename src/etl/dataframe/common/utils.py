# src/etl/dataframe/common/utils.py
"""Shared utility functions for all DataFrame implementations."""

import hashlib
import re


def compute_hash(value) -> str:
    """
    Compute the SHA-1 hash value of the given input value.

    Args:
        value: The input value to be hashed.

    Returns:
        The resulting hash value as a hexadecimal string.
    """
    return hashlib.sha1(str(value).encode()).hexdigest()


def standardize_column_name(name) -> str:
    """
    Standardize a column name to snake_case format.

    Removes special characters, replaces certain characters with meaningful
    alternatives, and converts to lowercase with underscores as separators.

    Args:
        name: The column name to be standardized.

    Returns:
        The standardized column name in snake_case.
    """
    name = (str(name).strip()
            .replace('?', '').replace('(', '').replace(')', '')
            .replace('\\', '').replace(',', '').replace('/', '')
            .replace('\'', '').replace('#', 'Num').replace('$', 'Dollars')
            .replace('&', 'And'))
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    return (name.replace('.', '_').replace(':', '_').replace(' ', '_')
            .replace('-', '_').replace('___', '_').replace('__', '_')
            .strip('_'))
