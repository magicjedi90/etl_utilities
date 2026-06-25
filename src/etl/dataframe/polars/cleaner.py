# src/etl/dataframe/polars/cleaner.py
"""PolarsCleaner — the public facade over the Polars cleaning helpers.

The implementations live in focused, independently-importable modules:
    column_names.py   — snake_case / PascalCase normalization
    type_cleaning.py  — clean_numbers/dates/bools, clean_all_types, clean_df
    cleaning_plan.py  — batch-safe infer_cleaning_plan / apply_cleaning_plan
    schema_tools.py   — optimize_dtypes, sanitize_float_columns,
                        localize_naive_datetimes, cast_null_columns
    column_ops.py     — coalesce_columns, generate_hash_column

`PolarsCleaner` re-exposes them as static methods so the historical
``PolarsCleaner.clean_df(df)`` style API keeps working unchanged. New code may
call the module functions directly.
"""
from .cleaning_plan import apply_cleaning_plan, infer_cleaning_plan
from .column_names import column_names_to_pascal_case, column_names_to_snake_case
from .column_ops import coalesce_columns, generate_hash_column
from .schema_tools import (
    cast_null_columns,
    localize_naive_datetimes,
    optimize_dtypes,
    sanitize_float_columns,
)
from .type_cleaning import (
    clean_all_types,
    clean_bools,
    clean_dates,
    clean_df,
    clean_numbers,
)


class PolarsCleaner:
    """
    Static-method facade for data cleaning operations on a Polars DataFrame.
    Optimized for Polars' lazy evaluation and expression system. Each method
    delegates to a function in one of the sibling modules listed above.
    """

    # column-name normalization
    column_names_to_snake_case = staticmethod(column_names_to_snake_case)
    column_names_to_pascal_case = staticmethod(column_names_to_pascal_case)

    # value cleaning + one-shot type inference
    clean_numbers = staticmethod(clean_numbers)
    clean_dates = staticmethod(clean_dates)
    clean_bools = staticmethod(clean_bools)
    clean_all_types = staticmethod(clean_all_types)
    clean_df = staticmethod(clean_df)

    # batch-safe / streaming cleaning
    infer_cleaning_plan = staticmethod(infer_cleaning_plan)
    apply_cleaning_plan = staticmethod(apply_cleaning_plan)

    # dtype / schema stabilizers
    optimize_dtypes = staticmethod(optimize_dtypes)
    sanitize_float_columns = staticmethod(sanitize_float_columns)
    localize_naive_datetimes = staticmethod(localize_naive_datetimes)
    to_utc_datetimes = staticmethod(localize_naive_datetimes)  # back-compat alias
    cast_null_columns = staticmethod(cast_null_columns)

    # column-deriving ops
    coalesce_columns = staticmethod(coalesce_columns)
    generate_hash_column = staticmethod(generate_hash_column)
