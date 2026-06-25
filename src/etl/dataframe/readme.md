# Data Processing Library

These modules provide tools for data analysis and cleaning. The `Analyzer`, `Cleaner`,
and `Parser` documented here are the **Pandas** backend (`etl.dataframe`), which is the
original and most complete implementation. Two parallel backends share the same API shape:

- **Polars** (`etl.dataframe.polars`) — expression-based, and the only backend with a
  batch-safe streaming API. See [Polars backend](#polars-backend) below.
- **Spark** (`etl.dataframe.spark`) — distributed, native Spark SQL with sampling-based
  inference. See [Spark backend](#spark-backend) below.

All backends follow the same type-narrowing order: **Boolean → Integer → Float → Date → String**.

## Table of Contents
- [Analyzer Module](#analyzer-module)
  - [find_unique_columns](#find_unique_columns)
  - [find_unique_column_pairs](#find_unique_column_pairs)
  - [find_empty_columns](#find_empty_columns)
  - [generate_column_metadata](#generate_column_metadata)
  - [find_categorical_columns](#find_categorical_columns)
- [Cleaner Module](#cleaner-module)
  - [column_names_to_snake_case](#column_names_to_snake_case)
  - [clean_series](#clean_series)
  - [clean_numbers](#clean_numbers)
  - [clean_df](#clean_df)
- [Parser Module](#parser-module)
  - [parse_boolean](#parse_boolean)
  - [parse_float](#parse_float)
  - [parse_date](#parse_date)
  - [parse_integer](#parse_integer)
- [Polars backend](#polars-backend)
- [Spark backend](#spark-backend)

## Analyzer Module

The `Analyzer` class provides methods for analyzing pandas DataFrames.

### `find_unique_columns`

Identifies columns in a DataFrame where all values are unique.

```python
import pandas as pd
from etl.dataframe.analyzer import Analyzer

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
unique_columns = Analyzer.find_unique_columns(df)
print(unique_columns)
```

### `find_unique_column_pairs`

Finds pairs of columns that, when combined, form a unique set of data.

```python
unique_column_pairs = Analyzer.find_unique_column_pairs(df)
print(unique_column_pairs)
```

### `find_empty_columns`

Returns a list of columns that contain only NaN or None values.

```python
empty_columns = Analyzer.find_empty_columns(df)
print(empty_columns)
```

### `generate_column_metadata`

Generates metadata for each column, including data type and uniqueness.

```python
primary_key = 'A'
unique_columns = ['A']
metadata = Analyzer.generate_column_metadata(df, primary_key, unique_columns, 2)
print(metadata)
```

### `find_categorical_columns`

Identifies columns that are considered categorical based on a uniqueness threshold.

```python
categorical_columns = Analyzer.find_categorical_columns(df, 0.5)
print(categorical_columns)
```

## Cleaner Module

The `Cleaner` class offers static methods to clean and preprocess data within a pandas DataFrame.

### `column_names_to_snake_case`

Converts DataFrame column names to snake_case.

```python
from etl.dataframe.cleaner import Cleaner

Cleaner.column_names_to_snake_case(df)
print(df.columns)
```

### `clean_series`

Cleans a pandas Series using a specified cleaning function.

```python
cleaned_series = Cleaner.clean_series(df['A'], Parser.parse_float)
print(cleaned_series)
```

### `clean_numbers`

Cleans numeric columns by parsing floats and integers.

```python
cleaned_df = Cleaner.clean_numbers(df)
print(cleaned_df)
```

### `clean_df`

Drops fully empty rows and columns, then cleans the DataFrame by types.

```python
cleaned_full_df = Cleaner.clean_df(df)
print(cleaned_full_df)
```

## Parser Module

The `Parser` class provides static parsing functions for converting values to specific data types.

### `parse_boolean`

Parses a value into a boolean based on common truthy and falsy strings.

```python
boolean_value = Parser.parse_boolean("yes")
print(boolean_value)
```

### `parse_float`

Converts a value into a float, removing common formatting symbols.

```python
float_value = Parser.parse_float("123.45")
print(float_value)
```

### `parse_date`

Parses a value into a datetime object.

```python
date_value = Parser.parse_date("2023-01-01")
print(date_value)
```

### `parse_integer`

Attempts to parse a value into an integer.

```python
int_value = Parser.parse_integer("123")
print(int_value)
```

## Polars backend

`etl.dataframe.polars` provides `PolarsCleaner` and `PolarsParser`. The one-shot methods
mirror the Pandas `Cleaner` — `column_names_to_snake_case`, `clean_numbers`, `clean_dates`,
`clean_bools`, `clean_all_types`, `clean_df`, `coalesce_columns`, `generate_hash_column`,
`optimize_dtypes` — but operate on `polars.DataFrame` using Polars expressions.

```python
import polars as pl
from etl.dataframe.polars.cleaner import PolarsCleaner

df = pl.read_csv("export.csv")
df = PolarsCleaner.column_names_to_snake_case(df)
df = PolarsCleaner.clean_df(df)            # one-shot: drop empty columns + infer types
```

### Batch-safe / streaming API

Per-frame inference is unstable across batches (batch A infers `Int8`, batch B infers
`String`, and the append fails). Split **inference** from **application** so every batch
lands on the same schema:

```python
plan = PolarsCleaner.infer_cleaning_plan(sample_df, prefer_float=True)
for batch in stream:
    batch = PolarsCleaner.apply_cleaning_plan(batch, plan)   # identical schema every time
    sink.write(batch)
```

- `infer_cleaning_plan(df, columns=None, threshold=1.0, prefer_float=False)` returns a
  `column -> dtype-kind` dict (`"boolean" | "int64" | "float64" | "datetime" | "utf8" | "keep"`).
- `apply_cleaning_plan(df, plan, optimize=False)` applies it deterministically (every cast is
  `strict=False`, so dirty values become null rather than raising).

### Schema-stability helpers

Useful right before a strict sink (Delta/Parquet/warehouse append):

- `sanitize_float_columns(df)` — NaN/±Infinity (native floats **and** `"NaN"`/`"Infinity"` string literals) → null.
- `localize_naive_datetimes(df, tz="UTC")` — naive `Datetime` columns → tz-aware (alias: `to_utc_datetimes`).
- `cast_null_columns(df, to=pl.Utf8)` — all-null `pl.Null` columns → a concrete dtype.
- `clean_df(df, drop_all_null_columns=False)` — keep all-null columns so the schema stays constant across files.
- `column_names_to_snake_case(df, on_collision="coalesce")` — handle names that normalize to the
  same snake_case (`"coalesce"` merges them, `"error"` raises, `"suffix"` disambiguates with `_2`, `_3`, …).
- `generate_hash_column(df, cols, name, algorithm="sha1")` — `"sha1"` (default, stable hex digest) or
  `"xxhash"` (vectorized `UInt64`, much faster for in-run dedup).

For reconciling mismatched Parquet schemas across many files, see
[`etl.io.arrow_schema.unify_parquet_schemas`](../io/arrow_schema.py).

## Spark backend

`etl.dataframe.spark` provides a distributed `Cleaner` built on native Spark SQL (no Python
UDFs, for performance), with a modular subsystem: `type_parsers`, `type_checkers`,
`type_inference`, `diagnostics`, and `config`. Inference is sampling-based with automatic
retry and type broadening, uses `DoubleType` (64-bit) for financial precision, and normalizes
all timestamps to UTC. The public cleaning surface mirrors the Pandas/Polars backends.