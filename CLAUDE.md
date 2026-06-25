# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**etl_utilities** is a Python ETL framework providing data cleaning, type parsing, SQL generation, and database loading across multiple DataFrame libraries (Pandas, Polars, Spark) and database backends (MSSQL, MySQL/MariaDB, PostgreSQL). Published to PyPI as `etl_utilities`.

## Build & Development Commands

```bash
# Install dependencies (uses uv for package management)
uv sync

# Build the package
hatch build

# Run all tests
pytest tests/

# Run a single test file
pytest tests/parser_test.py

# Run a specific test
pytest tests/parser_test.py::TestParser::test_parse_boolean -v

# Type checking
hatch run types:check
```

**Build system:** Hatchling. Version is defined in `src/etl/__about__.py` and read dynamically by `pyproject.toml`.

**CI/CD:** GitHub Actions publishes to PyPI on every push to `main` via Trusted Publishing (no API tokens). Bump the version in `__about__.py` before merging.

## Architecture

### Source Layout

All source code lives under `src/etl/`. The package is importable as `etl`.

### Three DataFrame Backends

`src/etl/dataframe/` has parallel implementations for each backend:
- **`pandas/`** — Eager evaluation, the original and most complete implementation
- **`polars/`** — Expression-based, uses Polars lazy evaluation patterns. `PolarsCleaner` (`cleaner.py`) is a thin static-method **facade** over focused modules: `column_names.py` (snake/Pascal case), `type_cleaning.py` (`clean_numbers/dates/bools`, `clean_all_types`, `clean_df` — the one-shot in-memory path), `cleaning_plan.py` (the **batch-safe streaming API**: `infer_cleaning_plan` decides a deterministic `column -> dtype-kind` plan once, `apply_cleaning_plan` applies it identically to every batch), `schema_tools.py` (`optimize_dtypes`, `sanitize_float_columns`, `localize_naive_datetimes`/`to_utc_datetimes`, `cast_null_columns`), and `column_ops.py` (`coalesce_columns`, `generate_hash_column`). New code may call the module functions directly; the facade preserves the historical `PolarsCleaner.X(...)` API. Collision-safe `column_names_to_snake_case(on_collision=)` and `clean_df(drop_all_null_columns=)` are opt-in flags
- **`spark/`** — Distributed processing, uses native Spark SQL operations (no Python UDFs for performance). Has its own modular subsystem: `type_parsers.py`, `type_checkers.py`, `type_inference.py`, `diagnostics.py`, `config.py`

**Backwards compatibility:** `src/etl/dataframe/__init__.py` re-exports `Cleaner`, `Parser`, `Analyzer` from the `pandas/` subpackage. Legacy code using `from etl.dataframe import Cleaner` continues to work.

### Database Layer (`src/etl/database/`)

- **`sql_dialects.py`** — Frozen `SqlDialect` dataclass defining escaping, types, placeholders, and constraint generation per database. Three instances: `mssql`, `mariadb`, `postgres`
- **`connector.py`** — SQLAlchemy engine factory with static methods per database type
- **`unified_loader.py`** — Dialect-aware INSERT loader; `mssql_loader.py` and `mysql_loader.py` provide database-specific optimizations
- **`validator.py`** — Pre-upload validation (extra columns, type mismatches, truncation detection)
- **`creator.py` / `query/creator.py`** — SQL CREATE TABLE generation from DataFrame analysis

### I/O Layer (`src/etl/io/`)

- **`arrow_schema.py`** — `unify_parquet_schemas(paths)` reads Parquet footers only (concurrently) and reconciles incompatible physical types across files into one castable PyArrow schema (any decimal → `float64`; any cross-file type conflict → `large_string`). Pass the result as `ds.dataset(paths, schema=...)` to cast on read. Requires `pyarrow`.

### Type Inference Hierarchy

All backends follow the same narrowing order: **Boolean → Integer → Float → Date → String**, falling back to wider types when narrower ones fail. Spark adds sampling-based inference with automatic retry and type broadening.

### Key Design Decisions

- **Spark uses DoubleType** (64-bit) over FloatType for financial precision
- **All timestamps normalize to UTC** for consistent serialization
- **Singleton logger** (`src/etl/logger.py`) with colored Rich output, dual stdout/stderr streams
- **`constants.py`** maps database types and numpy types used throughout validation and type conversion

## Testing

Tests are in `tests/` with the naming pattern `*_test.py`. Spark tests use `conftest_spark.py` which provides a `SparkTestCase` base class with session management and helper assertions (`assert_type()`, `assert_values()`, `assert_dates()`).

Test files are organized by module: `parser_test.py`, `cleaner_test.py`, `analyzer_test.py` for Pandas; `polars_cleaner_test.py` for Polars; `spark_type_parsing_test.py`, `spark_diagnostics_test.py`, etc. for Spark.
