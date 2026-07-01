# etl_utilities

A Python ETL framework for **data cleaning, type inference, SQL generation, and database loading** that works across three DataFrame backends (**Pandas, Polars, Spark**) and three database families (**MSSQL, MySQL/MariaDB, PostgreSQL**).

It exists to remove the boilerplate from the boring-but-critical middle of a data pipeline: taking a messy extract, coercing every column to a sane type, generating a matching `CREATE TABLE`, and loading it efficiently — without hand-writing per-column casts or dialect-specific SQL.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start — an end-to-end pipeline](#quick-start--an-end-to-end-pipeline)
3. [Batch-safe cleaning for streaming (Polars)](#batch-safe-cleaning-for-streaming-polars)
4. [Reconciling mismatched Parquet schemas](#reconciling-mismatched-parquet-schemas)
5. [Backends & databases](#backends--databases)
6. [Class reference](#class-reference)
7. [Package documentation](#package-documentation)
8. [Logging](#logging)
9. [License](#license)

## Installation

```bash
pip install etl_utilities
```

The core install pulls in `pandas`, `numpy`, `SQLAlchemy`, `python-dateutil`, and `rich`.
Backend-specific dependencies are optional extras:

```bash
pip install etl_utilities[polars]     # Polars cleaning
pip install etl_utilities[spark]      # Spark cleaning
pip install etl_utilities[postgres]   # Postgres loading (psycopg2)
pip install etl_utilities[arrow]      # Parquet schema reconciliation (pyarrow)
pip install etl_utilities[all]        # everything
```

Requires Python 3.10+.

## Quick Start — an end-to-end pipeline

Take a messy CSV export, clean it, generate a Postgres table from the inferred schema, and load it:

```python
import pandas as pd

from etl.dataframe.cleaner import Cleaner
from etl.query.creator import Creator
from etl.database.sql_dialects import postgres
from etl.database.connector import Connector
from etl.database.unified_loader import Loader

# 1. EXTRACT — read a messy upstream export
raw = pd.read_csv("sales_export.csv")

# 2. TRANSFORM — standardize column names (in place) and coerce types
Cleaner.column_names_to_snake_case(raw)        # "Order ID" -> "order_id"
clean = Cleaner.clean_df(raw)                   # drop empty rows/cols + infer bool/int/float/date

# 3. GENERATE DDL — a CREATE TABLE matching the cleaned frame's inferred schema
ddl = Creator.create_table(
    clean,
    schema_name="analytics",
    table_name="sales",
    dialect=postgres,
    primary_key_column="order_id",
)

# 4. LOAD — open a connection and bulk-insert
connection = Connector(
    host="localhost", port=5432, instance="",
    database="warehouse", username="etl", password="***",
).to_user_postgres()

connection.execute(ddl)
Loader(connection, clean, schema="analytics", table="sales", dialect=postgres).insert()
```

Swap `postgres` for `mssql` or `mariadb` (from `etl.database.sql_dialects`) and the generated DDL and the loader's placeholders/escaping change with it — the rest of the pipeline is identical.

## Batch-safe cleaning for streaming (Polars)

Per-frame type inference is correct for one in-memory table but **breaks batch-by-batch**: batch A might infer `Boolean`/`Int8` while batch B infers `String`/`Int32`, and any downstream concat / Parquet-dataset / DB append then rejects the schema mismatch.

`PolarsCleaner` solves this by separating **inference** (decide the plan once) from **application** (apply identically to every batch):

```python
from etl.dataframe.polars.cleaner import PolarsCleaner

# Infer a deterministic plan once, from a representative sample.
# prefer_float keeps numeric columns stable even if a later batch turns out fractional.
plan = PolarsCleaner.infer_cleaning_plan(sample_df, prefer_float=True)

for batch in stream:                                  # every batch gets the SAME schema
    batch = PolarsCleaner.apply_cleaning_plan(batch, plan)
    sink.write(batch)
```

For one-shot in-memory cleaning, the original `PolarsCleaner.clean_all_types(df)` / `clean_df(df)` still do everything in a single pass.

A few more schema-stability helpers, useful right before a strict sink (Delta/Parquet/warehouse append):

```python
df = PolarsCleaner.sanitize_float_columns(df)      # NaN / +-Inf (and "NaN"/"Infinity" strings) -> null
df = PolarsCleaner.localize_naive_datetimes(df)    # naive Datetime -> tz-aware (UTC by default)
df = PolarsCleaner.cast_null_columns(df)           # all-null (pl.Null) columns -> a concrete dtype
df = PolarsCleaner.clean_df(df, drop_all_null_columns=False)  # keep all-null columns for schema parity

# Collision-safe snake_case: when two source columns normalize to the same name,
# coalesce them (default), or pass on_collision="error" / "suffix".
df = PolarsCleaner.column_names_to_snake_case(df)
```

## Reconciling mismatched Parquet schemas

A pile of Parquet files with the same logical columns but incompatible physical types
(`Decimal(38,12)` vs `Decimal(38,13)`, string-vs-decimal) defeats `pl.scan_parquet([...])`
and `pa.unify_schemas(...)` alike. `unify_parquet_schemas` reads only the footers (concurrently)
and builds one schema every file can be **cast** to on read:

```python
import pyarrow.dataset as ds
from etl.io.arrow_schema import unify_parquet_schemas

unified = unify_parquet_schemas(parquet_paths)         # any decimal -> float64; cross-file conflict -> large_string
table = ds.dataset(parquet_paths, schema=unified).to_table()   # PyArrow casts on read
```

Combine the three: `unify schemas → stream batches → apply_cleaning_plan` rebuilds a full
schema-stable, memory-safe cleaning pipeline from library primitives alone.

## Backends & databases

| DataFrame backend | Module | Notes |
|---|---|---|
| Pandas | `etl.dataframe.cleaner` / `parser` / `analyzer` | Eager, the most complete implementation |
| Polars | `etl.dataframe.polars` | Expression-based; adds the batch-safe streaming API above |
| Spark | `etl.dataframe.spark` | Distributed; native Spark SQL (no Python UDFs), sampling-based inference |

All backends follow the same type-narrowing order: **Boolean → Integer → Float → Date → String**.

Database dialects (`etl.database.sql_dialects`): `mssql`, `mariadb`, `postgres` — each defines its own
escaping, type mapping, placeholders, and constraint syntax, consumed by the `Creator` and `Loader`.

## Class reference

- **Connector** — SQLAlchemy connection factory for MSSQL (trusted/user), PostgreSQL, and MySQL.
- **Loader** (`unified_loader`) — dialect-aware `INSERT` loader from a DataFrame; `MsSqlLoader` / `MySqlLoader` add backend-specific optimizations (e.g. `fast_executemany`).
- **Parser** — static type parsers (boolean, float, date, integer) shared by the cleaners.
- **Cleaner** / **PolarsCleaner** — column-name normalization plus value cleaning and type inference.
- **Creator** — generates `CREATE TABLE` DDL from a DataFrame's analyzed schema for any dialect.
- **Analyzer** — finds unique columns/pairs, empty and categorical columns, and column metadata.
- **Validator** — pre-upload checks: extra columns, type mismatches, truncation detection.
- **MsSqlUpdater** — builds `MERGE` / `UPSERT` / `APPEND` SQL between source and target tables.

## Package documentation

Each subpackage has its own focused README:

- [`src/etl/dataframe/readme.md`](src/etl/dataframe/readme.md) — Analyzer, Cleaner, Parser across the Pandas/Polars/Spark backends.
- [`src/etl/database/readme.md`](src/etl/database/readme.md) — Connector, Loaders, Validator.
- [`src/etl/query/readme.md`](src/etl/query/readme.md) — Creator and MsSqlUpdater SQL generation.

## Logging

Library modules log through standard `logging.getLogger(__name__)` loggers and emit no output
unless your application configures logging. The optional `etl.logger.Logger` singleton is a
convenience for applications: it attaches ANSI-colored dual stdout/stderr handlers.

## License

MIT — see [`LICENSE`](LICENSE).
