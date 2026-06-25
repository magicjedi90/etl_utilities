# Query Generation

SQL generation utilities: build `CREATE TABLE` DDL from a DataFrame, and build
`MERGE` / `UPSERT` / `APPEND` statements between a source and target table.

## Table of Contents

- [Creator Class](#creator-class)
  - [create_table](#create_table)
  - [Example](#example)
- [MsSqlUpdater Class](#mssqlupdater-class)
  - [Methods](#methods)
  - [Example](#example-1)

## Creator Class

The `Creator` class generates `CREATE TABLE` DDL from a Pandas DataFrame. Column types,
sizes, and nullability are derived from the data via the `Analyzer`, and the SQL is rendered
for whichever dialect you pass (`mssql`, `mariadb`, or `postgres` from
`etl.database.sql_dialects`).

### `create_table`

```python
Creator.create_table(
    data_frame,            # pd.DataFrame to model the schema from
    schema_name,           # target schema, e.g. "dbo" / "analytics"
    table_name,            # target table name
    dialect,               # one of etl.database.sql_dialects.{mssql, mariadb, postgres}
    primary_key_column=None,
    unique_columns=None,
    history=False,
    varchar_padding=20,
    float_precision=10,
    decimal_places=2,
    generate_identity_column=False,
) -> str                   # returns the CREATE TABLE statement
```

### Example

```python
import pandas as pd
from etl.query.creator import Creator
from etl.database.sql_dialects import mssql

df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30],
})

ddl = Creator.create_table(
    df,
    schema_name="dbo",
    table_name="users",
    dialect=mssql,
    primary_key_column="id",
    generate_identity_column=True,
)
print(ddl)
```

## MsSqlUpdater Class

The `MsSqlUpdater` class builds SQL statements for moving data between a source and a target
table within MSSQL. The `*_sql` static methods are pure string builders; the instance methods
(`merge`, `upsert`, `append`) build the same SQL from attributes supplied at construction and
return the statement for you to execute.

### Methods

- **`merge_sql(...)`** / **`merge(delete_unmatched=True)`** — `MERGE` statement, optionally deleting unmatched target rows.
- **`upsert_sql(...)`** / **`upsert()`** — insert-or-update statement.
- **`append_sql(...)`** / **`append()`** — `INSERT INTO ... EXCEPT ...` that avoids inserting duplicates.

### Example

```python
from etl.query.mssql_updater import MsSqlUpdater

updater = MsSqlUpdater(
    source_schema="stage", source_table="new_users",
    source_columns=["name", "age"], source_id_column="user_id",
    target_schema="dbo", target_table="users",
    target_columns=["name", "age"], target_id_column="user_id",
)

merge_query = updater.merge()
print(merge_query)
```

## License

MIT — see the [repository LICENSE](../../../LICENSE).
