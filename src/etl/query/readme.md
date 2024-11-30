## Table of Contents


  - [Creator Class](#creator-class)
    - [Initialization](#initialization)
    - [Methods](#methods)
    - [Example](#example)
  - [MsSqlUpdater Class](#mssqlupdater-class)
    - [Initialization](#initialization-1)
    - [Methods](#methods-1)
    - [Example](#example-1)

## Creator Class

The `Creator` class is designed to facilitate the generation of SQL table creation queries for both MSSQL and MariaDB databases from a given Pandas DataFrame.

## Initialization

```python
def __init__(self, df: pd.DataFrame, schema: str, table: str, primary_key: str = None,
             unique_columns: list[str] = None, history: bool = False,
             varchar_padding: int = 20, float_precision: int = 10, decimal_places: int = 2,
             generate_id: bool = False) -> None
```

## Methods

- **create_mssql_table()**: Generates a SQL query to create a MSSQL table.
- **create_mariadb_table()**: Generates a SQL query to create a MariaDB table.
- **new_mssql_table()**: Returns a MSSQL table creation query using instance parameters.
- **new_mariadb_table()**: Returns a MariaDB table creation query using instance parameters.

## Example

```python
import pandas as pd
from your_module.creator import Creator  # Replace with actual module path

# Sample DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30]
})

creator = Creator(df, schema="dbo", table="users", primary_key="id", generate_id=True)
mssql_query = creator.new_mssql_table()
print(mssql_query)
```

## MsSqlUpdater Class

The `MsSqlUpdater` class provides utilities to generate SQL queries for updating, merging, and appending data between source and target tables within a MSSQL database.

## Initialization

```python
def __init__(self, source_schema: str, source_table: str, source_columns: list[str], source_id_column: str,
             target_schema: str, target_table: str, target_columns: list[str], target_id_column: str)
```

## Methods

- **merge_sql()**: Creates a SQL `MERGE` statement for combining datasets with optional deletion of unmatched records.
- **upsert_sql()**: Generates a SQL statement for inserting and updating records.
- **append_sql()**: Forms a SQL `INSERT INTO` statement with an `EXCEPT` clause to avoid duplicates.
- **merge()**: Instance method for executing a `MERGE` using initialized attributes.
- **upsert()**: Instance method for executing an `UPSERT` using initialized attributes.
- **append()**: Instance method for executing an `APPEND` using initialized attributes.

## Example

```python
from your_module.mssql_updater import MsSqlUpdater  # Replace with actual module path

updater = MsSqlUpdater(
    source_schema="stage", source_table="new_users", source_columns=["name", "age"], source_id_column="user_id",
    target_schema="dbo", target_table="users", target_columns=["name", "age"], target_id_column="user_id"
)

merge_query = updater.merge()
print(merge_query)
```

## Contributing

Instructions on how others can contribute to your project.

## License

Specify the license under which your project is distributed.
