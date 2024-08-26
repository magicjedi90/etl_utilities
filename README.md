# Data Cleaning and SQL Query Generation Utilities

This repository provides a collection of utility functions and classes for data cleaning, SQL query generation, and data analysis. The code is written in Python and uses libraries such as `pandas`, `numpy`, and `dateutil`.

## Table of Contents


- [Usage](#usage)
  - [Cleaning Functions](#cleaning-functions)
  - [Cleaner Class](#cleaner-class)
  - [Analyzer Class](#analyzer-class)
  - [Creator Class](#creator-class)
  - [Updater Class](#updater-class)
  - [Validator Class](#validator-class)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)


## Usage

### Cleaning Functions

#### `parse_boolean(value)`

Parses boolean values from various possible string representations.

- **Parameters:** 
  - `value` (*any*): The value to be parsed.
- **Returns:** 
  - `True` or `False` or `None`
- **Raises:** 
  - `ValueError`: If the value is not a recognized boolean representation.

```python
parse_boolean('yes')   # True
parse_boolean('no')    # False
```

#### `parse_float(value)`

Parses a float value, cleaning up common characters like commas, dollar signs, and percentages.

- **Parameters:** 
  - `value` (*any*): The value to be parsed.
- **Returns:** 
  - `float` or `None`
  
```python
parse_float('$1,234.56')    # 1234.56
parse_float('45%')          # 45.0
```

#### `parse_date(value)`

Parses values into a date.

- **Parameters:** 
  - `value` (*any*): The value to be parsed.
- **Returns:** 
  - `datetime` or `None`
  
```python
parse_date('2023-10-04')    # datetime object
```

#### `parse_integer(value)`

Parses integer values.

- **Parameters:** 
  - `value` (*any*): The value to be parsed.
- **Returns:** 
  - `int` or `None`
- **Raises:** 
  - `ValueError`: If the value cannot be converted to an integer.
  
```python
parse_integer('123')   # 123
```

#### `compute_hash(value)`

Computes a SHA-1 hash of the input value.

- **Parameters:** 
  - `value` (*any*): The value to be hashed.
- **Returns:** 
  - `str`: SHA-1 hash of the input value.
  
```python
compute_hash('test')    # 'a94a8fe5ccb19ba61c4c0873d391e987982fbbd3'
```

### Helper Functions

#### `standardize_column_name(name)`

Standardizes column names into snake_case.

- **Parameters:** 
  - `name` (*str*): The column name to be standardized.
- **Returns:** 
  - `str`: The standardized column name.
  
```python
standardize_column_name('Date of Birth')    # 'date_of_birth'
standardize_column_name('Employee ID#')     # 'employee_id_num'
```

## `Cleaner` Class

This class provides static methods to clean and manipulate pandas DataFrame columns.

### Methods

#### `Cleaner.column_names_to_snake_case(df)`

Converts all column names in the DataFrame to snake_case.

- **Parameters:** 
  - `df` (*pd.DataFrame*): The DataFrame whose column names need to be converted.
- **Returns:** 
  - None
  
```python
Cleaner.column_names_to_snake_case(df)
```

#### `Cleaner.clean_column(df, column, clean_function)`

Cleans a specified column using a provided function.

- **Parameters:** 
  - `df` (*pd.DataFrame*): The DataFrame with the column to be cleaned.
  - `column` (*str*): The column name.
  - `clean_function` (*function*): The function to clean the column.
- **Returns:** 
  - The cleaned column.
  
```python
Cleaner.clean_column(df, 'salary', parse_float)
```

#### `Cleaner.clean_numbers(df)`

Cleans columns in the DataFrame that contain numerical values.

- **Parameters:** 
  - `df` (*pd.DataFrame*): The DataFrame to be cleaned.
- **Returns:** 
  - `pd.DataFrame`: The cleaned DataFrame.
  
```python
Cleaner.clean_numbers(df)
```

#### `Cleaner.clean_dates(df)`

Cleans columns in the DataFrame that contain date values.

- **Parameters:** 
  - `df` (*pd.DataFrame*): The DataFrame to be cleaned.
- **Returns:** 
  - `pd.DataFrame`: The cleaned DataFrame.
  
```python
Cleaner.clean_dates(df)
```

#### `Cleaner.clean_bools(df)`

Cleans columns in the DataFrame that contain boolean values.

- **Parameters:** 
  - `df` (*pd.DataFrame*): The DataFrame to be cleaned.
- **Returns:** 
  - `pd.DataFrame`: The cleaned DataFrame.
  
```python
Cleaner.clean_bools(df)
```

#### `Cleaner.clean_all(df)`

Attempts to clean all columns in the DataFrame.

- **Parameters:** 
  - `df` (*pd.DataFrame*): The DataFrame to be cleaned.
- **Returns:** 
  - `pd.DataFrame`: The cleaned DataFrame.
  
```python
Cleaner.clean_all(df)
```

#### `Cleaner.generate_hash_column(df, columns_to_hash, new_column_name)`

Generates a new column that contains the SHA-1 hash of specified columns.

- **Parameters:** 
  - `df` (*pd.DataFrame*): The DataFrame to be processed.
  - `columns_to_hash` (*list*): The list of column names to hash.
  - `new_column_name` (*str*): The name of the new column.
- **Returns:** 
  - `pd.DataFrame`: The DataFrame with the new hash column.
  
```python
Cleaner.generate_hash_column(df, ['name', 'birth_date'], 'identity_hash')
```

#### `Cleaner.coalesce_columns(df, columns_to_coalesce, new_column_name, drop=False)`

Combines multiple columns into a single column, prioritizing non-null values.

- **Parameters:** 
  - `df` (*pd.DataFrame*): The DataFrame to be processed.
  - `columns_to_coalesce` (*list*): The list of columns to coalesce.
  - `new_column_name` (*str*): The name of the new column.
  - `drop` (*bool*): Whether to drop the original columns.
- **Returns:** 
  - `pd.DataFrame`: The DataFrame with the coalesced column.
  
```python
Cleaner.coalesce_columns(df, ['phone_home', 'phone_mobile', 'phone_work'], 'phone', drop=True)
```

#### Example Usage

Here's an example of how you can use these utilities:

```python
import pandas as pd

data = {
    'name': ['Alice', 'Bob'],
    'birth_date': ['1990-01-01', '1985-05-12'],
    'is_active': ['yes', 'no'],
    'salary': ['$1,200.50', '$2,500.75'],
}

df = pd.DataFrame(data)

# Clean the DataFrame
Cleaner.column_names_to_snake_case(df)
df = Cleaner.clean_dates(df)
df = Cleaner.clean_bools(df)
df = Cleaner.clean_numbers(df)

# Generate hash column
df = Cleaner.generate_hash_column(df, ['name', 'birth_date'], 'identity_hash')

print(df)
```

The above script will clean the DataFrame, convert column names to snake_case, cleanse date, boolean, and numerical values, and generate a hash column based on the specified columns.
### Analyzer Class

Provides utilities for analyzing DataFrames.
#### `Analyzer.find_single_id_candidate_columns(df)`

Finds columns that can serve as unique identifiers.

```python
import pandas as pd
from etl.dataframe.analyzer import Analyzer

df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Alice']})
candidates = Analyzer.find_single_id_candidate_columns(df)
print(candidates)  # Output: ['id']

```
#### `Analyzer.find_id_pair_candidates(df)`
Finds pairs of columns that can serve as unique identifiers.

```python
import pandas as pd
from etl.dataframe.analyzer import Analyzer

df = pd.DataFrame({'first': [1, 2, 2], 'second': [3, 3, 4]})
candidates = Analyzer.find_id_pair_candidates(df)
print(candidates)  # Output: [('first', 'second')]
```
### Creator Class

Generates SQL queries for creating tables.
#### `Creator.make_mssql_table(df, schema, table, primary_key=None, history=False, varchar_padding=20, float_precision=10, decimal_places=2)`

Generates a SQL CREATE TABLE statement based on a DataFrame.
### Updater Class

Generates SQL queries for inserting and merging data.
#### `Updater.merge_mssql(source_schema, source_table, target_schema, target_table, columns, id_column, delete_unmatched=True)`

Generates a SQL MERGE statement.
#### `Updater.upsert_mssql(source_schema, source_table, target_schema, target_table, columns, id_column)`

Generates a SQL UPSERT statement.
#### `Updater.append_mssql(source_schema, source_table, target_schema, target_table, columns)`

Generates a SQL INSERT statement with EXCEPT.

### `Validator` Class

This class provides static methods to validate the structure and data of a DataFrame before uploading it to a database.

#### Methods

##### `validate_upload(connection, df: pd.DataFrame, schema: str, table: str)`

Validates the DataFrame against the specified database schema and table.

- **Parameters:**
  - `connection` (*sqlalchemy.engine.Connection*): Database connection.
  - `df` (*pd.DataFrame*): The DataFrame to be validated.
  - `schema` (*str*): The schema name.
  - `table` (*str*): The table name.
- **Returns:** 
  - None

```python
Validator.validate_upload(connection, df, 'public', 'my_table')
```

##### `_fetch_column_info(connection, df, schema, table)`

Fetches column information from the database for the specified schema and table.

- **Parameters:** 
  - `connection` (*sqlalchemy.engine.Connection*): Database connection.
  - `df` (*pd.DataFrame*): The DataFrame to be validated.
  - `schema` (*str*): The schema name.
  - `table` (*str*): The table name.
- **Returns:** 
  - Tuple(List, pd.DataFrame): List of DataFrame columns and DataFrame containing column information.

##### `_check_extra_columns(df_columns, column_info_df, schema, table)`

Checks for extra columns in the DataFrame that are not in the database table schema.

- **Parameters:** 
  - `df_columns` (*List[str]*): List of DataFrame columns.
  - `column_info_df` (*pd.DataFrame*): DataFrame containing column information.
  - `schema` (*str*): The schema name.
  - `table` (*str*): The table name.
- **Raises:** 
  - `ExtraColumnsException`: If any extra columns are found.

##### `_validate_column_types(df, df_columns, column_info_df)`

Validates the data types of DataFrame columns against the database schema.

- **Parameters:** 
  - `df` (*pd.DataFrame*): The DataFrame to be validated.
  - `df_columns` (*List[str]*): List of DataFrame columns.
  - `column_info_df` (*pd.DataFrame*): DataFrame containing column information.
- **Raises:** 
  - `ColumnDataException`: If any column data type mismatches or truncation issues are found.

##### `_is_type_mismatch(df_column_data_type, db_column_data_type)`

Checks if there is a type mismatch between the DataFrame and database column data types.

- **Parameters:** 
  - `df_column_data_type` (*numpy.dtype*): Data type of the DataFrame column.
  - `db_column_data_type` (*str*): Data type of the database column.
- **Returns:** 
  - `bool`: `True` if there is a type mismatch, `False` otherwise.

##### `_check_numeric_truncation(column, df, db_column_info)`

Checks for numeric truncation issues.

- **Parameters:** 
  - `column` (*str*): The column name.
  - `df` (*pd.DataFrame*): The DataFrame containing the column.
  - `db_column_info` (*pd.Series*): Series containing information about the database column.
- **Returns:** 
  - `str` or `None`: Error message if truncation is detected, `None` otherwise.

##### `_check_string_or_date_truncation(column, df, db_column_info)`

Checks for string or date truncation issues.

- **Parameters:** 
  - `column` (*str*): The column name.
  - `df` (*pd.DataFrame*): The DataFrame containing the column.
  - `db_column_info` (*pd.Series*): Series containing information about the database column.
- **Returns:** 
  - `str` or `None`: Error message if truncation is detected, `None` otherwise.

#### Custom Exceptions

#### `ExtraColumnsException`

This exception is raised when extra columns are found in the DataFrame that are not present in the database table schema.

```python
class ExtraColumnsException(Exception):
    pass
```

#### `ColumnDataException`

This exception is raised when there are data type mismatches or truncation issues in the DataFrame columns.

```python
class ColumnDataException(Exception):
    pass
```

#### Example Usage

Here's an example of how you can use the `Validator` class to validate a DataFrame before uploading it to a database:

```python
import pandas as pd
from sqlalchemy import create_engine

data = {
    'name': ['Alice', 'Bob'],
    'age': [25, 30],
    'join_date': ['2020-01-01', '2019-06-15'],
}

df = pd.DataFrame(data)

# Create a connection to the database
engine = create_engine('sqlite:///example.db')
connection = engine.connect()

# Validate upload
try:
    Validator.validate_upload(connection, df, 'public', 'employees')
    print('DataFrame is valid for upload.')
except (ExtraColumnsException, ColumnDataException) as e:
    print(f'Validation failed: {e}')

connection.close()
```

The above script will validate the DataFrame against the specified database schema and table, ensuring that column types match and no extra columns are present.
## Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss your ideas.
## License

This project is licensed under the MIT License.