# Data Cleaning and SQL Query Generation Utilities

This repository provides a collection of utility functions and classes for data cleaning, SQL query generation, and data analysis. The code is written in Python and uses libraries such as `pandas`, `numpy`, and `dateutil`.

## Table of Contents


- [Usage](#usage)
  - [Cleaning Functions](#cleaning-functions)
  - [Cleaner Class](#cleaner-class)
  - [Analyzer Class](#analyzer-class)
  - [TableMaker Class](#tablemaker-class)
  - [Inserter Class](#inserter-class)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)


## Usage

### Cleaning Functions

#### `clean_float(dirty_float)`

Cleans a float by removing unwanted characters and converting it to a float type.

```python
from etl.dataframe.cleaner import clean_float

cleaned_float = clean_float('$1,234.56')
print(cleaned_float)  # Output: 1234.56

```

#### `clean_date(dirty_date)`

Parses a date string and converts it to a datetime object.
```python
from etl.dataframe.cleaner import clean_date

cleaned_date = clean_date('2021-01-01')
print(cleaned_date)  # Output: 2021-01-01 00:00:00
```

#### `clean_int(dirty_int)`

Cleans an integer by removing unwanted characters and converting it to an integer type.
```python
from etl.dataframe.cleaner import clean_int

cleaned_int = clean_int('123.00')
print(cleaned_int)  # Output: 123
```

#### `hash_str(not_hashed)`

Hashes a string using SHA-1.

```python
from etl.dataframe.cleaner import hash_str

hashed_value = hash_str('test')
print(hashed_value)  # Output: a94a8fe5ccb19ba61c4c0873d391e987982fbbd3```
```

### Cleaner Class

Provides various data cleaning utilities.
#### `Cleaner.column_names_to_snake_case(df)`

Converts column names in a DataFrame to snake_case.

```python
import pandas as pd
from etl.dataframe.cleaner import Cleaner

df = pd.DataFrame({'MixedCase': [1, 2], 'with spaces': [3, 4]})
Cleaner.column_names_to_snake_case(df)
print(df.columns)  # Output: ['mixed_Case', 'with_spaces']
```


#### `Cleaner.clean_numbers(df)`

Cleans numerical columns in a DataFrame.
#### `Cleaner.clean_dates(df)`

Cleans date columns in a DataFrame.
#### `Cleaner.clean_bools(df)`

Cleans boolean columns in a DataFrame.
#### `Cleaner.clean_all(df)`

Cleans all columns in a DataFrame.
#### `Cleaner.generate_hash_column(df, columns_to_hash, new_column_name)`

Generates a hash column based on specified columns.
#### `Cleaner.coalesce_columns(df, columns_to_coalesce, new_column_name, drop=False)`

Coalesces multiple columns into a new column.
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
### Maker Class

Generates SQL queries for creating tables.
#### `Maker.make_mssql_table(df, schema, table, primary_key=None, history=False, varchar_padding=20, float_precision=10, decimal_places=2)`

Generates a SQL CREATE TABLE statement based on a DataFrame.
### Inserter Class

Generates SQL queries for inserting and merging data.
#### `Inserter.merge_mssql(source_schema, source_table, target_schema, target_table, columns, id_column, delete_unmatched=True)`

Generates a SQL MERGE statement.
#### `Inserter.upsert_mssql(source_schema, source_table, target_schema, target_table, columns, id_column)`

Generates a SQL UPSERT statement.
#### `Inserter.append_mssql(source_schema, source_table, target_schema, target_table, columns)`

Generates a SQL INSERT statement with EXCEPT.
## Testing

To run the unit tests, use the following command:

```sh
python -m unittest discover tests
```
## Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss your ideas.
## License

This project is licensed under the MIT License.