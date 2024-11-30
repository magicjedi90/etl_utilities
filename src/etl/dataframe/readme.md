# Data Processing Library

These modules provide tools for data analysis and cleaning, specifically focused on handling pandas DataFrames. It consists of three main components: the `Analyzer`, `Cleaner`, and `Parser` modules.

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