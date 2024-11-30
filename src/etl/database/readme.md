# Database Utilities

## Table of Contents

1. [Overview](#overview)
2. [Components](#components)
   - [Connector Class](#1-connector-class)
   - [Loader Classes](#2-loader-classes)
   - [Validator Class](#3-validator-class)
3. [General Considerations](#general-considerations)

## Overview

These utilities provide tools for connecting to various SQL databases, efficiently loading data into tables, and validating data before uploading.

## Components

### 1. Connector Class

The `Connector` class is used to establish connections to SQL databases including SQL Server, PostgreSQL, and MySQL.

#### Usage

```python
from etl.database.connector import Connector

# Create an instance of the Connector class
connector = Connector(host='localhost', port=1433, instance='SQLEXPRESS', database='your_db', username='user', password='pass')

# Trusted SQL Server connection
trusted_connection = connector.to_trusted_msql()

# User-based SQL Server connection
user_connection = connector.to_user_msql()

# PostgreSQL connection
postgres_connection = connector.to_user_postgres()

# MySQL connection
mysql_connection = connector.to_user_mysql()
```

### 2. Loader Classes

The `Loader` class and its derivatives (`MySqlLoader` and `MsSqlLoader`) handle data insertion into database tables using Pandas DataFrames.

#### Usage

```python
from etl.database.mysql_loader import MySqlLoader
from etl.database.mssql_loader import MsSqlLoader

import pandas as pd
from sqlalchemy import create_engine

# Create a database connection
engine = create_engine('your_connection_string')
connection = engine.connect()

# Assume df is your DataFrame and already exists
df = pd.DataFrame(...)

# MySQL Loader Example
mysql_loader = MySqlLoader(cursor=connection, df=df, schema='your_schema', table='your_table')
mysql_loader.to_table()

# MSSQL Loader Example
mssql_loader = MsSqlLoader(cursor=connection, df=df, schema='your_schema', table='your_table')
mssql_loader.to_table()
```

#### Fast Insertion for MSSQL

```python
mssql_loader.to_table_fast(batch_size=500)
```

### 3. Validator Class

The `Validator` class ensures that the DataFrame you are trying to upload is structured correctly to match the database schema.

#### Usage

```python
from etl.database.validator import Validator

# Assume df is your DataFrame and already exists
df = pd.DataFrame(...)

# Validate the DataFrame against the database schema
validator = Validator(connection=connection, df=df, schema='your_schema', table='your_table')
validator.validate()
```

#### Handling Exceptions

- **ExtraColumnsException**: Raised when extra columns in the DataFrame do not exist in the target table.
- **ColumnDataException**: Raised when there are data type mismatches or truncation issues.

## General Considerations

- Ensure your SQLAlchemy engine and connection strings are correctly initialized.
- Pre-process DataFrames for NaN values before database loading.
- For large datasets, use `fast_executemany` or batch insertion for efficiency.
