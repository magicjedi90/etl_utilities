# Project Documentation

## Table of Contents

1. [Overview](#overview)
2. [Classes](#classes)
   - [Connector](#connector)
   - [Loader](#loader)
   - [MySqlLoader](#mysqlloader)
   - [MsSqlLoader](#mssqlloader)
   - [Parser](#parser)
   - [Cleaner](#cleaner)
   - [Creator](#creator)
   - [Analyzer](#analyzer)
   - [Validator](#validator)
   - [MsSqlUpdater](#mssqlupdater)
3. [Logging](#logging)
4. [Additional Utilities](#additional-utilities)

## Overview

This project provides a comprehensive Data ETL \(Extract, Transform, Load\) and data manipulation framework using Python. It integrates with databases using SQLAlchemy and provides tools for data parsing, cleaning, loading, validating, and more. The project is structured with classes that encapsulate different functionalities.

## Classes

### Connector

The `Connector` class handles creating connections to various types of databases \(MSSQL, PostgreSQL, MySQL\) using SQLAlchemy. It provides static methods for obtaining both trusted and user connections.

**Key Methods:**
- `get_mssql_trusted_connection`
- `get_mssql_user_connection`
- `get_postgres_user_connection`
- `get_mysql_user_connection`
- Instance methods for returning database connections based on stored configuration.

### Loader

The `Loader` class is responsible for loading data from a Pandas DataFrame into a database. It manages the insertion process, ensuring data is inserted efficiently and effectively with the use of SQLAlchemy and custom logging.

### MySqlLoader

A slight extension of the `Loader` class specifically for MySQL databases. It provides overrides to manage MySQL-specific data types and query formatting.

### MsSqlLoader

A specialized loader for loading data into MSSQL databases with additional functionalities like fast insertions using bulk methods.

### Parser

The `Parser` class consists of a series of static methods dedicated to parsing various data types—boolean, float, date, and integer. These methods are essential for data type conversion and consistency across the application.

### Cleaner

The `Cleaner` class provides methods for sanitizing and formatting data in a DataFrame. It includes functions for setting column name casing conventions, cleaning various types of data, and preparing data for reliable analysis and insertion.

### Creator

This class deals with generating SQL `CREATE TABLE` statements for different databases like MSSQL and MariaDB. The query generation considers data types deduced from DataFrame content.

### Analyzer

The `Analyzer` class assesses DataFrame characteristics and helps identify unique columns, column pairs, empty columns, and more. It aids in generating metadata for data types, which is crucial for creating or validating schemas.

### Validator

The `Validator` class ensures DataFrame compatibility with the target database table structure by checking for extra columns, validating data types, and ensuring that no data truncation will occur during upload.

### MsSqlUpdater

A class designed for constructing SQL statements for operations like mergers, updates, inserts, and appends to manage data transitions between tables efficiently.

## Logging

The project uses a singleton `Logger` class with colored output format for console logging. This helps in debugging and understanding the flow by logging messages at various severity levels.

## Additional Utilities

- **Parsing and Cleaning Functions:** Utility functions for parsing and cleaning various data types.
- **Standardization:** A set of utility functions to standardize and clean DataFrame column names and content.