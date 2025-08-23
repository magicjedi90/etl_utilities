# src/etl/spark/udfs.py

from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType, FloatType, StringType, TimestampType, IntegerType
from src.etl.dataframe.parser import Parser

def is_boolean(value):
    try:
        Parser.parse_boolean(value)
        return True
    except (ValueError, TypeError):
        return False

def is_integer(value):
    try:
        Parser.parse_integer(value)
        return True
    except (ValueError, TypeError):
        return False

def is_float(value):
    try:
        Parser.parse_float(value)
        return True
    except (ValueError, TypeError):
        return False

def is_date(value):
    try:
        Parser.parse_date(value)
        return True
    except (ValueError, TypeError, OverflowError):
        return False

# Register the test functions as UDFs
is_boolean_udf = udf(is_boolean, BooleanType())
is_integer_udf = udf(is_integer, BooleanType())
is_float_udf = udf(is_float, BooleanType())
is_date_udf = udf(is_date, BooleanType())

# Register UDFs for each parsing function
parse_boolean_udf = udf(Parser.parse_boolean, BooleanType())
parse_float_udf = udf(Parser.parse_float, FloatType())
parse_date_udf = udf(Parser.parse_date, TimestampType())
parse_integer_udf = udf(Parser.parse_integer, IntegerType())