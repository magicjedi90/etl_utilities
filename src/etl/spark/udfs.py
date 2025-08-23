# src/etl/spark/udfs.py

from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType, FloatType, StringType, TimestampType, IntegerType
from etl.dataframe.parser import Parser # Import your existing parser

# Register UDFs for each parsing function
parse_boolean_udf = udf(Parser.parse_boolean, BooleanType())
parse_float_udf = udf(Parser.parse_float, FloatType())
parse_date_udf = udf(Parser.parse_date, TimestampType())
parse_integer_udf = udf(Parser.parse_integer, IntegerType())