import unittest

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from src.etl.dataframe.spark.cleaner import SparkCleaner


class TestCleaner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local").appName("CleanerTest").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def create_dataframe(self, data, schema) -> DataFrame:
        return self.spark.createDataFrame(data, schema)

    def test_column_names_to_snake_case(self):
        data = [(1, "John Doe", "Value1"), (2, "Jane Doe", "Value2")]
        schema = StructType([
            StructField("UserID", IntegerType(), True),
            StructField("user name", StringType(), True),
            StructField("with$Special#Chars", StringType(), True)

        ])
        df = self.create_dataframe(data, schema)

        result_df = SparkCleaner.column_names_to_snake_case(df)

        self.assertEqual(set(result_df.columns), {"user_id", "user_name", "with_dollars_special_num_chars"})

    def test_clean_all_types(self):
        data = [
            ("y", "%123", "$45.67", "2021-01-01"),
            ("false", "$456", "%89.10", "2022-11-19"),
            ("no", "789.00", "50", "2022-05-05")
        ]
        schema = StructType([
            StructField("boolean_col", StringType(), True),
            StructField("integer_col", StringType(), True),
            StructField("float_col", StringType(), True),
            StructField("date_col", StringType(), True)
        ])
        df = self.create_dataframe(data, schema)

        result_df = SparkCleaner.clean_all_types(df)

        self.assertEqual(result_df.schema["boolean_col"].dataType.simpleString(), "boolean")
        self.assertEqual(result_df.schema["integer_col"].dataType.simpleString(), "int")
        self.assertEqual(result_df.schema["float_col"].dataType.simpleString(), "float")
        self.assertEqual(result_df.schema["date_col"].dataType.simpleString(), "timestamp")

    def test_clean_df(self):
        data = [
            (1, "John", None),
            (None, None, None),
            (2, "Jane", 3)
        ]
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("value", IntegerType(), True)
        ])
        df = self.create_dataframe(data, schema)

        result_df = SparkCleaner.clean_df(df)

        self.assertEqual(result_df.count(), 2)

    def test_clean_all_null_columns(self):
        data = [
            (1, None, "Text"),
            (2, None, "More Text"),
            (3, None, "Even More Text")
        ]
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("empty_col", StringType(), True),
            StructField("text_col", StringType(), True)
        ])
        df = self.create_dataframe(data, schema)

        result_df = SparkCleaner.clean_df(df)

        self.assertNotIn("empty_col", result_df.columns)
        self.assertIn("text_col", result_df.columns)


if __name__ == "__main__":
    unittest.main()
