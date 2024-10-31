import unittest
import pandas as pd
from datetime import datetime
from src.etl.query.creator import Creator


class TestTableMaker(unittest.TestCase):

    def test_make_mssql_table(self):
        df = pd.DataFrame({
            'id_column': [2147483649, 2147483650, 2147483651],
            'int_column': [1, 2, 3],
            'float_column': [1.1, 2.2, 3.3],
            'date_column': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'bool_column': [True, False, True],
            'str_column': ['a', 'bb', 'ccc'],
            'empty_column': [None, None, None]
        })

        expected_query = (
            "create table dbo.[test_table]\n(\n"
            "[id_column] bigint constraint pk_test_table_id_column primary key,\n"
            "\t[int_column] tinyint,\n"
            "\t[float_column] decimal(10, 2),\n"
            "\t[date_column] datetime2,\n"
            "\t[bool_column] bit,\n"
            "\t[str_column] nvarchar(23),\n"
            "\t[empty_column] nvarchar(max),\n"
            "\tsystem_record_start datetime2 generated always as row start\n"
            "\t\tconstraint df_test_table_system_record_start\n"
            "\t\tdefault sysutcdatetime() not null,\n"
            "\tsystem_record_end datetime2 generated always as row end\n"
            "\t\tconstraint df_test_table_system_record_end\n"
            "\t\tdefault sysutcdataetime() not null,\n"
            "\t\tperiod for system_time(system_record_start, system_record_end)\n)"
            " with \n(\tsystem_versioning = on (history_table = dbo.[test_table_history])\n);"
        )

        actual_query = Creator.create_mssql_table(df, 'dbo', 'test_table',
                                                  primary_key='id_column', history=True)

        self.assertEqual(expected_query, actual_query)


if __name__ == '__main__':
    unittest.main()
