import unittest
from src.etl.query.inserter import Inserter


class TestInserter(unittest.TestCase):

    source_schema = 'source_schema'
    source_table = 'source_table'
    target_schema = 'target_schema'
    target_table = 'target_table'
    columns = ['id', 'name', 'value']
    id_column = 'id'

    def test_merge_mssql(self):
        expected_query = (
            'merge target_schema.target_table a using source_schema.source_table b on a.[id] = b.[id] '
            'when matched  and (a.[name] <> b.[name] or (a.[name] is null and b.[name] is not null) '
            ' or a.[value] <> b.[value] or (a.[value] is null and b.[value] is not null) ) '
            'then update set a.[name] = b.[name], a.[value] = b.[value] '
            'when not matched by target then insert ([id], [name], [value]) values (b.[id], b.[name], b.[value]) '
            'when not matched by source then delete'
        )

        actual_query = Inserter.merge_mssql(
            self.source_schema, self.source_table, self.target_schema, self.target_table, self.columns, self.id_column
        )

        self.assertEqual(expected_query, actual_query)

    def test_upsert_mssql(self):

        expected_query = (
            'Delete from source_schema.source_table from source_schema.source_table s where exists (select s.[id], s.[name], s.[value] intersect select [id], [name], [value] from target_schema.target_table); '
            'delete from target_schema.target_table where id in ( select id from source_schema.source_table intersect select id from target_schema.target_table); '
            'insert into target_schema.target_table ([id], [name], [value]) select [id], [name], [value] from source_schema.source_table;'
        )

        actual_query = Inserter.upsert_mssql(
            self.source_schema, self.source_table, self.target_schema, self.target_table, self.columns, self.id_column
        )

        self.assertEqual(expected_query, actual_query)

    def test_append_mssql(self):

        expected_query = (
            'insert into target_schema.target_table ([id],[name],[value]) select [id],[name],[value] from source_schema.source_table'
            'except select [id],[name],[value] from target_schema.target_table'
        )

        actual_query = Inserter.append_mssql(
            self.source_schema, self.source_table, self.target_schema, self.target_table, self.columns
        )

        self.assertEqual(expected_query, actual_query)


if __name__ == '__main__':
    unittest.main()
