class MsSqlUpdater:

    def __init__(self, source_schema: str, source_table: str, source_columns: list[str], source_id_column: str,
                 target_schema: str, target_table: str, target_columns: list[str], target_id_column: str):
        self._source_schema = source_schema
        self._source_table = source_table
        self._source_columns = source_columns
        self._source_id_column = source_id_column
        self._target_schema = target_schema
        self._target_table = target_table
        self._target_columns = target_columns
        self._target_id_column = target_id_column

    @staticmethod
    def merge_sql(source_schema: str, source_table: str, source_columns: list[str], source_id_column: str,
                  target_schema: str, target_table: str, target_columns: list[str], target_id_column: str,
                  delete_unmatched: bool = True) -> str:
        if len(source_columns) != len(target_columns):
            raise ValueError("source_columns and target_columns must have the same length")
        stage = f'{source_schema}.{source_table}'
        target_id_column = f'[{target_id_column}]'
        source_id_column = f'[{source_id_column}]'

        location = f'{target_schema}.{target_table}'
        clean_target_columns = [f'[{column}]' for column in target_columns]
        clean_source_columns = [f'[{column}]' for column in source_columns]

        target_columns_str = ', '.join([f'{column}' for column in clean_target_columns])
        source_columns_str = ', '.join([f'b.{column}' for column in clean_source_columns])
        comparison_list = [(src_col, tgt_col) for src_col, tgt_col in zip(clean_source_columns, clean_target_columns)]
        comparison_str = ' or '.join(
            [f'a.{column[0]} <> b.{column[1]} or (a.{column[0]} is null and b.{column[1]} is not null) ' for column in
             comparison_list if column[0] != target_id_column]
        )
        update_str = ',\n\t\t'.join(
            [f'a.{column[0]} = b.{column[1]}' for column in comparison_list if column[0] != target_id_column])
        query = (
            f'merge {location} a\n'
            f'using {stage} b\n'
            f'on a.{target_id_column} = b.{source_id_column}\n'
            f'when matched and ({comparison_str}) then\n'
            f'\tupdate\n'
            f'\tset {update_str}\n'
            f'when not matched by target then\n'
            f'\tinsert ({target_columns_str})\n'
            f'\tvalues ({source_columns_str})'
        )
        if delete_unmatched:
            query = f'{query}\nwhen not matched by source then delete'
        return f'{query};'

    @staticmethod
    def upsert_sql(source_schema: str, source_table: str, source_columns: list[str], source_id_column: str,
                   target_schema: str, target_table: str, target_columns: list[str], target_id_column: str) -> str:
        stage = f'{source_schema}.{source_table}'
        location = f'{target_schema}.{target_table}'
        clean_target_columns = [f'[{column}]' for column in target_columns]
        clean_source_columns = [f'[{column}]' for column in source_columns]
        target_column_string = ', '.join(clean_target_columns)
        source_column_string = ', '.join(clean_source_columns)

        stage_columns = [f's.{column}' for column in clean_source_columns]
        stage_column_string = ', '.join(stage_columns)
        delete_dupes_query = (
            f'Delete from {stage} from {stage} s where exists (select '
            f'{stage_column_string} intersect select {target_column_string} from {location})'
        )
        delete_old_query = (
            f'delete from {location} where {target_id_column} in ( '
            f'select {source_id_column} from {stage} intersect select {target_id_column} from {location})'
        )
        insert_query = (
            f'insert into {location} ({target_column_string}) select {source_column_string} from {stage}'
        )
        query = f'{delete_dupes_query}; {delete_old_query}; {insert_query};'
        return query

    @staticmethod
    def append_sql(source_schema: str, source_table: str, source_columns: list[str], target_schema: str,
                   target_table: str, target_columns: list[str]) -> str:
        stage = f'{source_schema}.{source_table}'
        location = f'{target_schema}.{target_table}'
        clean_target_columns = [f'[{column}]' for column in target_columns]
        clean_source_columns = [f'[{column}]' for column in source_columns]

        target_column_string = ','.join(clean_target_columns)
        source_column_string = ','.join(clean_source_columns)

        query = (
            f'insert into {location} ({target_column_string}) select {source_column_string} from {stage}'
            f' except select {target_column_string} from {location}'
        )
        return query

    def merge(self, delete_unmatched: bool = True) -> str:
        return self.merge_sql(
            self._source_schema, self._source_table, self._source_columns, self._source_id_column,
            self._target_schema, self._target_table, self._target_columns, self._target_id_column, delete_unmatched
        )

    def upsert(self) -> str:
        return self.upsert_sql(
            self._source_schema, self._source_table, self._source_columns, self._source_id_column,
            self._target_schema, self._target_table, self._target_columns, self._target_id_column
        )

    def append(self) -> str:
        return self.append_sql(
            self._source_schema, self._source_table, self._source_columns,
            self._target_schema, self._target_table, self._target_columns
        )