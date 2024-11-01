class Updater:
    @staticmethod
    def merge_mssql(source_schema: str, source_table: str, source_columns: list[str], source_id_column: str,
                    target_schema: str, target_table: str, target_columns: list[str], target_id_column: str,
                    delete_unmatched: bool = True):
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
        update_str = ', '.join(
            [f'a.{column[0]} = b.{column[1]}' for column in comparison_list if column[0] != target_id_column])
        query = (
            f'merge {location} a using {stage} b on a.{target_id_column} = b.{source_id_column} '
            f'when matched and ({comparison_str}) then update set {update_str} '
            f'when not matched by target then insert ({target_columns_str}) values ({source_columns_str})'
        )
        if delete_unmatched:
            query = f'{query} when not matched by source then delete'
        return f'{query};'

    @staticmethod
    def upsert_mssql(source_schema: str, source_table: str, source_columns: list[str], source_id_column: str,
                     target_schema: str, target_table: str, target_columns: list[str], target_id_column: str):
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
    def append_mssql(source_schema: str, source_table: str, source_columns: list[str], target_schema: str,
                     target_table: str, target_columns: list[str]):
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
