class Updater:
    @staticmethod
    def merge_mssql(source_schema: str, source_table: str, target_schema: str, target_table: str, columns: list[str],
                    id_column: str, delete_unmatched: bool = True):
        stage = f'{source_schema}.{source_table}'
        id_column = f'[{id_column}]'
        location = f'{target_schema}.{target_table}'
        clean_columns = [f'[{column}]' for column in columns]
        source_columns = ', '.join([f'b.{column}' for column in clean_columns])
        column_string = ', '.join(clean_columns)
        update_check = ' or '.join(
            [f'a.{column} <> b.{column} or (a.{column} is null and b.{column} is not null) ' for column in
             clean_columns if column != id_column]
        )
        update_check = f' and ({update_check})'
        update_columns = ', '.join([f'a.{column} = b.{column}' for column in clean_columns if column != id_column])
        query = (
            f'merge {location} a using {stage} b on a.{id_column} = b.{id_column} '
            f'when matched {update_check} then update set {update_columns} '
            f'when not matched by target then insert ({column_string}) values ({source_columns})'
        )
        if delete_unmatched:
            query = f'{query} when not matched by source then delete'
        return f'{query};'

    @staticmethod
    def upsert_mssql(source_schema: str, source_table: str, target_schema: str, target_table: str, columns: list[str],
                     id_column: str):
        stage = f'{source_schema}.{source_table}'
        location = f'{target_schema}.{target_table}'
        clean_columns = [f'[{column}]' for column in columns]
        column_string = ', '.join(clean_columns)
        stage_columns = [f's.{column}' for column in clean_columns]
        stage_column_string = ', '.join(stage_columns)
        delete_dupes_query = (
            f'Delete from {stage} from {stage} s where exists (select '
            f'{stage_column_string} intersect select {column_string} from {location})'
        )
        delete_old_query = (
            f'delete from {location} where {id_column} in ( '
            f'select {id_column} from {stage} intersect select {id_column} from {location})'
        )
        insert_query = (
            f'insert into {location} ({column_string}) select {column_string} from {stage}'
        )
        query = f'{delete_dupes_query}; {delete_old_query}; {insert_query};'
        return query

    @staticmethod
    def append_mssql(source_schema: str, source_table: str, target_schema: str, target_table: str, columns: list[str]):
        stage = f'{source_schema}.{source_table}'
        location = f'{target_schema}.{target_table}'
        clean_columns = [f'[{column}]' for column in columns]
        column_string = ','.join(clean_columns)
        query = (
            f'insert into {location} ({column_string}) select {column_string} from {stage}'
            f' except select {column_string} from {location}'
        )
        return query
