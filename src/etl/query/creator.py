from __future__ import annotations
from typing import Sequence
import pandas as pd
from ..database.sql_dialects import SqlDialect
from ..dataframe.analyzer import Analyzer
from ..logger import Logger

logger = Logger().get_logger()


class Creator:
    def __init__(
            self,
            data_frame: pd.DataFrame,
            schema_name: str,
            table_name: str,
            primary_key_column: str | None = None,
            unique_columns: list[str] | None = None,
            history: bool = False,
            varchar_padding: int = 20,
            float_precision: int = 10,
            decimal_places: int = 2,
            generate_identity_column: bool = False,
    ) -> None:
        self._data_frame = data_frame
        self._schema_name = schema_name
        self._table_name = table_name
        self._primary_key_column = primary_key_column
        self._unique_columns = unique_columns
        self._history = history
        self._varchar_padding = varchar_padding
        self._float_precision = float_precision
        self._decimal_places = decimal_places
        self._generate_identity_column = generate_identity_column

    @staticmethod
    def create_table(
            data_frame: pd.DataFrame,
            schema_name: str,
            table_name: str,
            dialect: SqlDialect,
            primary_key_column: str | None = None,
            unique_columns: list[str] | None = None,
            history: bool = False,
            varchar_padding: int = 20,
            float_precision: int = 10,
            decimal_places: int = 2,
            generate_identity_column: bool = False,
    ) -> str:

        column_metadata = Analyzer.generate_column_metadata(
            data_frame, primary_key_column, unique_columns, decimal_places
        )

        if not column_metadata:
            return ""

        column_fragments: list[str] = []

        # Optional automatically generated identity column
        if generate_identity_column:
            column_fragments.append(dialect.identity_fragment_function(table_name))

        for column_info in column_metadata:
            column_name = column_info["column_name"]
            escaped_column = f"{dialect.opening_escape}{column_name}{dialect.closing_escape}"

            # -------------- empty column ----------------------------------------
            if column_info["is_empty"]:
                if dialect.name == "mssql":
                    logger.info("%s is empty – setting to nvarchar(max)", column_name)
                    column_fragments.append(f"{escaped_column} nvarchar(max)")
                else:  # MariaDB – silently skip empty columns
                    logger.info("%s is empty – skipping", column_name)
                continue

            # -------------- data-type mapping -----------------------------------
            data_type = column_info["data_type"]
            data_definition_clause: str | None = None

            if data_type == "datetime":
                data_definition_clause = f"{escaped_column} {dialect.datetime_type}"

            elif data_type == "float":
                precision_to_use = max(float_precision, column_info["float_precision"])
                data_definition_clause = (
                    f"{escaped_column} decimal({precision_to_use}, {decimal_places})"
                )

            elif data_type == "integer":
                smallest = column_info["smallest_num"]
                largest = column_info["biggest_num"]

                if smallest < -2147483648 or largest > 2147483648:
                    data_definition_clause = f"{escaped_column} bigint"
                if smallest >= -2147483648 and largest <= 2147483648:
                    data_definition_clause = f"{escaped_column} int"
                if smallest >= -32768 and largest <= 32768:
                    data_definition_clause = f"{escaped_column} smallint"
                if dialect.name == "mssql" and 0 <= smallest <= largest <= 255:
                    data_definition_clause = f"{escaped_column} tinyint"
                if dialect.name == "mariadb" and -128 <= smallest <= largest <= 127:
                    data_definition_clause = f"{escaped_column} tinyint"

            elif data_type == "boolean":
                data_definition_clause = f"{escaped_column} {dialect.boolean_type}"

            elif data_type == "string":
                required_length = column_info["max_str_size"] + varchar_padding
                if dialect.maximum_varchar_length is None:
                    # Microsoft SQL Server
                    if required_length >= 4000:
                        data_definition_clause = f"{escaped_column} nvarchar(max)"
                    else:
                        data_definition_clause = f"{escaped_column} nvarchar({required_length})"
                else:
                    if required_length > dialect.maximum_varchar_length:
                        logger.info(
                            f"{column_name} exceeds varchar({dialect.maximum_varchar_length}) – skipping")
                        continue
                    data_definition_clause = f"{escaped_column} varchar({required_length})"

            if data_definition_clause is None:  # Defensive guard
                continue

            # -------------- inline constraints for MSSQL -------------------------
            if column_info["is_id"]:
                data_definition_clause += dialect.primary_key_fragment_function(
                    table_name, column_name
                )
            if column_info["is_unique"]:
                data_definition_clause += dialect.unique_key_fragment_function(
                    table_name, column_name
                )

            column_fragments.append(data_definition_clause)

        if not column_fragments:
            return ""

        column_list_sql = ",\n\t".join(column_fragments)
        location = f"{schema_name}.{table_name}"
        create_statement = f"create table {location}\n(\n\t{column_list_sql}\n);"

        # -------------------- history / system-versioning (MSSQL only) -----
        if dialect.name == "mssql" and history:
            history_location = f"{schema_name}.[{table_name}_history]"
            history_column_sql = (
                f"{column_list_sql},\n"
                "\tsystem_record_start datetime2 generated always as row start\n"
                f"\t\tconstraint df_{table_name}_system_record_start\n"
                "\t\tdefault sysutcdatetime() not null,\n"
                "\tsystem_record_end datetime2 generated always as row end\n"
                f"\t\tconstraint df_{table_name}_system_record_end\n"
                "\t\tdefault sysutcdataetime() not null,\n"
                "\t\tperiod for system_time(system_record_start, system_record_end)"
            )
            create_statement = (
                f"create table {location}\n(\n{history_column_sql}\n) with \n("
                f"\tsystem_versioning = on (history_table = {history_location})\n);"
            )

        return create_statement
