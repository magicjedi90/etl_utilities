# src/etl/database/exceptions.py
"""Exceptions raised by pre-upload validation."""

from ..dataframe.analyzer import Analyzer


class ColumnDataException(Exception):
    """Raised when DataFrame columns mismatch or would truncate against the target table."""


class ExtraColumnsException(Exception):
    """Raised when the DataFrame has columns the target table lacks.

    Carries the offending columns and their metadata so callers can generate
    the missing DDL.
    """

    def __init__(self, extra_columns_df, column_metadata=None):
        self.extra_columns_df = extra_columns_df
        if column_metadata is None:
            column_metadata = Analyzer.generate_column_metadata(extra_columns_df, None, None, 0)
        self.column_metadata = column_metadata
        super().__init__(
            f"The following columns need to be added:\n {self._format_columns(column_metadata)}"
        )

    @staticmethod
    def _format_columns(column_metadata) -> str:
        lines = []
        for column in column_metadata:
            line = f"name: {column['column_name']} \t type: {column['data_type']}"
            if column['data_type'] == 'string':
                line += f" \t max_size: {column['max_str_size']}"
            elif column['data_type'] == 'float':
                line += f" \t float_precision: {column['float_precision']}"
            elif column['data_type'] == 'integer':
                line += (
                    f" \t biggest_number: {column['biggest_num']}"
                    f" \t smallest_number: {column['smallest_num']}"
                )
            lines.append(line)
        return "\n".join(lines)

    def get_column_metadata(self):
        return self.column_metadata

    def get_extra_columns_df(self):
        return self.extra_columns_df
