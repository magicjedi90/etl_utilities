from ..dataframe.analyzer import Analyzer


class ExtraColumnsException(Exception):
    def __init__(self, extra_columns_df):
        self.extra_columns_df = extra_columns_df
        self.column_metadata = self._generate_column_metadata()

        # Create the error message
        extra_column_string_list = []
        for column in self.column_metadata:
            if column['data_type'] == 'string':
                extra_column_string = f'name: {column['column_name']} \t type: {column['data_type']} \t max_size: {column['max_str_size']}'
            elif column['data_type'] == 'float':
                extra_column_string = f'name: {column['column_name']} \t type: {column['data_type']} \t float_precision: {column['float_precision']}'
            elif column['data_type'] == 'integer':
                extra_column_string = f'name: {column['column_name']} \t type: {column['data_type']} \t biggest_number: {column['biggest_num']} \t smallest_number: {column['smallest_num']}'
            else:
                extra_column_string = f'name: {column['column_name']} \t type: {column['data_type']}'
            extra_column_string_list.append(extra_column_string)

        extra_columns_string = "\n".join(extra_column_string_list)

        message = f'The following columns need to be added:\n {extra_columns_string}'

        # Call the parent constructor with the message
        super().__init__(message)

    def _generate_column_metadata(self):
        # Move the column metadata generation logic here
        return Analyzer.generate_column_metadata(self.extra_columns_df, None, None, 0)

    def get_column_metadata(self):
        return self.column_metadata

    def get_extra_columns_df(self):
        return self.extra_columns_df
