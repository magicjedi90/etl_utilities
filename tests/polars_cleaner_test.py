#!/usr/bin/env python3
"""
Test suite for Polars DataFrame Cleaner
"""
import datetime

import pytest
import polars as pl

from src.etl.dataframe.cleaner import standardize_column_name, compute_hash
from src.etl.dataframe.polars.cleaner import PolarsCleaner

class TestPolarsCleaner:
    """Test cases for PolarsCleaner class"""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing"""
        return pl.DataFrame({
            'Customer Name': ['John Doe', 'Jane Smith', 'Bob Johnson', None, 'Alice Brown'],
            'Age': ['25.0', '30.0', '35', '', None],
            'Salary': ['$50,000', '65000', '75000.50', '$80,000', ''],
            'Is Active': ['Yes', 'No', '1', '0', 'true'],
            'Join Date': ['2023-01-15', '2023/02/20', 'March 15, 2023', '2023-04-10', None],
            'Performance Score': ['85%', '92.5%', '78', '88.5%', None]
        })

    def test_standardize_column_name(self):
        """Test column name standardization"""
        assert standardize_column_name('Customer Name') == 'customer_name'
        assert standardize_column_name('First Name') == 'first_name'
        assert standardize_column_name('Total$Amount') == 'total_dollars_amount'
        assert standardize_column_name('Is Active?') == 'is_active'
        assert standardize_column_name('User&ID') == 'user_and_id'

    def test_compute_hash(self):
        """Test hash computation"""
        hash1 = compute_hash('test_string')
        hash2 = compute_hash('test_string')
        hash3 = compute_hash('different_string')

        assert len(hash1) == 40  # SHA-1 produces 40 character hex string
        assert hash1 == hash2  # Same input produces same hash
        assert hash1 != hash3  # Different input produces different hash

    def test_column_names_to_snake_case(self, sample_dataframe):
        """Test conversion to snake_case"""
        df = PolarsCleaner.column_names_to_snake_case(sample_dataframe)
        expected_columns = [
            'customer_name', 'age', 'salary', 'is_active',
            'join_date', 'performance_score'
        ]
        assert df.columns == expected_columns

    def test_column_names_to_pascal_case(self, sample_dataframe):
        """Test conversion to PascalCase"""
        df = PolarsCleaner.column_names_to_pascal_case(sample_dataframe)
        expected_columns = [
            'CustomerName', 'Age', 'Salary', 'IsActive',
            'JoinDate', 'PerformanceScore'
        ]
        assert df.columns == expected_columns

    def test_clean_numbers(self, sample_dataframe):
        """Test numeric cleaning: dtypes, values, and null handling"""
        df = PolarsCleaner.clean_numbers(sample_dataframe, ['Age', 'Salary', 'Performance Score'])

        # parse_integer_expr returns Float64 (when/then/otherwise supertype)
        assert df['Age'].dtype == pl.Float64
        assert df['Salary'].dtype == pl.Float64
        assert df['Performance Score'].dtype == pl.Float64

        # Age: '25.0' -> 25.0, '30.0' -> 30.0, '35' -> 35.0, '' -> None, None -> None
        assert df['Age'].to_list() == [25.0, 30.0, 35.0, None, None]

        # Salary: '$50,000' -> 50000.0, '65000' -> 65000.0, '75000.50' -> 75000.5,
        #         '$80,000' -> 80000.0, '' -> None
        assert df['Salary'].to_list() == [50000.0, 65000.0, 75000.5, 80000.0, None]

        # Performance Score: '85%' -> 85.0, '92.5%' -> 92.5, '78' -> 78.0,
        #                    '88.5%' -> 88.5, None -> None
        assert df['Performance Score'].to_list() == [85.0, 92.5, 78.0, 88.5, None]


    def test_clean_bools(self, sample_dataframe):
        """Test boolean cleaning: dtype and values"""
        df = PolarsCleaner.clean_bools(sample_dataframe, ['Is Active'])

        assert df['Is Active'].dtype == pl.Boolean
        assert df['Is Active'][0] is True   # "Yes" -> True
        assert df['Is Active'][1] is False  # "No" -> False
        assert df['Is Active'][2] is True   # "1" -> True
        assert df['Is Active'][3] is False  # "0" -> False
        assert df['Is Active'][4] is True   # "true" -> True

    def test_clean_dates(self, sample_dataframe):
        """Test date cleaning: dtype and all values"""
        df = PolarsCleaner.clean_dates(sample_dataframe, ['Join Date'])

        assert df['Join Date'].dtype == pl.Datetime
        assert df['Join Date'][0] == datetime.datetime(2023, 1, 15, 0, 0)
        assert df['Join Date'][1] == datetime.datetime(2023, 2, 20, 0, 0)
        assert df['Join Date'][2] == datetime.datetime(2023, 3, 15, 0, 0)
        assert df['Join Date'][3] == datetime.datetime(2023, 4, 10, 0, 0)
        assert df['Join Date'][4] is None

    def test_clean_all_types(self, sample_dataframe):
        """Test comprehensive cleaning: exact dtypes and values for every column"""
        df = PolarsCleaner.column_names_to_snake_case(sample_dataframe)
        df = PolarsCleaner.clean_all_types(df)

        # Columns preserved
        assert df.columns == ['customer_name', 'age', 'salary', 'is_active', 'join_date', 'performance_score']

        # Exact dtypes
        assert df['customer_name'].dtype == pl.String
        assert df['is_active'].dtype == pl.Boolean
        assert df['join_date'].dtype == pl.Datetime
        assert df['age'].dtype == pl.Float64
        assert df['salary'].dtype == pl.Float64
        assert df['performance_score'].dtype == pl.Float64

        # Boolean values
        assert df['is_active'].to_list() == [True, False, True, False, True]

        # Date values
        assert df['join_date'][0] == datetime.datetime(2023, 1, 15)
        assert df['join_date'][1] == datetime.datetime(2023, 2, 20)
        assert df['join_date'][2] == datetime.datetime(2023, 3, 15)
        assert df['join_date'][3] == datetime.datetime(2023, 4, 10)
        assert df['join_date'][4] is None

        # Numeric values
        assert df['age'].to_list() == [25.0, 30.0, 35.0, None, None]
        assert df['salary'].to_list() == [50000.0, 65000.0, 75000.5, 80000.0, None]
        assert df['performance_score'].to_list() == [85.0, 92.5, 78.0, 88.5, None]

        # String values unchanged
        assert df['customer_name'].to_list() == ['John Doe', 'Jane Smith', 'Bob Johnson', None, 'Alice Brown']

    def test_numbers_with_empty_and_whitespace_strings(self):
        """Ensure empty and whitespace-only strings become nulls in numeric columns."""
        df = pl.DataFrame({
            'n1': ['123', '', '   ', None, '4,567'],
            'n2': ['0', '  ', '\t', '', '$1,000']
        })

        cleaned = PolarsCleaner.clean_numbers(df, ['n1', 'n2'])

        assert cleaned['n1'].dtype == pl.Float64
        assert cleaned['n2'].dtype == pl.Float64

        assert cleaned['n1'].to_list() == [123.0, None, None, None, 4567.0]
        assert cleaned['n2'].to_list() == [0.0, None, None, None, 1000.0]

    def test_clean_bools_with_empty_strings(self):
        """Empty and whitespace-only strings become null in boolean columns."""
        df = pl.DataFrame({
            'bools': ['yes', '', 'no', '   ', None]
        })
        cleaned = PolarsCleaner.clean_bools(df, ['bools'])

        assert cleaned['bools'].dtype == pl.Boolean
        assert cleaned['bools'][0] is True
        assert cleaned['bools'][1] is None   # '' -> None
        assert cleaned['bools'][2] is False
        assert cleaned['bools'][3] is None   # '   ' -> None
        assert cleaned['bools'][4] is None   # None -> None

    def test_clean_dates_with_empty_strings(self):
        """Empty and whitespace-only strings become null in date columns."""
        df = pl.DataFrame({
            'dates': ['2021-01-01', '', '2021-02-02', '   ', None]
        })
        cleaned = PolarsCleaner.clean_dates(df, ['dates'])

        assert cleaned['dates'].dtype == pl.Datetime
        assert cleaned['dates'][0] == datetime.datetime(2021, 1, 1)
        assert cleaned['dates'][1] is None
        assert cleaned['dates'][2] == datetime.datetime(2021, 2, 2)
        assert cleaned['dates'][3] is None
        assert cleaned['dates'][4] is None

    def test_clean_all_types_with_empty_numeric_strings(self):
        """clean_all_types should choose numeric dtype when empties exist and keep them null."""
        df = pl.DataFrame({
            'age': ['25', '', '30', '   ', None],
            'name': ['A', 'B', 'C', 'D', 'E']
        })

        cleaned = PolarsCleaner.clean_all_types(df)

        assert cleaned['age'].dtype == pl.Float64
        assert cleaned['age'].to_list() == [25.0, None, 30.0, None, None]
        assert cleaned['name'].dtype == pl.String

    def test_clean_all_types_with_empty_strings(self):
        """clean_all_types picks correct dtype for each category when empties are present."""
        df = pl.DataFrame({
            'numbers': ['100', '', '200', '   '],
            'bools': ['yes', '', 'no', '   '],
            'dates': ['2021-01-01', '', '2021-02-02', '   '],
        })
        cleaned = PolarsCleaner.clean_all_types(df)

        assert cleaned['numbers'].dtype == pl.Float64
        assert cleaned['bools'].dtype == pl.Boolean
        assert cleaned['dates'].dtype == pl.Datetime

        assert cleaned['numbers'].to_list() == [100.0, None, 200.0, None]
        assert cleaned['bools'].to_list() == [True, None, False, None]
        assert cleaned['dates'][0] == datetime.datetime(2021, 1, 1)
        assert cleaned['dates'][1] is None
        assert cleaned['dates'][2] == datetime.datetime(2021, 2, 2)
        assert cleaned['dates'][3] is None

    def test_mixed_types_stay_string(self):
        """Columns with mixed parseable and unparseable values remain String."""
        df = pl.DataFrame({'mixed': ['100', 'abc', '200']})
        cleaned = PolarsCleaner.clean_all_types(df)
        assert cleaned['mixed'].dtype == pl.String
        assert cleaned['mixed'].to_list() == ['100', 'abc', '200']

    def test_clean_df(self):
        """clean_df removes all-null columns and cleans types."""
        data = {
            'col1': [1, 2, None, 4, None],
            'col2': [None, None, None, None, None],  # All null - should be removed
            'col3': ['a', 'b', 'c', 'd', 'e']
        }
        df = pl.DataFrame(data)

        cleaned_df = PolarsCleaner.clean_df(df)

        assert cleaned_df.shape == (5, 2)
        assert 'col1' in cleaned_df.columns
        assert 'col2' not in cleaned_df.columns
        assert 'col3' in cleaned_df.columns

    def test_clean_numbers_with_empty_strings_after_cleanup(self):
        """Test that empty strings after numeric cleanup are properly handled"""
        df = pl.DataFrame({
            'amount': ['$100', '', '  ', 'N/A', '1,000', '2,500.50', None]
        })

        cleaned = PolarsCleaner.clean_numbers(df, ['amount'])

        assert cleaned['amount'].dtype == pl.Float64
        assert cleaned['amount'][0] == 100.0    # "$100" -> 100.0
        assert cleaned['amount'][1] is None     # "" -> None
        assert cleaned['amount'][2] is None     # "  " -> None
        assert cleaned['amount'][3] is None     # "N/A" -> None
        assert cleaned['amount'][4] == 1000.0   # "1,000" -> 1000.0
        assert cleaned['amount'][5] == 2500.5   # "2,500.50" -> 2500.5
        assert cleaned['amount'][6] is None     # None -> None

    def test_generate_hash_column(self, sample_dataframe):
        """Test hash column generation"""
        df = PolarsCleaner.generate_hash_column(
            sample_dataframe,
            ['Customer Name', 'Age'],
            'hash_col'
        )

        assert 'hash_col' in df.columns
        assert len(df['hash_col'][0]) == 40  # SHA-1 hash length
        assert df['hash_col'][0] != df['hash_col'][1]  # Different inputs, different hashes

    def test_coalesce_columns(self):
        """Test column coalescing"""
        data = {
            'email': ['john@email.com', None, 'bob@email.com'],
            'phone': [None, '555-0123', None],
            'backup': ['john.backup@email.com', None, None]
        }
        df = pl.DataFrame(data)

        # Test without dropping original columns
        df_keep = PolarsCleaner.coalesce_columns(df, ['email', 'phone', 'backup'], 'primary_contact', drop=False)
        assert 'primary_contact' in df_keep.columns
        assert 'email' in df_keep.columns  # Original columns kept
        assert df_keep['primary_contact'][0] == 'john@email.com'
        assert df_keep['primary_contact'][1] == '555-0123'
        assert df_keep['primary_contact'][2] == 'bob@email.com'

        # Test with dropping original columns
        df_drop = PolarsCleaner.coalesce_columns(df, ['email', 'phone', 'backup'], 'primary_contact', drop=True)
        assert 'primary_contact' in df_drop.columns
        assert 'email' not in df_drop.columns  # Original columns dropped
        assert 'phone' not in df_drop.columns
        assert 'backup' not in df_drop.columns

    def test_optimize_dtypes(self):
        """Test data type optimization"""
        data = {
            'small_int': [1, 2, 3, None] * 1000,
            'medium_int': [1000, 2000, 3000, None] * 1000,
            'large_int': [100000, 200000, 300000, None] * 1000,
            'float_col': [1.5, 2.5, 3.5, None] * 1000
        }
        df = pl.DataFrame(data)

        original_size = df.estimated_size('mb')
        df_optimized = PolarsCleaner.optimize_dtypes(df)
        optimized_size = df_optimized.estimated_size('mb')

        # Memory should be reduced
        assert optimized_size <= original_size

        # Values should remain the same
        assert df['small_int'].to_list() == df_optimized['small_int'].to_list()

    def test_edge_case_empty_dataframe(self):
        """clean_all_types on an empty DataFrame returns empty."""
        empty_df = pl.DataFrame()
        result = PolarsCleaner.clean_all_types(empty_df)
        assert result.shape == (0, 0)

    def test_edge_case_all_null_columns(self):
        """clean_df removes all-null columns, yielding empty frame when all columns are null."""
        null_df = pl.DataFrame({'col1': [None, None, None], 'col2': [None, None, None]})
        result = PolarsCleaner.clean_df(null_df)
        assert result.columns == []
        assert result.shape == (0, 0)

    def test_edge_case_single_column(self):
        """clean_numbers on a single numeric column produces correct dtype and values."""
        single_col_df = pl.DataFrame({'values': ['1', '2', '3']})
        result = PolarsCleaner.clean_numbers(single_col_df)
        assert result['values'].dtype == pl.Float64
        assert result['values'].to_list() == [1.0, 2.0, 3.0]

    def test_edge_case_all_null_column_skipped(self):
        """clean_all_types skips all-null columns and still cleans others."""
        df = pl.DataFrame({
            'int_col': ['100', '200', '300'],
            'null_col': [None, None, None],
        })
        result = PolarsCleaner.clean_all_types(df)
        assert result['int_col'].dtype == pl.Float64
        assert result['int_col'].to_list() == [100.0, 200.0, 300.0]
        assert result['null_col'].dtype == pl.Null

if __name__ == "__main__":
    pytest.main(["-v", __file__])
