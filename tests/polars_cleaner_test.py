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
            'Age': ['25.0', '30.0', '35', '40', None],
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
        """Test numeric cleaning"""
        df = PolarsCleaner.clean_numbers(sample_dataframe, ['Age', 'Salary', 'Performance Score'])

        # Check that numeric columns are properly parsed
        assert df['Age'][0] == 25.0  # "25" -> 25.0
        assert df['Salary'][0] == 50000.0  # "$50,000" -> 50000.0

    
    def test_clean_bools(self, sample_dataframe):
        """Test boolean cleaning"""
        df = PolarsCleaner.clean_bools(sample_dataframe, ['Is Active'])
        
        assert df['Is Active'][0] == True  # "Yes" -> True
        assert df['Is Active'][1] == False  # "No" -> False
        assert df['Is Active'][2] == True  # "1" -> True
        assert df['Is Active'][3] == False  # "0" -> None
        assert df['Is Active'][4] == True  # "true" -> True
    
    def test_clean_dates(self, sample_dataframe):
        """Test date cleaning"""
        df = PolarsCleaner.clean_dates(sample_dataframe, ['Join Date'])

        # Should parse various date formats
        assert df['Join Date'][0] == datetime.datetime(2023,1,15,0,0)  # "2023-01-15"
        assert df['Join Date'][1] == datetime.datetime(2023,2,20,0,0)  # "2023/02/20"
        assert df['Join Date'][2] == datetime.datetime(2023,3,15,0,0) # "March 15, 2023"
        assert df['Join Date'][4] is None  # None remains None
    
    def test_clean_all_types(self, sample_dataframe):
        """Test comprehensive cleaning"""
        df = PolarsCleaner.column_names_to_snake_case(sample_dataframe)
        df = PolarsCleaner.clean_all_types(df)

        # Columns preserved
        assert df.columns == ['customer_name','age','salary','is_active','join_date','performance_score']

        # Types: is_active should be Boolean, join_date should be Datetime, numbers for age/salary/performance
        assert df['is_active'].dtype == pl.Boolean
        assert df['join_date'].dtype in (pl.Datetime,)
        assert df['age'].dtype in (pl.Int64, pl.Float64)
        assert df['salary'].dtype in (pl.Int64, pl.Float64)
        assert df['performance_score'].dtype in (pl.Int64, pl.Float64)
        # customer_name should remain Utf8
        assert df['customer_name'].dtype == pl.Utf8

        # Values: booleans parsed
        assert df['is_active'][0] is True
        assert df['is_active'][1] is False
        assert df['is_active'][2] is True
        assert df['is_active'][3] is False
        assert df['is_active'][4] is True

        # Dates parsed to consistent datetimes
        import datetime
        assert df['join_date'][0] == datetime.datetime(2023,1,15)
        assert df['join_date'][1] == datetime.datetime(2023,2,20)
        assert df['join_date'][2] == datetime.datetime(2023,3,15)
        assert df['join_date'][4] is None

        # Numeric parsing and cleanup
        # Age: '25' -> 25 (int), '30.5' -> 30.5 (float), 'thirty-five' -> None, '40' -> 40 (int)
        assert df['age'][0] in (25, 25.0)
        assert df['age'][1] == 30.0
        assert df['age'][3] in (40, 40.0)

        # Salary: '$50,000' -> 50000, None remains None
        assert df['salary'][0] in (50000, 50000.0)
        assert df['salary'][4] is None

        # Performance score: '85%' -> 85, '92.5%' -> 92.5
        assert df['performance_score'][0] in (85, 85.0)
        assert df['performance_score'][1] == 92.5
        assert df['performance_score'][2] in (78, 78.0)
    
    def test_clean_df(self):
        """Test DataFrame cleaning with empty rows/columns"""
        data = {
            'col1': [1, 2, None, 4, None],
            'col2': [None, None, None, None, None],  # All null - should be removed
            'col3': ['a', 'b', 'c', 'd', 'e']
        }
        df = pl.DataFrame(data)
        
        cleaned_df = PolarsCleaner.clean_df(df)
        
        # All-null column should be removed
        assert 'col2' not in cleaned_df.columns
        # Empty rows should be removed
        assert cleaned_df.height <= df.height
    
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
        
        # Test with dropping original columns
        df_drop = PolarsCleaner.coalesce_columns(df, ['email', 'phone', 'backup'], 'primary_contact', drop=True)
        assert 'primary_contact' in df_drop.columns
        assert 'email' not in df_drop.columns  # Original columns dropped
    
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

    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty DataFrame
        empty_df = pl.DataFrame()
        result = PolarsCleaner.clean_all_types(empty_df)
        assert result.shape == (0, 0)
        
        # DataFrame with all null values
        null_df = pl.DataFrame({'col1': [None, None, None], 'col2': [None, None, None]})
        result = PolarsCleaner.clean_df(null_df)
        # Should remove all-null columns and rows
        assert result.shape[0] <= null_df.shape[0]
        
        # Single column DataFrame
        single_col_df = pl.DataFrame({'values': ['1', '2', '3']})
        result = PolarsCleaner.clean_numbers(single_col_df)
        assert result.shape == single_col_df.shape

if __name__ == "__main__":
    pytest.main(["-v", __file__])