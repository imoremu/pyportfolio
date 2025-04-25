import pytest
import pandas as pd
import numpy as np
from typing import Any, Sequence, Union

from pyportfolio.transaction_manager import TransactionManager
from pyportfolio.calculators.base_calculator import BaseCalculator

class MockSingleCalculator(BaseCalculator):
    """Returns a value based on 'Input' column."""
    def calculate(self, row: pd.Series) -> Any:
        if 'Input' in row and pd.notna(row['Input']):
            return row['Input'] * 2
        return None

class MockMultiCalculator(BaseCalculator):
    """Returns two values based on 'Input' column."""
    def calculate(self, row: pd.Series) -> Union[Any, Sequence[Any]]:
        if 'Input' in row and pd.notna(row['Input']):
            return row['Input'] * 2, row['Input'] ** 2
        return None, None # Return sequence of Nones

class MockWrongLengthCalculator(BaseCalculator):
    """Intentionally returns wrong number of items for multi-column."""
    def calculate(self, row: pd.Series) -> Union[Any, Sequence[Any]]:
        if 'Input' in row and pd.notna(row['Input']):
            return row['Input'] * 2 
        return None

# --- Fixtures ---
@pytest.fixture
def sample_df():
    """Provides a basic DataFrame for testing."""
    return pd.DataFrame({'Input': [1, 2, None, 4]})

@pytest.fixture
def empty_df():
    """Provides an empty DataFrame."""
    return pd.DataFrame({'Input': pd.Series(dtype='float64')}) 

# --- Test Cases ---

# === Registration Tests ===

def test_register_single_column(sample_df):
    """Test registering a single column calculation."""
    manager = TransactionManager(sample_df.copy())
    calc = MockSingleCalculator(manager.transactions)
    
    manager.register_calculation('Output', calc, dtype='float')
    
    assert len(manager._registrations) == 1
    
    calculator, columns, dtypes = manager._registrations[0]
    
    assert isinstance(calculator, MockSingleCalculator)
    assert columns == ['Output']
    assert dtypes == ['float']

def test_register_multi_column(sample_df):
    """Test registering a multi-column calculation."""
    manager = TransactionManager(sample_df.copy())
    calc = MockMultiCalculator(manager.transactions)
    
    manager.register_calculation(['Out1', 'Out2'], calc, dtype=['float', 'int'])
    
    assert len(manager._registrations) == 1
    
    calculator, columns, dtypes = manager._registrations[0]
    
    assert isinstance(calculator, MockMultiCalculator)
    assert columns == ['Out1', 'Out2']
    assert dtypes == ['float', 'int']

def test_register_multi_column_single_dtype(sample_df):
    """Test registering multi-column with one dtype applied to all."""
    manager = TransactionManager(sample_df.copy())
    calc = MockMultiCalculator(manager.transactions)
    manager.register_calculation(['Out1', 'Out2'], calc, dtype='float')
    
    assert len(manager._registrations) == 1
    
    _, columns, dtypes = manager._registrations[0]
    
    assert columns == ['Out1', 'Out2']
    assert dtypes == ['float', 'float']

def test_register_multi_column_none_dtype(sample_df):
    """Test registering multi-column with None dtype."""
    manager = TransactionManager(sample_df.copy())
    calc = MockMultiCalculator(manager.transactions)
    manager.register_calculation(['Out1', 'Out2'], calc, dtype=None)
    assert len(manager._registrations) == 1
    _, columns, dtypes = manager._registrations[0]
    assert columns == ['Out1', 'Out2']
    assert dtypes == [None, None]

def test_register_error_invalid_column_type(sample_df):
    """Test error if column is not str or sequence."""
    manager = TransactionManager(sample_df.copy())
    calc = MockSingleCalculator(manager.transactions)
    with pytest.raises(TypeError, match="column must be a string or a sequence"):
        manager.register_calculation(123, calc)

def test_register_error_invalid_column_sequence_element(sample_df):
    """Test error if sequence elements are not strings."""
    manager = TransactionManager(sample_df.copy())
    calc = MockMultiCalculator(manager.transactions)
    with pytest.raises(TypeError, match="all elements must be strings"):
        manager.register_calculation(['Out1', 2], calc)

def test_register_error_empty_column_sequence(sample_df):
    """Test error if column sequence is empty."""
    manager = TransactionManager(sample_df.copy())
    calc = MockMultiCalculator(manager.transactions)
    with pytest.raises(ValueError, match="cannot be empty"):
        manager.register_calculation([], calc)

def test_register_error_invalid_calculator(sample_df):
    """Test error if calculator is not a BaseCalculator instance."""
    manager = TransactionManager(sample_df.copy())
    with pytest.raises(TypeError, match="calculator must be an instance of BaseCalculator"):
        manager.register_calculation('Output', "not a calculator")

def test_register_error_invalid_dtype_type(sample_df):
    """Test error if dtype is not None, str, or sequence."""
    manager = TransactionManager(sample_df.copy())
    calc = MockMultiCalculator(manager.transactions)
    with pytest.raises(TypeError, match="dtype must be None, a string, or a sequence"):
        manager.register_calculation(['Out1', 'Out2'], calc, dtype=123)

def test_register_error_dtype_sequence_mismatch(sample_df):
    """Test error if dtype sequence length doesn't match column sequence length."""
    manager = TransactionManager(sample_df.copy())
    calc = MockMultiCalculator(manager.transactions)
    with pytest.raises(ValueError, match="length .* must match the number of columns"):
        manager.register_calculation(['Out1', 'Out2'], calc, dtype=['float']) # Only one dtype for two columns

def test_register_error_invalid_dtype_sequence_element(sample_df):
    """Test error if dtype sequence contains invalid types."""
    manager = TransactionManager(sample_df.copy())
    calc = MockMultiCalculator(manager.transactions)
    with pytest.raises(TypeError, match="all elements must be strings or None"):
        manager.register_calculation(['Out1', 'Out2'], calc, dtype=['float', 123])


# === Processing Tests ===

def test_process_single_column(sample_df):
    """Test processing a single registered column."""
    manager = TransactionManager(sample_df.copy())
    calc = MockSingleCalculator(manager.transactions)
    manager.register_calculation('Output', calc, dtype='float')
    manager.process_all()

    expected = pd.Series([2.0, 4.0, np.nan, 8.0], name='Output')
    pd.testing.assert_series_equal(manager.transactions['Output'], expected, check_dtype=False) # Check values
    assert pd.api.types.is_float_dtype(manager.transactions['Output']) # Check final dtype

def test_process_multi_column(sample_df):
    """Test processing multiple columns from one calculator."""
    manager = TransactionManager(sample_df.copy())
    calc = MockMultiCalculator(manager.transactions)
    # Register with specific dtypes
    manager.register_calculation(['Out1', 'Out2'], calc, dtype=['float', 'Int64']) # Use nullable Int
    manager.process_all()

    expected_out1 = pd.Series([2.0, 4.0, np.nan, 8.0], name='Out1')
    # Expected for Int64: Nones/NaNs become pd.NA
    expected_out2 = pd.Series([1, 4, pd.NA, 16], name='Out2', dtype='Int64') 

    pd.testing.assert_series_equal(manager.transactions['Out1'], expected_out1, check_dtype=False)
    pd.testing.assert_series_equal(manager.transactions['Out2'], expected_out2, check_dtype=True) # Check nullable int dtype

def test_process_multi_column_no_dtype(sample_df):
    """Test processing multi-column without specifying dtype."""
    manager = TransactionManager(sample_df.copy())
    calc = MockMultiCalculator(manager.transactions)
    manager.register_calculation(['Out1', 'Out2'], calc) # No dtype
    manager.process_all()

    # Pandas will infer dtype, likely float64 for Out1 due to potential NaNs, object or int for Out2
    expected_out1 = pd.Series([2.0, 4.0, np.nan, 8.0], name='Out1')
    expected_out2 = pd.Series([1.0, 4.0, np.nan, 16.0], name='Out2') # Likely inferred as float due to NaN possibility

    pd.testing.assert_series_equal(manager.transactions['Out1'], expected_out1, check_dtype=False)
    # Check Out2 allows for float due to inference with None -> NaN
    pd.testing.assert_series_equal(manager.transactions['Out2'], expected_out2, check_dtype=False)


def test_process_multiple_registrations(sample_df):
    """Test processing multiple single and multi-column registrations."""
    manager = TransactionManager(sample_df.copy())
    calc_single = MockSingleCalculator(manager.transactions)
    calc_multi = MockMultiCalculator(manager.transactions)

    manager.register_calculation('SingleOut', calc_single, dtype='float')
    manager.register_calculation(['Multi1', 'Multi2'], calc_multi, dtype='Int64')

    manager.process_all()

    expected_single = pd.Series([2.0, 4.0, np.nan, 8.0], name='SingleOut')
    expected_multi1 = pd.Series([2, 4, pd.NA, 8], name='Multi1', dtype='Int64')
    expected_multi2 = pd.Series([1, 4, pd.NA, 16], name='Multi2', dtype='Int64')

    pd.testing.assert_series_equal(manager.transactions['SingleOut'], expected_single, check_dtype=False)
    assert pd.api.types.is_float_dtype(manager.transactions['SingleOut'])

    pd.testing.assert_series_equal(manager.transactions['Multi1'], expected_multi1, check_dtype=True)
    pd.testing.assert_series_equal(manager.transactions['Multi2'], expected_multi2, check_dtype=True)


def test_process_error_calculator_wrong_return_length(sample_df):
    """Test ValueError if calculator returns wrong number of items for multi-column."""
    manager = TransactionManager(sample_df.copy())
    calc = MockWrongLengthCalculator(manager.transactions)
    manager.register_calculation(['Out1', 'Out2'], calc) # Expects 2 items

    with pytest.raises(ValueError, match="expected to return a sequence of 2 items"):
        manager.process_all()

def test_process_empty_dataframe(empty_df):
    """Test processing an empty DataFrame initializes columns correctly."""
    manager = TransactionManager(empty_df.copy())
    calc_single = MockSingleCalculator(manager.transactions)
    calc_multi = MockMultiCalculator(manager.transactions)

    manager.register_calculation('SingleOut', calc_single, dtype='float')
    manager.register_calculation(['Multi1', 'Multi2'], calc_multi, dtype='Int64')

    manager.process_all() # Should not raise error

    assert 'SingleOut' in manager.transactions.columns
    assert pd.api.types.is_float_dtype(manager.transactions['SingleOut'])
    assert manager.transactions['SingleOut'].empty

    assert 'Multi1' in manager.transactions.columns
    assert pd.api.types.is_integer_dtype(manager.transactions['Multi1']) # Nullable Int64
    assert manager.transactions['Multi1'].empty

    assert 'Multi2' in manager.transactions.columns
    assert pd.api.types.is_integer_dtype(manager.transactions['Multi2']) # Nullable Int64
    assert manager.transactions['Multi2'].empty

def test_process_dtype_conversion_warning(sample_df, capsys):
    """Test that a warning is printed if dtype conversion fails (e.g., string to float)."""
    
    class MockStringCalc(BaseCalculator):
         def calculate(self, row: pd.Series) -> Any:
             return "not a number" if row['Input'] == 1 else 5.0

    manager = TransactionManager(sample_df.copy())
    calc = MockStringCalc(manager.transactions)
    manager.register_calculation('Output', calc, dtype='float') # Try to force float
    
    manager.process_all()
    
    captured = capsys.readouterr()
    assert "Warning: Could not convert column 'Output' to dtype 'float'" in captured.out
    # Check that the column still exists, likely with object dtype
    assert 'Output' in manager.transactions.columns
    assert manager.transactions['Output'].dtype == 'object' 
    pd.testing.assert_series_equal(
        manager.transactions['Output'], 
        pd.Series(["not a number", 5.0, 5.0, 5.0], name='Output'), # Assuming None input maps to 5.0 here
        check_names=False
    )