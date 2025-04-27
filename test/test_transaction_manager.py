# test/test_transaction_manager.py

import pytest
import pandas as pd
import numpy as np
from typing import Any, Sequence, Union, Optional, List, Tuple
import logging # Import logging for caplog

# Importar las clases base y TransactionManager
from pyportfolio.transaction_manager import TransactionManager
from pyportfolio.calculators.base_calculator import BaseRowCalculator, BaseTableCalculator

# --- Mock Calculators ---

class MockRowCalculator(BaseRowCalculator):
    """Mock row-wise calculator."""
    # __init__ is inherited from BaseRowCalculator

    def calculate_row(self, row: pd.Series) -> Any:
        if 'Input' in row and pd.notna(row['Input']):
            return row['Input'] * 2
        return None

class MockMultiRowCalculator(BaseRowCalculator):
    """Mock row-wise calculator returning multiple values."""
    # __init__ is inherited from BaseRowCalculator

    def calculate_row(self, row: pd.Series) -> Union[Any, Sequence[Any]]:
        if 'Input' in row and pd.notna(row['Input']):
            return row['Input'] * 2, row['Input'] ** 2
        return None, None # Return sequence of Nones

class MockWrongLengthRowCalculator(BaseRowCalculator):
    """Intentionally returns wrong number of items for multi-column row-wise."""
    # __init__ is inherited from BaseRowCalculator

    def calculate_row(self, row: pd.Series) -> Union[Any, Sequence[Any]]:
        if 'Input' in row and pd.notna(row['Input']):
            # Returns only one item, but might be registered for two columns
            return row['Input'] * 2
        return None

class MockTableCalculator(BaseTableCalculator):
    """Mock table-wise calculator. Adds a new column 'TableOutput'."""
    # __init__ is inherited from BaseTableCalculator (which takes no args)

    def calculate_table(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = pd.DataFrame(index=df.index) # Preserve index
        if 'Input' in df.columns:
            result_df['TableOutput'] = df['Input'] + 100
        else:
            result_df['TableOutput'] = pd.NA
        return result_df

class MockTableCalculatorModify(BaseTableCalculator):
    """Mock table-wise calculator. Modifies an existing column 'Input'."""
    # __init__ is inherited from BaseTableCalculator

    def calculate_table(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = pd.DataFrame(index=df.index) # Preserve index
        if 'Input' in df.columns:
            # Return only the modified column
            result_df['Input'] = df['Input'] * 10
        return result_df

class MockTableCalculatorWrongReturn(BaseTableCalculator):
    """Mock table-wise calculator that returns wrong type."""
    # __init__ is inherited from BaseTableCalculator

    def calculate_table(self, df: pd.DataFrame) -> Any:
        return "not a dataframe" # Incorrect return type

class MockTableCalculatorMismatchedIndex(BaseTableCalculator):
    """Mock table-wise calculator that returns DataFrame with wrong index."""
    # __init__ is inherited from BaseTableCalculator

    def calculate_table(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create a DataFrame with a completely different index
        new_index = range(len(df) + 5, len(df) * 2 + 5)
        result_df = pd.DataFrame({'TableOutput': df['Input'].values + 50 if 'Input' in df else None}, index=new_index)
        return result_df

class MockCalculatorRaisesError(BaseRowCalculator):
    """Mock calculator that raises an error."""
    # __init__ is inherited from BaseRowCalculator

    # Corrected method name
    def calculate_row(self, row: pd.Series) -> Any:
        raise ValueError("Calculation failed intentionally")

class MockInvalidCalculator: # Does not inherit from BaseRow or BaseTable
    """Not a valid calculator type for the manager."""
    pass

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

def test_register_row_calculator(sample_df):
    """Test registering a BaseRowCalculator."""
    manager = TransactionManager(sample_df.copy())
    # Pass the DataFrame to the constructor
    calc = MockRowCalculator(transactions=sample_df)
    manager.register_calculation(calculator=calc, column='Output', dtype='float')

    assert len(manager._registrations) == 1
    calculator, columns, dtypes, mode = manager._registrations[0]
    assert isinstance(calculator, MockRowCalculator)
    assert columns == ['Output']
    assert dtypes == ['float']
    assert mode == 'row'

def test_register_multi_column_row_calculator(sample_df):
    """Test registering a multi-column BaseRowCalculator."""
    manager = TransactionManager(sample_df.copy())
    # Pass the DataFrame to the constructor
    calc = MockMultiRowCalculator(transactions=sample_df)
    manager.register_calculation(calculator=calc, column=['Out1', 'Out2'], dtype=['float', 'int'])

    assert len(manager._registrations) == 1
    calculator, columns, dtypes, mode = manager._registrations[0]
    assert isinstance(calculator, MockMultiRowCalculator)
    assert columns == ['Out1', 'Out2']
    assert dtypes == ['float', 'int']
    assert mode == 'row'

def test_register_table_calculator(sample_df):
    """Test registering a BaseTableCalculator (column/dtype ignored)."""
    manager = TransactionManager(sample_df.copy())
    # BaseTableCalculator constructor takes no arguments
    calc = MockTableCalculator()
    # Pass column/dtype, they should be ignored
    manager.register_calculation(calculator=calc, column='Ignored', dtype='Ignored')

    assert len(manager._registrations) == 1
    calculator, columns, dtypes, mode = manager._registrations[0]
    assert isinstance(calculator, MockTableCalculator)
    assert columns is None # Stored as None for table-wise
    assert dtypes is None  # Stored as None for table-wise
    assert mode == 'table'

def test_register_row_calculator_error_missing_column(sample_df):
    """Test error registering BaseRowCalculator without column."""
    manager = TransactionManager(sample_df.copy())
    # Pass the DataFrame to the constructor
    calc = MockRowCalculator(transactions=sample_df)
    with pytest.raises(ValueError, match="Argument 'column' is required for row-wise"):
        manager.register_calculation(calculator=calc) # Missing column

def test_register_error_invalid_calculator_type(sample_df):
    """Test error registering an object not inheriting from valid bases."""
    manager = TransactionManager(sample_df.copy())
    calc = MockInvalidCalculator()
    with pytest.raises(TypeError, match="Calculator must inherit from BaseRowCalculator or BaseTableCalculator"):
        manager.register_calculation(calculator=calc)

def test_register_row_calculator_error_invalid_column_type(sample_df):
    """Test error if column is not str or sequence for row calculator."""
    manager = TransactionManager(sample_df.copy())
    # Pass the DataFrame to the constructor
    calc = MockRowCalculator(transactions=sample_df)
    with pytest.raises(TypeError, match="For row-wise calculators, 'column' must be a string or a sequence"):
        manager.register_calculation(calculator=calc, column=123)

def test_register_row_calculator_error_dtype_sequence_mismatch(sample_df):
    """Test error if dtype sequence length mismatch for row calculator."""
    manager = TransactionManager(sample_df.copy())
    # Pass the DataFrame to the constructor
    calc = MockMultiRowCalculator(transactions=sample_df)
    with pytest.raises(ValueError, match="length .* must match the number of columns"):
        manager.register_calculation(calculator=calc, column=['Out1', 'Out2'], dtype=['float'])

# === Processing Tests ===

def test_process_row_calculator(sample_df):
    """Test processing a single registered row-wise calculator."""
    manager = TransactionManager(sample_df.copy())
    # Pass the DataFrame to the constructor
    calc = MockRowCalculator(transactions=manager.transactions) # Pass manager's df
    manager.register_calculation(calculator=calc, column='Output', dtype='float')
    manager.process_all()

    expected = pd.Series([2.0, 4.0, np.nan, 8.0], name='Output')
    pd.testing.assert_series_equal(manager.transactions['Output'], expected, check_dtype=False)
    assert pd.api.types.is_float_dtype(manager.transactions['Output'])

def test_process_multi_column_row_calculator(sample_df):
    """Test processing multiple columns from one row-wise calculator."""
    manager = TransactionManager(sample_df.copy())
    # Pass the DataFrame to the constructor
    calc = MockMultiRowCalculator(transactions=manager.transactions) # Pass manager's df
    manager.register_calculation(calculator=calc, column=['Out1', 'Out2'], dtype=['float', 'Int64'])
    manager.process_all()

    expected_out1 = pd.Series([2.0, 4.0, np.nan, 8.0], name='Out1')
    expected_out2 = pd.Series([1, 4, pd.NA, 16], name='Out2', dtype='Int64')

    pd.testing.assert_series_equal(manager.transactions['Out1'], expected_out1, check_dtype=False)
    pd.testing.assert_series_equal(manager.transactions['Out2'], expected_out2, check_dtype=True)

def test_process_table_calculator_adds_column(sample_df):
    """Test processing a table-wise calculator that adds a new column."""
    manager = TransactionManager(sample_df.copy())
    calc = MockTableCalculator() # No args needed
    manager.register_calculation(calculator=calc)
    manager.process_all()

    assert 'TableOutput' in manager.transactions.columns
    expected = pd.Series([101.0, 102.0, np.nan, 104.0], name='TableOutput') # Input + 100
    pd.testing.assert_series_equal(manager.transactions['TableOutput'], expected, check_dtype=False)

def test_process_table_calculator_modifies_column(sample_df):
    """Test processing a table-wise calculator that modifies an existing column."""
    manager = TransactionManager(sample_df.copy())
    calc = MockTableCalculatorModify() # No args needed
    manager.register_calculation(calculator=calc)
    manager.process_all()

    expected = pd.Series([10.0, 20.0, np.nan, 40.0], name='Input') # Input * 10
    pd.testing.assert_series_equal(manager.transactions['Input'], expected, check_dtype=False)

def test_process_mixed_calculators(sample_df):
    """Test processing a mix of row-wise and table-wise calculators."""
    manager = TransactionManager(sample_df.copy())
    # Pass manager's df to row calculators
    calc_row1 = MockRowCalculator(transactions=manager.transactions)
    calc_table = MockTableCalculator() # No args needed
    calc_row2 = MockMultiRowCalculator(transactions=manager.transactions)

    manager.register_calculation(calculator=calc_row1, column='Row1Out', dtype='float')
    manager.register_calculation(calculator=calc_table) # Table calc runs second
    # This row calc runs *after* the table calc
    manager.register_calculation(calculator=calc_row2, column=['Row2Out1', 'Row2Out2'], dtype='Int64')

    manager.process_all()

    # Check Row1Out (based on original Input)
    expected_row1 = pd.Series([2.0, 4.0, np.nan, 8.0], name='Row1Out')
    pd.testing.assert_series_equal(manager.transactions['Row1Out'], expected_row1, check_dtype=False)

    # Check TableOutput (based on original Input)
    expected_table = pd.Series([101.0, 102.0, np.nan, 104.0], name='TableOutput')
    pd.testing.assert_series_equal(manager.transactions['TableOutput'], expected_table, check_dtype=False)

    # Check Row2 outputs (based on original Input)
    expected_row2_1 = pd.Series([2, 4, pd.NA, 8], name='Row2Out1', dtype='Int64')
    expected_row2_2 = pd.Series([1, 4, pd.NA, 16], name='Row2Out2', dtype='Int64')
    pd.testing.assert_series_equal(manager.transactions['Row2Out1'], expected_row2_1, check_dtype=True)
    pd.testing.assert_series_equal(manager.transactions['Row2Out2'], expected_row2_2, check_dtype=True)

    # Check original 'Input' column is unchanged by row calcs, but might be by table calcs if designed so
    # In this specific test, MockTableCalculator adds a column, MockTableCalculatorModify (not used here) would change it.
    expected_input = pd.Series([1.0, 2.0, np.nan, 4.0], name='Input')
    pd.testing.assert_series_equal(manager.transactions['Input'], expected_input, check_dtype=False)


def test_process_error_row_calculator_wrong_return_length(sample_df):
    """Test ValueError if row calculator returns wrong number of items."""
    manager = TransactionManager(sample_df.copy())
    # Pass the DataFrame to the constructor
    calc = MockWrongLengthRowCalculator(transactions=manager.transactions)
    manager.register_calculation(calculator=calc, column=['Out1', 'Out2']) # Expects 2 items

    with pytest.raises(ValueError, match="expected to return a sequence of 2 items"):
        manager.process_all()

def test_process_error_table_calculator_wrong_return_type(sample_df):
    """Test TypeError if table calculator returns non-DataFrame."""
    manager = TransactionManager(sample_df.copy())
    calc = MockTableCalculatorWrongReturn() # No args needed
    manager.register_calculation(calculator=calc)

    with pytest.raises(TypeError, match="Table calculator .* must return a pandas DataFrame"):
        manager.process_all()

def test_process_warning_table_calculator_mismatched_index(sample_df, caplog):
    """Test warning logged if table calculator returns mismatched index."""
    manager = TransactionManager(sample_df.copy())
    calc = MockTableCalculatorMismatchedIndex() # No args needed
    manager.register_calculation(calculator=calc)

    with caplog.at_level(logging.WARNING):
        manager.process_all()

    assert "Index of DataFrame returned by table calculator MockTableCalculatorMismatchedIndex does not match" in caplog.text
    # The column should exist but likely contain NaNs due to reindexing failure
    assert 'TableOutput' in manager.transactions.columns
    assert manager.transactions['TableOutput'].isna().all() # Expect all NaN due to index mismatch

def test_process_calculator_raises_exception(sample_df):
    """Test that an exception raised by a calculator stops processing and is re-raised."""
    manager = TransactionManager(sample_df.copy())
    # Pass manager's df to row calculators
    calc_ok = MockRowCalculator(transactions=manager.transactions)
    calc_bad = MockCalculatorRaisesError(transactions=manager.transactions)
    calc_never_runs = MockTableCalculator() # No args needed

    manager.register_calculation(calculator=calc_ok, column='OKOut')
    manager.register_calculation(calculator=calc_bad, column='BadOut') # This one will fail
    manager.register_calculation(calculator=calc_never_runs)

    with pytest.raises(ValueError, match="Calculation failed intentionally"):
        manager.process_all()

    # Check that the first calculation ran, but the later ones didn't
    assert 'OKOut' in manager.transactions.columns
    assert 'BadOut' not in manager.transactions.columns # Failed during apply/assignment
    assert 'TableOutput' not in manager.transactions.columns # Never reached

def test_process_empty_dataframe_mixed(empty_df):
    """Test processing empty DataFrame with mixed calculators."""
    manager = TransactionManager(empty_df.copy())
    # Pass manager's df (which is empty) to row calculator
    calc_row = MockRowCalculator(transactions=manager.transactions)
    calc_table = MockTableCalculator() # No args needed

    manager.register_calculation(calculator=calc_row, column='RowOut', dtype='float')
    manager.register_calculation(calculator=calc_table) # Should run even if df is empty

    manager.process_all() # Should not raise error

    # Row calculator column should be initialized
    assert 'RowOut' in manager.transactions.columns
    assert pd.api.types.is_float_dtype(manager.transactions['RowOut'])
    assert manager.transactions['RowOut'].empty

    # Table calculator column should also be initialized (or added)
    assert 'TableOutput' in manager.transactions.columns
    assert manager.transactions['TableOutput'].empty
    # Check original 'Input' column still exists and is empty
    assert 'Input' in manager.transactions.columns
    assert manager.transactions['Input'].empty
