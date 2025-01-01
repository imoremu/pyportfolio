# tests/test_transaction_manager.py

import pytest

import pandas as pd
import numpy as np

from unittest.mock import MagicMock
from pyportfolio.transaction_manager import TransactionManager
from pyportfolio.calculators.base_calculator import BaseCalculator


def test_single_calculator_returns_value_for_every_row():
    """
    Single calculator that always returns a constant (e.g., 5) for every row.
    We expect each row in the DataFrame to have a new column with this value.
    """
    transactions = pd.DataFrame([
        {"TransactionType": "buy"},
        {"TransactionType": "sell"},
        {"TransactionType": "dividend"}
    ])
    manager = TransactionManager(transactions)

    # Create a test double for the calculator that always returns 5
    calculator_double = MagicMock(spec=BaseCalculator)
    calculator_double.calculate.side_effect = [5, 5, 5]  # always 5 for each row

    manager.register_calculation("CalculatedValue", calculator_double)
    manager.process_all()

    # Check that each row has "CalculatedValue" == 5
    for idx, row in transactions.iterrows():
        assert row["CalculatedValue"] == 5

    # Verify one calculate() call per row
    assert calculator_double.calculate.call_count == len(transactions)


def test_single_calculator_always_returns_none():
    """
    Single calculator that always returns None for every row.
    We expect no new column to be added to the DataFrame, 
    because None means 'no value to store'.
    """
    transactions = pd.DataFrame([
        {"TransactionType": "buy"},
        {"TransactionType": "sell"}
    ])
    manager = TransactionManager(transactions)

    calculator_double = MagicMock(spec=BaseCalculator)
    calculator_double.calculate.side_effect = [None, None]

    # Now we expect the column to exist but hold None values in each row
    manager.register_calculation("NoColumn", calculator_double)
    manager.process_all()

    for idx, row in transactions.iterrows():
        assert "NoColumn" in row
        assert row["NoColumn"] is None

    assert calculator_double.calculate.call_count == len(transactions)


def test_multiple_calculators_different_columns():
    """
    Multiple calculators, each providing a different column:
      - Calculator A returns 10 for the first row and None for the second row.
      - Calculator B always returns 100.
    We verify that each row is updated according to the calculator's behavior.
    """
    transactions = pd.DataFrame([
        {"TransactionType": "buy"},
        {"TransactionType": "sell"}
    ])
    manager = TransactionManager(transactions)

    calc_a = MagicMock(spec=BaseCalculator)
    calc_a.calculate.side_effect = [10, np.nan]

    calc_b = MagicMock(spec=BaseCalculator)
    calc_b.calculate.side_effect = [100, 100]

    manager.register_calculation("ColumnA", calc_a, dtype="object")
    manager.register_calculation("ColumnB", calc_b)
    manager.process_all()

    # First row: ColumnA => 10, ColumnB => 100
    assert transactions.loc[0, "ColumnA"] == 10
    assert transactions.loc[0, "ColumnB"] == 100

    # Second row: ColumnA => None => no column added; ColumnB => 100
    assert pd.isna(transactions.loc[1, "ColumnA"])
    assert transactions.loc[1, "ColumnB"] == 100

    assert calc_a.calculate.call_count == 2
    assert calc_b.calculate.call_count == 2


def test_multiple_rows_mixed_returns():
    """
    A single calculator over multiple rows with a mix of None and non-None returns.
    We confirm only rows that receive a non-None value get updated.
    """
    transactions = pd.DataFrame([
        {"TransactionType": "buy"},
        {"TransactionType": "buy"},
        {"TransactionType": "sell"},
        {"TransactionType": "sell"},
        {"TransactionType": "dividend"}
    ])
    manager = TransactionManager(transactions)

    # This corresponds to the 5 rows:
    #   [None, 5, None, 10, None]
    calc_double = MagicMock(spec=BaseCalculator)
    calc_double.calculate.side_effect = [np.nan, 5, np.nan, 10, np.nan]

    manager.register_calculation("MixedColumn", calc_double)
    manager.process_all()

    # Row 0 => None => no "MixedColumn"
    assert pd.isna(transactions.loc[0, "MixedColumn"])

    # Row 1 => 5 => "MixedColumn" == 5
    assert transactions.loc[1, "MixedColumn"] == 5

    # Row 2 => None => no "MixedColumn"
    assert pd.isna(transactions.loc[2, "MixedColumn"])

    # Row 3 => 10 => "MixedColumn" == 10
    assert transactions.loc[3, "MixedColumn"] == 10

    # Row 4 => None => no "MixedColumn"
    assert pd.isna(transactions.loc[4, "MixedColumn"])

    assert calc_double.calculate.call_count == len(transactions)


def test_calculator_raises_exception():
    """
    A calculator that raises an exception when a certain condition is met.
    We verify how the manager behaves when an exception occurs.
    By default, we expect the exception to propagate (unless the manager
    is designed to handle it).
    """
    transactions = pd.DataFrame([
        {"TransactionType": "buy"},
        {"TransactionType": "sell"}
    ])
    manager = TransactionManager(transactions)

    calc_double = MagicMock(spec=BaseCalculator)
    calc_double.calculate.side_effect = [10, Exception("Simulated error")]

    manager.register_calculation("PotentialErrorColumn", calc_double)

    # If the default behavior is to let the exception propagate:
    with pytest.raises(Exception) as exc_info:
        manager.process_all()

    # Check the exception message
    assert str(exc_info.value) == "Simulated error"

    # Depending on the implementation, the first row may or may not have been set:
    # assert transactions.loc[0, "PotentialErrorColumn"] == 10
    # Or not, if the manager stops immediately after the first exception.


def test_calculator_filters_by_transaction_type():
    """
    A calculator that returns a value only for rows where 'TransactionType' == 'buy',
    and None for all other rows. We confirm that only 'buy' rows are updated.
    """
    transactions = pd.DataFrame([
        {"TransactionType": "buy"},
        {"TransactionType": "sell"},
        {"TransactionType": "dividend"},
        {"TransactionType": "buy"}
    ])
    manager = TransactionManager(transactions)

    calc_double = MagicMock(spec=BaseCalculator)

    def filter_func(row):
        if row.get("TransactionType") == "buy":
            return 999
        return None

    calc_double.calculate.side_effect = filter_func

    manager.register_calculation("TypeCheckColumn", calc_double)
    manager.process_all()

    # Only rows 0 and 3 have "buy"
    assert transactions.loc[0, "TypeCheckColumn"] == 999
    assert pd.isna(transactions.loc[1, "TypeCheckColumn"])
    assert pd.isna(transactions.loc[2, "TypeCheckColumn"])
    assert transactions.loc[3, "TypeCheckColumn"] == 999

    assert calc_double.calculate.call_count == len(transactions)


def test_calls_per_row():
    """
    Verifies that the TransactionManager calls the calculator exactly once per row.
    We use a short DataFrame of transactions and confirm the number of calls.
    """
    transactions = pd.DataFrame([
        {"TransactionType": "buy"},
        {"TransactionType": "buy"},
        {"TransactionType": "sell"}
    ])
    manager = TransactionManager(transactions)

    calc_double = MagicMock(spec=BaseCalculator)
    # Return None so we only count the calls
    calc_double.calculate.return_value = None

    manager.register_calculation("RowCountColumn", calc_double)
    manager.process_all()

    # We have 3 rows => the calculator should have been called 3 times
    assert calc_double.calculate.call_count == 3