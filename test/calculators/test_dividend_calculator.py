import pytest
import pandas as pd
from pyportfolio.calculators.dividend_calculator import DividendCalculator

from pyportfolio.columns import (
    SHARE_PRICE,
    SHARES,
    TRANSACTION_TYPE,
    COMISION,
    TYPE_DIVIDEND,
    TYPE_BUY,
    TYPE_SELL
)   

@pytest.fixture
def sample_dividend_transactions():
    """Provides a sample DataFrame for dividend calculator tests."""
    return pd.DataFrame([
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 10, SHARE_PRICE: 50, COMISION: 5},
        {TRANSACTION_TYPE: TYPE_DIVIDEND, SHARES: 20, SHARE_PRICE: 2, COMISION: 1}, 
        {TRANSACTION_TYPE: TYPE_SELL, SHARES: 5, SHARE_PRICE: 80, COMISION: 4},
        {TRANSACTION_TYPE: TYPE_DIVIDEND, SHARES: 15, SHARE_PRICE: 3, COMISION: 0.5}, 
    ])

def test_non_dividend_transactions_return_none(sample_dividend_transactions):
    """
    Tests that non-dividend transactions (e.g., TYPE_BUY, TYPE_SELL)
    result in None from the DividendCalculator.
    """
    calc = DividendCalculator(sample_dividend_transactions)

    # Test TYPE_BUY transaction (index 0)
    buy_result = calc.calculate_row(sample_dividend_transactions.iloc[0])
    assert buy_result is None, "Buy transaction should return None"

    # Test TYPE_SELL transaction (index 2)
    sell_result = calc.calculate_row(sample_dividend_transactions.iloc[2])
    assert sell_result is None, "Sell transaction should return None"

def test_dividend_transactions_are_calculated(sample_dividend_transactions):
    """
    Tests that 'dividend' transactions are correctly calculated.
    """
    calc = DividendCalculator(sample_dividend_transactions)

    # Test first dividend transaction (index 1)
    # Dividend = Shares * Price - Commission = 20 * 2 = 40
    # Note: Spanish legislaton does not allow to deduct comissions from dividends
    first_dividend_result = calc.calculate_row(sample_dividend_transactions.iloc[1])
    assert first_dividend_result == 40.0, "Calculation for the first dividend is incorrect"

    # Test second dividend transaction (index 3)
    # Dividend = Shares * Price = 15 * 3 = 45
    second_dividend_result = calc.calculate_row(sample_dividend_transactions.iloc[3])
    assert second_dividend_result == 45, "Calculation for the second dividend is incorrect"