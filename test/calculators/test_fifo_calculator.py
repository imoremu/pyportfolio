import pytest
import pandas as pd
from pyportfolio.calculators.fifo_calculator import FIFOCalculator
from pyportfolio.columns import AVAILABLE_SHARES, SHARE_PRICE, SHARES, TRANSACTION_TYPE, TYPE_BUY, TYPE_SELL

def test_fifo_simple():
    """
    Basic scenario where we have 2 buy transactions and 1 sell transaction.
    The sell should consume shares from the first buy in FIFO order.
    """
    transactions = pd.DataFrame([
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 10, SHARE_PRICE: 100},
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 5, SHARE_PRICE: 120},
        {TRANSACTION_TYPE: TYPE_SELL, SHARES: 8, SHARE_PRICE: 150},
    ])
    calc = FIFOCalculator(transactions)

    result = calc.calculate(transactions.iloc[2])  # Pass the third row as a pd.Series
    assert result == pytest.approx(400.0)


def test_fifo_partial_consumption_across_buys():
    """
    A sell transaction that spans multiple buy transactions.
    """
    transactions = pd.DataFrame([
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 5, SHARE_PRICE: 100},
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 5, SHARE_PRICE: 120},
        {TRANSACTION_TYPE: TYPE_SELL, SHARES: 8, SHARE_PRICE: 150},
    ])
    calc = FIFOCalculator(transactions)

    result = calc.calculate(transactions.iloc[2])
    assert result == pytest.approx(340.0)


def test_fifo_sell_exactly_all_shares():
    """
    Boundary case where the sell transaction matches exactly the total shares
    that were bought.
    """
    transactions = pd.DataFrame([
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 4, SHARE_PRICE: 100},
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 6, SHARE_PRICE: 110},
        {TRANSACTION_TYPE: TYPE_SELL, SHARES: 10, SHARE_PRICE: 130},
    ])
    calc = FIFOCalculator(transactions)

    result = calc.calculate(transactions.iloc[2])
    assert result == pytest.approx(240.0)

    # Verify that available shares are reduced to 0
    assert transactions.at[0, AVAILABLE_SHARES] == 0
    assert transactions.at[1, AVAILABLE_SHARES] == 0


def test_fifo_sell_more_than_available():
    """
    A scenario where the sell transaction tries to sell more shares
    than the total available.
    """
    transactions = pd.DataFrame([
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 3, SHARE_PRICE: 100},
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 2, SHARE_PRICE: 120},
        {TRANSACTION_TYPE: TYPE_SELL, SHARES: 10, SHARE_PRICE: 150},
    ])
    calc = FIFOCalculator(transactions)

    with pytest.raises(ValueError) as exc_info:
        calc.calculate(transactions.iloc[2])

    expected_msg = "Cannot sell 10 shares; only 5.0 are available in previous buys."
    assert expected_msg in str(exc_info.value)

    # Verify that available shares remain unchanged
    assert transactions.at[0, AVAILABLE_SHARES] == 3
    assert transactions.at[1, AVAILABLE_SHARES] == 2


def test_fifo_no_sell_transaction():
    """
    If we call FIFOCalculator with a transaction that is NOT a sell,
    we should get None.
    """
    transactions = pd.DataFrame([
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 10, SHARE_PRICE: 100},
    ])
    calc = FIFOCalculator(transactions)

    result = calc.calculate(transactions.iloc[0])
    assert result is None


def test_fifo_multiple_sells_in_sequence():
    """
    Multiple sells in a row, each one picks up from the modified state of the
    'Available Shares' in the buy transactions.
    """
    transactions = pd.DataFrame([
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 10, SHARE_PRICE: 50},
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 5, SHARE_PRICE: 60},
        {TRANSACTION_TYPE: TYPE_SELL, SHARES: 8, SHARE_PRICE: 80},
        {TRANSACTION_TYPE: TYPE_SELL, SHARES: 5, SHARE_PRICE: 100},
    ])
    calc = FIFOCalculator(transactions)

    # First sell
    result_sell1 = calc.calculate(transactions.iloc[2])
    assert result_sell1 == pytest.approx(240.0)
    assert transactions.at[0, AVAILABLE_SHARES] == 2
    assert transactions.at[1, AVAILABLE_SHARES] == 5

    # Second sell
    result_sell2 = calc.calculate(transactions.iloc[3])
    assert result_sell2 == pytest.approx(220.0)
    assert transactions.at[0, AVAILABLE_SHARES] == 0
    assert transactions.at[1, AVAILABLE_SHARES] == 2


def test_fifo_sell_ignores_future_buys_and_raises_exception():
    """
    Checks that a 'sell' transaction cannot use shares from a future buy,
    and raises an exception if insufficient shares are available among
    the earlier buys.
    """
    transactions = pd.DataFrame([
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 5, SHARE_PRICE: 100},
        {TRANSACTION_TYPE: TYPE_SELL, SHARES: 6, SHARE_PRICE: 130},
        {TRANSACTION_TYPE: TYPE_BUY, SHARES: 10, SHARE_PRICE: 80},
    ])
    calc = FIFOCalculator(transactions)

    with pytest.raises(ValueError) as exc_info:
        calc.calculate(transactions.iloc[1])

    assert "Cannot sell 6 shares; only 5.0 are available in previous buys." in str(exc_info.value)
