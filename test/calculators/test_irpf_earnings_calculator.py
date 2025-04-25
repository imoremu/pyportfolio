import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pyportfolio.calculators.irpf_earnings_calculator import IrpfEarningsCalculator
# Import constants used in test data setup
from pyportfolio.columns import DATE, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE

# --- Fixtures for common data ---
@pytest.fixture
def sample_transactions_base():
    """ Provides an empty DataFrame with the expected structure. """
    return pd.DataFrame({
        DATE: pd.to_datetime([]),
        TRANSACTION_TYPE: pd.Series([], dtype=str),
        TICKER: pd.Series([], dtype=str),
        SHARES: pd.Series([], dtype=float),
        SHARE_PRICE: pd.Series([], dtype=float)
        # No longer needs 'FIFO Gain/Loss' column
    })

# Helper to create transactions and calculator
def create_calc_from_data(data, base_fixture):
    """ Creates DataFrame from test data, ensures sorting, and initializes calculator. """
    df = pd.DataFrame(data)
    df[DATE] = pd.to_datetime(df[DATE])
    transactions = pd.concat([base_fixture, df], ignore_index=True)
    # Ensure sorting by date and type (buys before sells on same day)
    transactions = transactions.sort_values(
        by=[DATE, TRANSACTION_TYPE],
        ascending=[True, True],
        key=lambda col: col.map({'buy': 0, 'sell': 1}) if col.name == TRANSACTION_TYPE else col
    ).reset_index(drop=True)
    calculator = IrpfEarningsCalculator(transactions_df=transactions)
    return calculator, transactions

# --- Test Cases ---

def test_irpf_gain_is_returned_directly(sample_transactions_base):
    """ Test gain: Sell returns (gain, 0.0), Buy returns (0.0, 0.0). """
    data = {
        DATE: ['2023-01-10', '2023-03-15'],
        TRANSACTION_TYPE: ['buy', 'sell'],
        TICKER: ['XYZ', 'XYZ'],
        SHARES: [100, 100],
        SHARE_PRICE: [10, 12]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row = transactions[transactions[TRANSACTION_TYPE] == 'buy'].iloc[0]
    sell_row = transactions[transactions[TRANSACTION_TYPE] == 'sell'].iloc[0]

    result_buy = calculator.calculate(buy_row)
    result_sell = calculator.calculate(sell_row)

    assert result_buy == (0.0, 0.0), "Buy row should have (0.0, 0.0)"
    assert result_sell == (200.0, 0.0), "Sell row should have (Gain, 0.0)"

def test_irpf_loss_no_repurchase_within_window(sample_transactions_base):
    """ Test loss (no deferral): Sell returns (loss, 0.0), Buys return (0.0, 0.0). """
    data = {
        DATE: ['2023-01-10', '2023-06-15', '2023-10-01'], # Buy is > 2 months after sell
        TRANSACTION_TYPE: ['buy', 'sell', 'buy'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 100, 50],
        SHARE_PRICE: [10, 8, 7]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row_1 = transactions[transactions[TRANSACTION_TYPE] == 'buy'].iloc[0]
    sell_row = transactions[transactions[TRANSACTION_TYPE] == 'sell'].iloc[0]
    buy_row_2 = transactions[transactions[TRANSACTION_TYPE] == 'buy'].iloc[1]

    result_buy_1 = calculator.calculate(buy_row_1)
    result_sell = calculator.calculate(sell_row)
    result_buy_2 = calculator.calculate(buy_row_2)

    assert result_buy_1 == (0.0, 0.0), "Initial Buy row"
    assert result_sell == (-200.0, 0.0), "Sell row should have (Loss, 0.0) as not deferred"
    assert result_buy_2 == (0.0, 0.0), "Later Buy row"

def test_irpf_loss_deferred_due_to_repurchase_within_2_months_after(sample_transactions_base):
    """ Test loss partial deferral (buy after): Sell=(Allowed Loss, 0.0), Blocking Buy=(0.0, +DeferredLoss). """
    data = {
        DATE: ['2023-01-10', '2023-06-15', '2023-07-20'], # Repurchase within 2 months after
        TRANSACTION_TYPE: ['buy', 'sell', 'buy'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 100, 50], # Sell 100, Repurchase 50
        SHARE_PRICE: [10, 8, 9]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row_initial = transactions[transactions[TRANSACTION_TYPE] == 'buy'].iloc[0]
    sell_row = transactions[transactions[TRANSACTION_TYPE] == 'sell'].iloc[0]
    buy_row_blocking = transactions[transactions[TRANSACTION_TYPE] == 'buy'].iloc[1]

    result_buy_initial = calculator.calculate(buy_row_initial)
    result_sell = calculator.calculate(sell_row)
    result_buy_blocking = calculator.calculate(buy_row_blocking)

    # Sell 100@8. Cost 100@10 = 1000. Loss = 800-1000 = -200 (-2/share).
    # Repurchase 50 shares blocks 50 shares.
    # Deferred Loss = 50 * abs(-2) = 100.
    # Allowed Loss = -200 + 100 = -100.
    expected_allowable_loss = -100.0
    expected_buy_adjustment = 100.0

    assert result_buy_initial == (0.0, 0.0), "Initial Buy row"
    assert result_sell == (expected_allowable_loss, 0.0), "Sell row loss is partially deferred"
    assert result_buy_blocking == (0.0, expected_buy_adjustment), "Blocking Buy row gets adjustment"

def test_irpf_loss_deferred_due_to_repurchase_within_2_months_before(sample_transactions_base):
    """ Test loss partial deferral (buy before): Sell=(Allowed Loss, 0.0), Blocking Buy=(0.0, +DeferredLoss). """
    data = {
        DATE: ['2023-01-10', '2023-05-20', '2023-06-15'], # Buy before within 2 months
        TRANSACTION_TYPE: ['buy', 'buy', 'sell'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 50, 100], # Sell 100, Repurchase 50 before
        SHARE_PRICE: [10, 9, 8]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row_initial = transactions[transactions[TRANSACTION_TYPE] == 'buy'].iloc[0]
    buy_row_blocking = transactions[transactions[TRANSACTION_TYPE] == 'buy'].iloc[1]
    sell_row = transactions[transactions[TRANSACTION_TYPE] == 'sell'].iloc[0]

    result_buy_initial = calculator.calculate(buy_row_initial)
    result_buy_blocking = calculator.calculate(buy_row_blocking)
    result_sell = calculator.calculate(sell_row)

    # Sell 100@8. Cost 100@10 = 1000. Loss = 800-1000 = -200 (-2/share).
    # Repurchase 50 shares blocks 50 shares.
    # Deferred Loss = 50 * abs(-2) = 100.
    # Allowed Loss = -200 + 100 = -100.
    expected_allowable_loss = -100.0
    expected_buy_adjustment = 100.0

    assert result_buy_initial == (0.0, 0.0), "Initial Buy row"
    assert result_buy_blocking == (0.0, expected_buy_adjustment), "Blocking Buy row gets adjustment"
    assert result_sell == (expected_allowable_loss, 0.0), "Sell row loss is partially deferred"

def test_irpf_loss_not_deferred_if_repurchase_is_different_ticker(sample_transactions_base):
    """ Test loss not deferred (diff ticker): Sell returns (loss, 0.0), Buys return (0.0, 0.0). """
    data = {
        DATE: ['2023-01-10', '2023-06-15', '2023-07-20'],
        TRANSACTION_TYPE: ['buy', 'sell', 'buy'],
        TICKER: ['XYZ', 'XYZ', 'ABC'], # Repurchase is 'ABC', sale is 'XYZ'
        SHARES: [100, 100, 50],
        SHARE_PRICE: [10, 8, 9]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row_xyz = transactions[transactions[TICKER] == 'XYZ'].iloc[0]
    sell_row_xyz = transactions[transactions[TICKER] == 'XYZ'].iloc[1]
    buy_row_abc = transactions[transactions[TICKER] == 'ABC'].iloc[0]

    result_buy_xyz = calculator.calculate(buy_row_xyz)
    result_sell_xyz = calculator.calculate(sell_row_xyz)
    result_buy_abc = calculator.calculate(buy_row_abc)

    # Sell 100@8. Cost 100@10 = 1000. Loss = -200.
    # Repurchase is different ticker, so no deferral.
    assert result_buy_xyz == (0.0, 0.0), "Initial XYZ Buy row"
    assert result_sell_xyz == (-200.0, 0.0), "XYZ Sell row loss NOT deferred"
    assert result_buy_abc == (0.0, 0.0), "ABC Buy row"

def test_irpf_loss_deferred_and_gain_realized(sample_transactions_base):
    """ Test sequence: Deferred loss, then a Gain using adjusted cost basis. """
    data = {
        DATE: ['2023-01-10', '2023-05-20', '2023-06-15', '2023-07-20', '2023-09-10'],
        TRANSACTION_TYPE: ['buy', 'buy', 'sell', 'buy', 'sell'],
        TICKER: ['XYZ', 'XYZ', 'XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 50, 80, 60, 70], # Sell 80, blocked by Buy 50 (before) and Buy 60 (after)
        SHARE_PRICE: [10, 9, 8, 10, 12]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_1_row = transactions.iloc[0]
    buy_2_blocking_row = transactions.iloc[1]
    sell_1_deferred_row = transactions.iloc[2]
    buy_3_blocking_row = transactions.iloc[3]
    sell_2_gain_row = transactions.iloc[4]

    result_buy_1 = calculator.calculate(buy_1_row)
    result_buy_2_blocking = calculator.calculate(buy_2_blocking_row)
    result_sell_1_deferred = calculator.calculate(sell_1_deferred_row)
    result_buy_3_blocking = calculator.calculate(buy_3_blocking_row)
    result_sell_2_gain = calculator.calculate(sell_2_gain_row)

    # Sell 1: 80@8. Cost 80@10=800. Loss=-160 (-2/share).
    # Blocked by Buy 2 (50 sh) and Buy 3 (30 sh). Allowable=0.
    # Buy 2 Adj = 50 * |-2| = 100. Buy 2 Adj Cost/Sh = (50*9+100)/50 = 11.
    # Buy 3 Adj = 30 * |-2| = 60. Buy 3 Adj Cost/Sh = (60*10+60)/60 = 11.
    expected_adj_buy2 = 100.0
    expected_adj_buy3 = 60.0

    # Sell 2: 70@12 = 840.
    # Cost: Consume 20 sh from Buy 1 (rem 100-80=20) -> 20 * 10 = 200.
    #       Consume 50 sh from Buy 2 -> 50 * 11 (adj cost) = 550.
    # Total Cost = 200 + 550 = 750.
    # Gain = 840 - 750 = 90.
    expected_gain_sell2 = 90.0

    assert result_buy_1 == (0.0, 0.0), "Initial Buy"
    assert result_buy_2_blocking == (0.0, expected_adj_buy2), "Blocking Buy 2 gets adjustment"
    assert result_sell_1_deferred == (0.0, 0.0), "Deferred Sell 1"
    assert result_buy_3_blocking == (0.0, expected_adj_buy3), "Blocking Buy 3 gets adjustment"
    assert result_sell_2_gain == (expected_gain_sell2, 0.0), "Gain Sell 2 uses adjusted cost"

def test_irpf_loss_deferred_and_loss_realized(sample_transactions_base):
    """ Test sequence: Deferred loss, then a Loss using adjusted cost basis. """
    data = {
        DATE: ['2023-01-10', '2023-05-20', '2023-06-15', '2023-07-20', '2023-10-10'],
        TRANSACTION_TYPE: ['buy', 'buy', 'sell', 'buy', 'sell'],
        TICKER: ['XYZ', 'XYZ', 'XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 50, 80, 60, 80], # Sell 80, blocked by Buy 50 (before) and Buy 60 (after)
        SHARE_PRICE: [10, 9, 8, 10, 7]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_1_row = transactions.iloc[0]
    buy_2_blocking_row = transactions.iloc[1]
    sell_1_deferred_row = transactions.iloc[2]
    buy_3_blocking_row = transactions.iloc[3]
    sell_2_loss_row = transactions.iloc[4]

    result_buy_1 = calculator.calculate(buy_1_row)
    result_buy_2_blocking = calculator.calculate(buy_2_blocking_row)
    result_sell_1_deferred = calculator.calculate(sell_1_deferred_row)
    result_buy_3_blocking = calculator.calculate(buy_3_blocking_row)
    result_sell_2_loss = calculator.calculate(sell_2_loss_row)

    # Sell 1: 80@8. Cost 80@10=800. Loss=-160 (-2/share).
    # Blocked by Buy 2 (50 sh) and Buy 3 (30 sh). Allowable=0.
    # Buy 2 Adj = 50 * |-2| = 100. Buy 2 Adj Cost/Sh = (50*9+100)/50 = 11.
    # Buy 3 Adj = 30 * |-2| = 60. Buy 3 Adj Cost/Sh = (60*10+60)/60 = 11.
    expected_adj_buy2 = 100.0
    expected_adj_buy3 = 60.0

    # Sell 2: 80@7 = 560.
    # Cost: Consume 20 sh from Buy 1 (rem 100-80=20) -> 20 * 10 = 200.
    #       Consume 50 sh from Buy 2 -> 50 * 11 (adj cost) = 550.
    #       Consume 10 sh from Buy 3 (rem 60) -> 10 * 11 (adj cost) = 110.
    # Total Cost = 200 + 550 + 110 = 860.
    # Loss = 560 - 860 = -300.
    # No repurchases within +/- 2 months of Sell 2 (2023-10-10), so loss is allowable.
    expected_loss_sell2 = -300.0

    assert result_buy_1 == (0.0, 0.0), "Initial Buy"
    assert result_buy_2_blocking == (0.0, expected_adj_buy2), "Blocking Buy 2 gets adjustment"
    assert result_sell_1_deferred == (0.0, 0.0), "Deferred Sell 1"
    assert result_buy_3_blocking == (0.0, expected_adj_buy3), "Blocking Buy 3 gets adjustment"
    assert result_sell_2_loss == (expected_loss_sell2, 0.0), "Loss Sell 2 uses adjusted cost"

def test_irpf_single_buy_blocks_two_separate_sells_capacity_logic(sample_transactions_base):
    """
    Test scenario where one 'buy' (70 shares) blocks two 'sells' (50 shares each).
    Assumes the buy's blocking capacity (70) is consumed chronologically,
    and the sell's allowable loss reflects only the portion not blocked by capacity.
    """
    data = {
        DATE: ['2023-01-10', '2023-06-10', '2023-07-10', '2023-08-10'],
        TRANSACTION_TYPE: ['buy', 'sell', 'buy', 'sell'],
        TICKER: ['XYZ', 'XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 50, 70, 50],
        SHARE_PRICE: [10, 8, 9, 7]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_1_row = transactions.iloc[0]
    sell_1_row = transactions.iloc[1]
    buy_blocking_row = transactions.iloc[2]
    sell_2_row = transactions.iloc[3]

    result_buy_1 = calculator.calculate(buy_1_row)
    result_sell_1 = calculator.calculate(sell_1_row)
    result_buy_blocking = calculator.calculate(buy_blocking_row)
    result_sell_2 = calculator.calculate(sell_2_row)

    # Sell 1: Loss -100 (-2/sh). Buy Blocking (70sh) blocks 50sh. Deferred=100. Allowable=0. Capacity Used=50. Rem=20.
    allowable_sell1 = 0.0
    deferred_from_sell1 = 100.0

    # Sell 2: Loss -150 (-3/sh). Buy Blocking (20sh rem capacity) blocks 20sh. Deferred=60. Allowable=-150+60=-90.
    deferred_from_sell2_by_this_buy = 60.0
    expected_allowable_sell2 = -90.0

    # Buy Blocking Adj = Deferred from Sell 1 + Deferred from Sell 2 = 100 + 60 = 160.
    expected_total_adjustment_on_buy = 160.0

    assert result_buy_1 == (0.0, 0.0), "Initial Buy"
    assert result_sell_1 == (allowable_sell1, 0.0), "Sell 1 allowable loss"
    assert result_buy_blocking == (0.0, expected_total_adjustment_on_buy), "Blocking Buy adjustment"
    assert result_sell_2 == (expected_allowable_sell2, 0.0), "Sell 2 allowable loss"

def test_irpf_returns_correct_tuples_for_non_sell_transactions(sample_transactions_base):
    """ Test Buy returns (0.0, 0.0), others return (None, None). """
    data = {
        DATE: ['2023-01-10', '2023-03-15'],
        TRANSACTION_TYPE: ['buy', 'dividend'], # No 'sell'
        TICKER: ['XYZ', 'XYZ'],
        SHARES: [100, np.nan],
        SHARE_PRICE: [10, np.nan]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row = transactions.iloc[0]
    dividend_row = transactions.iloc[1]

    result_buy = calculator.calculate(buy_row)
    result_dividend = calculator.calculate(dividend_row)

    assert result_buy == (0.0, 0.0), "Buy row"
    assert result_dividend == (None, None), "Non buy/sell row"

# Test removed as calculator no longer depends on input FIFO column
# def test_irpf_handles_nan_fifo_gain_loss_for_sell(sample_transactions_base): ...

def test_irpf_edge_case_exactly_two_months_before_exclusive(sample_transactions_base):
    """ Test loss NOT deferred (exact 2mo before): Sell=(loss, 0.0), Buy=(0.0, 0.0). """
    data = {
        DATE: ['2023-01-15', '2023-04-15', '2023-06-15'], # Buy exactly 2 months before sell
        TRANSACTION_TYPE: ['buy', 'buy', 'sell'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [50, 50, 100],
        SHARE_PRICE: [9, 9, 8]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row = transactions.iloc[1] # The potential blocker
    sell_row = transactions.iloc[2]

    result_buy = calculator.calculate(buy_row)
    result_sell = calculator.calculate(sell_row)

    # Sell 100@8. Cost: 50@9 + 50@9 = 900. Loss = 800-900 = -100.
    # Buy is exactly 2 months before, so window is > date, loss not deferred.
    assert result_buy == (0.0, 0.0)
    assert result_sell == (-100.0, 0.0)

def test_irpf_edge_case_exactly_two_months_after_exclusive(sample_transactions_base):
    """ Test loss NOT deferred (exact 2mo after): Sell=(loss, 0.0), Buy=(0.0, 0.0). """
    data = {
        DATE: ['2023-01-15', '2023-06-15', '2023-08-15'], # Buy exactly 2 months after sell
        TRANSACTION_TYPE: ['buy', 'sell', 'buy'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 100, 50],
        SHARE_PRICE: [10, 8, 9]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    sell_row = transactions.iloc[1]
    buy_row = transactions.iloc[2] # The potential blocker

    result_sell = calculator.calculate(sell_row)
    result_buy = calculator.calculate(buy_row)

    # Sell 100@8. Cost 100@10 = 1000. Loss = -200.
    # Buy is exactly 2 months after, so window is < date, loss not deferred.
    assert result_sell == (-200.0, 0.0)
    assert result_buy == (0.0, 0.0)

# --- Initialization Tests ---

def test_init_raises_error_if_df_missing():
    """ Test ValueError if initialized with non-DataFrame. """
    with pytest.raises(ValueError, match="transactions_df must be a pandas DataFrame"):
        IrpfEarningsCalculator(transactions_df=None)
    with pytest.raises(ValueError, match="transactions_df must be a pandas DataFrame"):
        IrpfEarningsCalculator(transactions_df=[1, 2, 3])

def test_init_raises_error_if_missing_required_columns(sample_transactions_base):
    """ Test ValueError if DataFrame is missing essential columns. """
    valid_data = {
        DATE: [datetime(2023,1,1)], TRANSACTION_TYPE: ['buy'], TICKER: ['T'],
        SHARES: [1], SHARE_PRICE: [1]
    }
    transactions_ok = pd.DataFrame(valid_data)
    transactions_ok[DATE] = pd.to_datetime(transactions_ok[DATE])

    # Check missing columns needed by the calculator
    required_cols = [DATE, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE]
    for col in required_cols:
        if col not in transactions_ok.columns: continue
        transactions_bad = transactions_ok.drop(columns=[col])
        with pytest.raises(ValueError, match=f"DataFrame must contain columns:.*{col}"):
             IrpfEarningsCalculator(transactions_df=transactions_bad)

def test_init_raises_error_if_date_column_not_convertible(sample_transactions_base):
    """ Test ValueError if the date column cannot be converted to datetime. """
    data = {
        DATE: ['2023-01-10', 'invalid-date-string'],
        TRANSACTION_TYPE: ['buy', 'sell'], TICKER: ['XYZ', 'XYZ'],
        SHARES: [100, 100], SHARE_PRICE: [10, 12]
    }
    transactions = pd.concat([sample_transactions_base, pd.DataFrame(data)], ignore_index=True)

    with pytest.raises(ValueError, match="Could not convert date column 'Date' to datetime"):
        IrpfEarningsCalculator(transactions_df=transactions)

def test_init_converts_date_column_if_possible(sample_transactions_base):
    """ Test that the date column is converted to datetime during init if it's not already. """
    data = {
        DATE: ['2023-01-10', '2023-03-15'], # Dates as strings
        TRANSACTION_TYPE: ['buy', 'sell'], TICKER: ['XYZ', 'XYZ'],
        SHARES: [100, 100], SHARE_PRICE: [10, 12]
    }
    transactions = pd.concat([sample_transactions_base, pd.DataFrame(data)], ignore_index=True)

    calculator = IrpfEarningsCalculator(transactions_df=transactions)
    assert pd.api.types.is_datetime64_any_dtype(calculator.internal_transactions[DATE])

