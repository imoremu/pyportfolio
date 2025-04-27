# test/calculators/test_irpf_earnings_calculator.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Import the calculator and result column names
from pyportfolio.calculators.irpf_earnings_calculator import (
    IrpfEarningsCalculator,
    RESULT_TAXABLE_GAIN_LOSS,
    RESULT_DEFERRED_ADJUSTMENT
)

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
    })

# Helper to create transactions, run calculator directly, and merge results
def run_irpf_calc_direct(data, base_fixture):
    """
    Creates DataFrame from test data, ensures sorting,
    instantiates IrpfEarningsCalculator, calls calculate_table directly,
    and merges the results back with the input for easier testing.
    """
    df_input = pd.DataFrame(data)
    df_input[DATE] = pd.to_datetime(df_input[DATE])
    transactions_input = pd.concat([base_fixture, df_input], ignore_index=True)
    # Ensure sorting by date and type (buys before sells on same day)
    transactions_input = transactions_input.sort_values(
        by=[DATE, TRANSACTION_TYPE],
        ascending=[True, True],
        key=lambda col: col.map({'buy': 0, 'sell': 1}) if col.name == TRANSACTION_TYPE else col
    ).reset_index(drop=True) # Keep original index for comparison if needed later

    # Instantiate calculator and call calculate_table
    calculator = IrpfEarningsCalculator()
    # Pass a copy to calculate_table to mimic TransactionManager behavior
    results_df = calculator.calculate_table(transactions_input.copy())

    # Merge results back to the sorted input based on index for easier assertions
    # The results_df should have the same index as transactions_input
    merged_df = transactions_input.join(results_df)
    return merged_df

# --- Test Cases ---

def test_irpf_gain_is_calculated(sample_transactions_base):
    """ Test gain: Sell row has gain, Buy row has 0.0. """
    data = {
        DATE: ['2023-01-10', '2023-03-15'],
        TRANSACTION_TYPE: ['buy', 'sell'],
        TICKER: ['XYZ', 'XYZ'],
        SHARES: [100, 100],
        SHARE_PRICE: [10, 12]
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_result = processed_df[processed_df[TRANSACTION_TYPE] == 'buy'].iloc[0]
    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == 'sell'].iloc[0]

    assert buy_row_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_row_result[RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(200.0)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_loss_no_repurchase_within_window(sample_transactions_base):
    """ Test loss (no deferral): Sell row has loss, Buy rows have 0.0. """
    data = {
        DATE: ['2023-01-10', '2023-06-15', '2023-10-01'], # Buy is > 2 months after sell
        TRANSACTION_TYPE: ['buy', 'sell', 'buy'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 100, 50],
        SHARE_PRICE: [10, 8, 7]
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_1_result = processed_df[processed_df[TRANSACTION_TYPE] == 'buy'].iloc[0]
    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == 'sell'].iloc[0]
    buy_row_2_result = processed_df[processed_df[TRANSACTION_TYPE] == 'buy'].iloc[1]

    assert buy_row_1_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_row_result[RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(-200.0)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_row_2_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_2_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_loss_deferred_due_to_repurchase_within_2_months_after(sample_transactions_base):
    """ Test loss partial deferral (buy after): Sell has Allowed Loss, Blocking Buy has Adjustment. """
    data = {
        DATE: ['2023-01-10', '2023-06-15', '2023-07-20'], # Repurchase within 2 months after
        TRANSACTION_TYPE: ['buy', 'sell', 'buy'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 100, 50], # Sell 100, Repurchase 50
        SHARE_PRICE: [10, 8, 9]
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_initial_result = processed_df[processed_df[TRANSACTION_TYPE] == 'buy'].iloc[0]
    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == 'sell'].iloc[0]
    buy_row_blocking_result = processed_df[processed_df[TRANSACTION_TYPE] == 'buy'].iloc[1]

    # Sell 100@8. Cost 100@10 = 1000. Loss = 800-1000 = -200 (-2/share).
    # Repurchase 50 shares blocks 50 shares.
    # Deferred Loss = 50 * abs(-2) = 100.
    # Allowed Loss = -200 + 100 = -100.
    expected_allowable_loss = -100.0
    expected_buy_adjustment = 100.0

    assert buy_row_initial_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_initial_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_row_result[RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(expected_allowable_loss)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_row_blocking_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_buy_adjustment)

def test_irpf_loss_deferred_due_to_repurchase_within_2_months_before(sample_transactions_base):
    """ Test loss partial deferral (buy before): Sell has Allowed Loss, Blocking Buy has Adjustment. """
    data = {
        DATE: ['2023-01-10', '2023-05-20', '2023-06-15'], # Buy before within 2 months
        TRANSACTION_TYPE: ['buy', 'buy', 'sell'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 50, 100], # Sell 100, Repurchase 50 before
        SHARE_PRICE: [10, 9, 8]
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_initial_result = processed_df[processed_df[TRANSACTION_TYPE] == 'buy'].iloc[0]
    buy_row_blocking_result = processed_df[processed_df[TRANSACTION_TYPE] == 'buy'].iloc[1]
    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == 'sell'].iloc[0]

    # Sell 100@8. Cost 100@10 = 1000. Loss = 800-1000 = -200 (-2/share).
    # Repurchase 50 shares blocks 50 shares.
    # Deferred Loss = 50 * abs(-2) = 100.
    # Allowed Loss = -200 + 100 = -100.
    expected_allowable_loss = -100.0
    expected_buy_adjustment = 100.0

    assert buy_row_initial_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_initial_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_row_blocking_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_buy_adjustment)
    assert sell_row_result[RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(expected_allowable_loss)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_loss_not_deferred_if_repurchase_is_different_ticker(sample_transactions_base):
    """ Test loss not deferred (diff ticker): Sell has loss, Buys have 0.0. """
    data = {
        DATE: ['2023-01-10', '2023-06-15', '2023-07-20'],
        TRANSACTION_TYPE: ['buy', 'sell', 'buy'],
        TICKER: ['XYZ', 'XYZ', 'ABC'], # Repurchase is 'ABC', sale is 'XYZ'
        SHARES: [100, 100, 50],
        SHARE_PRICE: [10, 8, 9]
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_xyz_result = processed_df[processed_df[TICKER] == 'XYZ'].iloc[0]
    sell_row_xyz_result = processed_df[processed_df[TICKER] == 'XYZ'].iloc[1]
    buy_row_abc_result = processed_df[processed_df[TICKER] == 'ABC'].iloc[0]

    # Sell 100@8. Cost 100@10 = 1000. Loss = -200.
    # Repurchase is different ticker, so no deferral.
    assert buy_row_xyz_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_xyz_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_row_xyz_result[RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(-200.0)
    assert sell_row_xyz_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_row_abc_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_abc_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_loss_deferred_and_gain_realized(sample_transactions_base):
    """ Test sequence: Deferred loss, then a Gain using adjusted cost basis. """
    data = {
        DATE: ['2023-01-10', '2023-05-20', '2023-06-15', '2023-07-20', '2023-09-10'],
        TRANSACTION_TYPE: ['buy', 'buy', 'sell', 'buy', 'sell'],
        TICKER: ['XYZ', 'XYZ', 'XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 50, 80, 60, 70], # Sell 80, blocked by Buy 50 (before) and Buy 60 (after)
        SHARE_PRICE: [10, 9, 8, 10, 12]
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_1_result = processed_df.iloc[0]
    buy_2_blocking_result = processed_df.iloc[1]
    sell_1_deferred_result = processed_df.iloc[2]
    buy_3_blocking_result = processed_df.iloc[3]
    sell_2_gain_result = processed_df.iloc[4]

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

    assert buy_1_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_2_blocking_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_2_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_adj_buy2)
    assert sell_1_deferred_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0 # Fully deferred
    assert sell_1_deferred_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_3_blocking_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_3_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_adj_buy3)
    assert sell_2_gain_result[RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(expected_gain_sell2)
    assert sell_2_gain_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_loss_deferred_and_loss_realized(sample_transactions_base):
    """ Test sequence: Deferred loss, then a Loss using adjusted cost basis. """
    data = {
        DATE: ['2023-01-10', '2023-05-20', '2023-06-15', '2023-07-20', '2023-10-10'],
        TRANSACTION_TYPE: ['buy', 'buy', 'sell', 'buy', 'sell'],
        TICKER: ['XYZ', 'XYZ', 'XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 50, 80, 60, 80], # Sell 80, blocked by Buy 50 (before) and Buy 60 (after)
        SHARE_PRICE: [10, 9, 8, 10, 7]
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_1_result = processed_df.iloc[0]
    buy_2_blocking_result = processed_df.iloc[1]
    sell_1_deferred_result = processed_df.iloc[2]
    buy_3_blocking_result = processed_df.iloc[3]
    sell_2_loss_result = processed_df.iloc[4]

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

    assert buy_1_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_2_blocking_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_2_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_adj_buy2)
    assert sell_1_deferred_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0 # Fully deferred
    assert sell_1_deferred_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_3_blocking_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_3_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_adj_buy3)
    assert sell_2_loss_result[RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(expected_loss_sell2)
    assert sell_2_loss_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_single_buy_blocks_two_separate_sells_capacity_logic(sample_transactions_base):
    """
    Test scenario where one 'buy' (70 shares) blocks two 'sells' (50 shares each).
    Assumes the buy's blocking capacity (70) is consumed chronologically.
    """
    data = {
        DATE: ['2023-01-10', '2023-06-10', '2023-07-10', '2023-08-10'],
        TRANSACTION_TYPE: ['buy', 'sell', 'buy', 'sell'],
        TICKER: ['XYZ', 'XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 50, 70, 50],
        SHARE_PRICE: [10, 8, 9, 7]
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_1_result = processed_df.iloc[0]
    sell_1_result = processed_df.iloc[1]
    buy_blocking_result = processed_df.iloc[2]
    sell_2_result = processed_df.iloc[3]

    # Sell 1: 50@8. Cost 50@10=500. Loss -100 (-2/sh).
    # Buy Blocking (70sh) blocks 50sh. Deferred=100. Allowable=0. Capacity Used=50. Rem=20.
    allowable_sell1 = 0.0
    deferred_from_sell1 = 100.0

    # Sell 2: 50@7. Cost 50@10=500. Loss -150 (-3/sh).
    # Buy Blocking (20sh rem capacity) blocks 20sh. Deferred=20*|-3|=60. Allowable=-150+60=-90.
    deferred_from_sell2_by_this_buy = 60.0
    expected_allowable_sell2 = -90.0

    # Buy Blocking Adj = Deferred from Sell 1 + Deferred from Sell 2 = 100 + 60 = 160.
    expected_total_adjustment_on_buy = 160.0

    assert buy_1_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_1_result[RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(allowable_sell1)
    assert sell_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_blocking_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_total_adjustment_on_buy)
    assert sell_2_result[RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(expected_allowable_sell2)
    assert sell_2_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_handles_non_buy_sell_transactions(sample_transactions_base):
    """ Test Buy has 0.0, others have None. """
    data = {
        DATE: ['2023-01-10', '2023-03-15'],
        TRANSACTION_TYPE: ['buy', 'dividend'], # No 'sell'
        TICKER: ['XYZ', 'XYZ'],
        SHARES: [100, np.nan],
        SHARE_PRICE: [10, np.nan]
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_result = processed_df.iloc[0]
    dividend_row_result = processed_df.iloc[1]

    assert buy_row_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert pd.isna(dividend_row_result[RESULT_TAXABLE_GAIN_LOSS])
    assert pd.isna(dividend_row_result[RESULT_DEFERRED_ADJUSTMENT])

def test_irpf_edge_case_exactly_two_months_before_exclusive(sample_transactions_base):
    """ Test loss NOT deferred (exact 2mo before): Sell has loss, Buy has 0.0. """
    data = {
        DATE: ['2023-01-15', '2023-04-15', '2023-06-15'], # Buy exactly 2 months before sell
        TRANSACTION_TYPE: ['buy', 'buy', 'sell'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [50, 50, 100],
        SHARE_PRICE: [9, 9, 8]
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_result = processed_df.iloc[1] # The potential blocker
    sell_row_result = processed_df.iloc[2]

    # Sell 100@8. Cost: 50@9 + 50@9 = 900. Loss = 800-900 = -100.
    # Buy is exactly 2 months before, so window is > date, loss not deferred.
    assert buy_row_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_row_result[RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(-100.0)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_edge_case_exactly_two_months_after_exclusive(sample_transactions_base):
    """ Test loss NOT deferred (exact 2mo after): Sell has loss, Buy has 0.0. """
    data = {
        DATE: ['2023-01-15', '2023-06-15', '2023-08-15'], # Buy exactly 2 months after sell
        TRANSACTION_TYPE: ['buy', 'sell', 'buy'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100, 100, 50],
        SHARE_PRICE: [10, 8, 9]
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    sell_row_result = processed_df.iloc[1]
    buy_row_result = processed_df.iloc[2] # The potential blocker

    # Sell 100@8. Cost 100@10 = 1000. Loss = -200.
    # Buy is exactly 2 months after, so window is < date, loss not deferred.
    assert sell_row_result[RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(-200.0)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_row_result[RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert buy_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

# --- Validation Tests (calling calculate_table directly) ---

def test_calculate_table_raises_error_if_df_missing():
    """ Test ValueError if calculate_table called with non-DataFrame. """
    calculator = IrpfEarningsCalculator()
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        calculator.calculate_table(None)
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        calculator.calculate_table([1, 2, 3])

def test_calculate_table_raises_error_if_missing_required_columns(sample_transactions_base):
    """ Test ValueError from calculate_table if DataFrame is missing essential columns. """
    valid_data = {
        DATE: [datetime(2023,1,1)], TRANSACTION_TYPE: ['buy'], TICKER: ['T'],
        SHARES: [1], SHARE_PRICE: [1]
    }
    transactions_ok = pd.DataFrame(valid_data)
    transactions_ok[DATE] = pd.to_datetime(transactions_ok[DATE])

    calculator = IrpfEarningsCalculator()

    # Check missing columns needed by the calculator
    required_cols = [DATE, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE]
    for col in required_cols:
        if col not in transactions_ok.columns: continue
        transactions_bad = transactions_ok.drop(columns=[col])
        with pytest.raises(ValueError, match=f"DataFrame must contain columns:.*{col}"):
             calculator.calculate_table(transactions_bad)

def test_calculate_table_raises_error_if_date_column_not_convertible(sample_transactions_base):
    """ Test ValueError from calculate_table if date column cannot be converted. """
    data = {
        DATE: ['2023-01-10', 'invalid-date-string'],
        TRANSACTION_TYPE: ['buy', 'sell'], TICKER: ['XYZ', 'XYZ'],
        SHARES: [100, 100], SHARE_PRICE: [10, 12]
    }
    transactions = pd.concat([sample_transactions_base, pd.DataFrame(data)], ignore_index=True)
    # Don't convert DATE here

    calculator = IrpfEarningsCalculator()
    with pytest.raises(ValueError, match=f"Could not convert date column '{DATE}' to datetime"):
        calculator.calculate_table(transactions)

def test_calculate_table_converts_date_column_if_possible(sample_transactions_base):
    """ Test that calculate_table converts date column if it's not already datetime. """
    data = {
        DATE: ['2023-01-10', '2023-03-15'], # Dates as strings
        TRANSACTION_TYPE: ['buy', 'sell'], TICKER: ['XYZ', 'XYZ'],
        SHARES: [100, 100], SHARE_PRICE: [10, 12]
    }
    transactions = pd.concat([sample_transactions_base, pd.DataFrame(data)], ignore_index=True)

    calculator = IrpfEarningsCalculator()
    # Call calculate_table and check the result (doesn't modify input)
    result_df = calculator.calculate_table(transactions)
    # We can't directly check the internal_df's date type easily,
    # but the fact it runs without error implies conversion worked.
    # Check the output structure
    assert isinstance(result_df, pd.DataFrame)
    assert RESULT_TAXABLE_GAIN_LOSS in result_df.columns
    assert RESULT_DEFERRED_ADJUSTMENT in result_df.columns
    assert result_df.index.equals(transactions.index) # Check index preservation
