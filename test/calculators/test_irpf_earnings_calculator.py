# test/calculators/test_irpf_earnings_calculator.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Import the calculator and result column names
from pyportfolio.calculators.irpf_earnings_calculator import (
    IrpfEarningsCalculator,
    GPP,
    RESULT_DEFERRED_ADJUSTMENT
)

# Import constants used in test data setup
from pyportfolio.columns import (
    DATETIME, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE, COMISION, # Added COMISION
    TYPE_BUY, TYPE_SELL, TYPE_DIVIDEND
)

# --- Fixtures for common data ---
@pytest.fixture
def sample_transactions_base():
    """ Provides an empty DataFrame with the expected structure including COMISION. """
    return pd.DataFrame({
        DATETIME: pd.to_datetime([]),
        TRANSACTION_TYPE: pd.Series([], dtype=str),
        TICKER: pd.Series([], dtype=str),
        SHARES: pd.Series([], dtype=float),
        SHARE_PRICE: pd.Series([], dtype=float),
        COMISION: pd.Series([], dtype=float) # Added COMISION
    })

# Helper function to create DataFrames
def create_df(data):
    df = pd.DataFrame(data)
    df[DATETIME] = pd.to_datetime(df[DATETIME], format='mixed', dayfirst=True)
    if COMISION not in df.columns:
        df[COMISION] = 0.0
    else:
        df[COMISION] = df[COMISION].astype(float)
    # Ensure SHARES is float
    if SHARES in df.columns:
        df[SHARES] = df[SHARES].astype(float)
    return df


# Helper to create transactions, run calculator directly, and merge results
def run_irpf_calc_direct(data, base_fixture):
    """
    Creates DataFrame from test data, ensures sorting,
    instantiates IrpfEarningsCalculator, calls calculate_table directly,
    and merges the results back with the input for easier testing.
    """
    df_input = pd.DataFrame(data)
    df_input[DATETIME] = pd.to_datetime(df_input[DATETIME], format='mixed', dayfirst=True)
    df_input[SHARES] = df_input[SHARES].astype(float) # Ensure SHARES is float
    if COMISION not in df_input.columns:             # Ensure COMISION exists
        df_input[COMISION] = 0.0
    df_input[COMISION] = df_input[COMISION].astype(float)

    transactions_input = pd.concat([base_fixture, df_input], ignore_index=True)

    # Ensure sorting by date and type (buys before sells on same day)
    transactions_input = transactions_input.sort_values(
        by=[DATETIME, TRANSACTION_TYPE],
        ascending=[True, True],
        # Use key for sorting TYPE_BUY before TYPE_SELL if dates are equal
        key=lambda col: col.map({TYPE_BUY: 0, TYPE_SELL: 1}) if col.name == TRANSACTION_TYPE else col
    ).reset_index(drop=True)

    # Instantiate calculator and call calculate_table
    calculator = IrpfEarningsCalculator()
    # Pass a copy to calculate_table to mimic TransactionManager behavior
    results_df = calculator.calculate_table(transactions_input.copy())

    # Merge results back to the sorted input based on index for easier assertions
    merged_df = transactions_input.join(results_df)
    return merged_df

# --- Test Cases ---

def test_irpf_gain_is_calculated(sample_transactions_base):
    """ Test gain: Sell row has gain, Buy row has 0.0. """
    data = {
        DATETIME: ['2023-01-10', '2023-03-15'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL],
        TICKER: ['XYZ', 'XYZ'],
        SHARES: [100.0, -100.0], # Sell shares negative
        SHARE_PRICE: [10, 12],
        COMISION: [5.0, 6.0] # Added commissions
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_BUY].iloc[0]
    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]

    # Buy Cost = 100*10 + 5 = 1005
    # Sell Proceeds = 100*12 - 6 = 1194
    # Gain = 1194 - 1005 = 189
    expected_gain = 189.0

    assert buy_row_result[GPP] == 0.0
    assert buy_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_row_result[GPP] == pytest.approx(expected_gain)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_loss_no_repurchase_within_window(sample_transactions_base):
    """ Test loss (no deferral): Sell row has loss, Buy rows have 0.0. """
    data = {
        DATETIME: ['2023-01-10', '2023-06-15', '2023-10-01'], # Buy is > 2 months after sell
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL, TYPE_BUY],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100.0, -100.0, 50.0], # Sell shares negative
        SHARE_PRICE: [10, 8, 7],
        COMISION: [5.0, 4.0, 2.5] # Added commissions
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_1_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_BUY].iloc[0]
    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]
    buy_row_2_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_BUY].iloc[1]

    # Buy 1 Cost = 100*10 + 5 = 1005
    # Sell Proceeds = 100*8 - 4 = 796
    # Loss = 796 - 1005 = -209
    # No repurchase in window -> Allowable Loss = -209
    expected_loss = -209.0

    assert buy_row_1_result[GPP] == 0.0
    assert buy_row_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_row_result[GPP] == pytest.approx(expected_loss)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_row_2_result[GPP] == 0.0
    assert buy_row_2_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_loss_deferred_due_to_repurchase_within_2_months_after(sample_transactions_base):
    """ Test loss partial deferral (buy after): Sell has Allowed Loss, Blocking Buy has Adjustment. """
    data = {
        DATETIME: ['2023-01-10', '2023-06-15', '2023-07-20'], # Repurchase within 2 months after
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL, TYPE_BUY],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100.0, -100.0, 50.0], # Sell 100, Repurchase 50. Sell shares negative
        SHARE_PRICE: [10, 8, 9],
        COMISION: [5.0, 4.0, 2.5] # Added commissions
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_initial_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_BUY].iloc[0]
    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]
    buy_row_blocking_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_BUY].iloc[1]

    # Sell 100@8. Cost 100@10.05 = 1005. Loss = 796 - 1005 = -209 (-2.09/share).
    # Repurchase (Buy 2) of 50 shares blocks 50 shares from the sell.
    # Deferred Value = 50 shares * abs(-2.09/share) = 104.5.
    # Buy 2 remaining blocking capacity = 50 - 50 = 0.
    # Allowed Loss = -209 + 104.5 = -104.5.
    expected_allowable_loss = -104.5
    expected_buy_adjustment = 104.5

    assert buy_row_initial_result[GPP] == 0.0
    assert buy_row_initial_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_row_result[GPP] == pytest.approx(expected_allowable_loss)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_row_blocking_result[GPP] == 0.0
    assert buy_row_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_buy_adjustment)

def test_irpf_loss_deferred_due_to_repurchase_within_2_months_before(sample_transactions_base):
    """ Test loss partial deferral (buy before): Sell has Allowed Loss, Blocking Buy has Adjustment. """
    data = {
        DATETIME: ['2023-01-10', '2023-05-20', '2023-06-15'], # Buy before within 2 months
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY, TYPE_SELL],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100.0, 50.0, -100.0], # Sell 100, Repurchase 50 before. Sell shares negative
        SHARE_PRICE: [10, 9, 8],
        COMISION: [5.0, 2.5, 4.0] # Added commissions
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_initial_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_BUY].iloc[0]
    buy_row_blocking_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_BUY].iloc[1]
    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]

    # Sell 100@8. Cost 100@10.05 = 1005. Loss = 796 - 1005 = -209 (-2.09/share).
    # Repurchase (Buy 2) of 50 shares blocks 50 shares from the sell.
    # Deferred Value = 50 shares * abs(-2.09/share) = 104.5.
    # Buy 2 remaining blocking capacity = 50 - 50 = 0.
    # Allowed Loss = -209 + 104.5 = -104.5.
    expected_allowable_loss = -104.5
    expected_buy_adjustment = 104.5

    assert buy_row_initial_result[GPP] == 0.0
    assert buy_row_initial_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0 # Not the blocker
    assert buy_row_blocking_result[GPP] == 0.0
    assert buy_row_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_buy_adjustment) # Is the blocker
    assert sell_row_result[GPP] == pytest.approx(expected_allowable_loss)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_loss_not_deferred_if_repurchase_is_different_ticker(sample_transactions_base):
    """ Test loss not deferred (diff ticker): Sell has loss, Buys have 0.0. """
    data = {
        DATETIME: ['2023-01-10', '2023-06-15', '2023-07-20'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL, TYPE_BUY],
        TICKER: ['XYZ', 'XYZ', 'ABC'], # Repurchase is 'ABC', sale is 'XYZ'
        SHARES: [100.0, -100.0, 50.0], # Sell shares negative
        SHARE_PRICE: [10, 8, 9],
        COMISION: [5.0, 4.0, 2.5] # Added commissions
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_xyz_result = processed_df[processed_df[TICKER] == 'XYZ'].iloc[0]
    sell_row_xyz_result = processed_df[processed_df[TICKER] == 'XYZ'].iloc[1]
    buy_row_abc_result = processed_df[processed_df[TICKER] == 'ABC'].iloc[0]

    # Sell 100@8. Cost 100@10.05 = 1005. Loss = 796 - 1005 = -209.
    # Repurchase is different ticker, so no deferral. Allowable Loss = -209.
    expected_loss = -209.0

    assert buy_row_xyz_result[GPP] == 0.0
    assert buy_row_xyz_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_row_xyz_result[GPP] == pytest.approx(expected_loss)
    assert sell_row_xyz_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_row_abc_result[GPP] == 0.0
    assert buy_row_abc_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_loss_deferred_and_gain_realized(sample_transactions_base):
    """ Test sequence: Deferred loss, then a Gain using adjusted cost basis. """
    data = {
        DATETIME: ['2023-01-10', '2023-05-20', '2023-06-15', '2023-07-20', '2023-09-10'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY, TYPE_SELL, TYPE_BUY, TYPE_SELL],
        TICKER: ['XYZ', 'XYZ', 'XYZ', 'XYZ', 'XYZ'],
        SHARES: [100.0, 50.0, -80.0, 60.0, -70.0], # Sell shares negative
        SHARE_PRICE: [10, 9, 8, 10, 12],
        COMISION: [5.0, 2.5, 4.0, 3.0, 3.5] # Added commissions
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_1_result = processed_df.iloc[0]
    buy_2_blocking_result = processed_df.iloc[1]
    sell_1_deferred_result = processed_df.iloc[2]
    buy_3_blocking_result = processed_df.iloc[3]
    sell_2_gain_result = processed_df.iloc[4]

    # Buy 1 Cost = 100*10 + 5 = 1005 (Cost/Sh = 10.05)
    # Buy 2 Cost = 50*9 + 2.5 = 452.5 (Cost/Sh = 9.05)
    # Buy 3 Cost = 60*10 + 3 = 603 (Cost/Sh = 10.05)

    # Sell 1: 80@8. Proceeds = 636. Cost 80@10.05 = 804. Loss = -168 (-2.1/share).
    # Blocked by Buy 2 (50 shares) and Buy 3 (30 shares). Allowable Loss = 0.
    # Adj on Buy 2 = 50 shares * |-2.1| = 105. Buy 2 rem capacity = 50-50=0. Buy 2 Adj Cost/Sh = 11.15.
    # Adj on Buy 3 = 30 shares * |-2.1| = 63. Buy 3 rem capacity = 60-30=30. Buy 3 Adj Cost/Sh = 11.10.
    expected_adj_buy2 = 105.0
    expected_adj_buy3 = 63.0

    # Sell 2: 70@12. Proceeds = 836.5.
    # Cost Basis:
    #   Consume 20 sh from Buy 1 (rem 100-80=20) -> 20 * 10.05 = 201.
    #   Consume 50 sh from Buy 2 -> 50 * 11.15 (adj cost) = 557.5.
    # Total Cost = 201 + 557.5 = 758.5.
    # Gain = 836.5 - 758.5 = 78.0.
    expected_gain_sell2 = 78.0

    assert buy_1_result[GPP] == 0.0
    assert buy_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_2_blocking_result[GPP] == 0.0
    assert buy_2_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_adj_buy2)
    assert sell_1_deferred_result[GPP] == 0.0 # Fully deferred
    assert sell_1_deferred_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_3_blocking_result[GPP] == 0.0
    assert buy_3_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_adj_buy3)
    assert sell_2_gain_result[GPP] == pytest.approx(expected_gain_sell2)
    assert sell_2_gain_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_loss_deferred_and_loss_realized(sample_transactions_base):
    """ Test sequence: Deferred loss, then a Loss using adjusted cost basis. """
    data = {
        DATETIME: ['2023-01-10', '2023-05-20', '2023-06-15', '2023-07-20', '2023-10-10'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY, TYPE_SELL, TYPE_BUY, TYPE_SELL],
        TICKER: ['XYZ', 'XYZ', 'XYZ', 'XYZ', 'XYZ'],
        SHARES: [100.0, 50.0, -80.0, 60.0, -80.0], # Sell shares negative
        SHARE_PRICE: [10, 9, 8, 10, 7],
        COMISION: [5.0, 2.5, 4.0, 3.0, 3.5] # Added commissions
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_1_result = processed_df.iloc[0]
    buy_2_blocking_result = processed_df.iloc[1]
    sell_1_deferred_result = processed_df.iloc[2]
    buy_3_blocking_result = processed_df.iloc[3]
    sell_2_loss_result = processed_df.iloc[4]

    # Sell 1: 80@8. Proceeds = 636. Cost 80@10.05 = 804. Loss = -168 (-2.1/share).
    # Blocked by Buy 2 (50 shares) and Buy 3 (30 shares). Allowable Loss = 0.
    # Adj on Buy 2 = 50 shares * |-2.1| = 105. Buy 2 rem capacity = 0. Buy 2 Adj Cost/Sh = 11.15.
    # Adj on Buy 3 = 30 shares * |-2.1| = 63. Buy 3 rem capacity = 30. Buy 3 Adj Cost/Sh = 11.10.
    expected_adj_buy2 = 105.0
    expected_adj_buy3 = 63.0

    # Sell 2: 80@7. Proceeds = 556.5.
    # Cost Basis:
    #   Consume 20 sh from Buy 1 (rem 100-80=20) -> 20 * 10.05 = 201.
    #   Consume 50 sh from Buy 2 -> 50 * 11.15 (adj cost) = 557.5.
    #   Consume 10 sh from Buy 3 (rem 60-30=30) -> 10 * 11.10 (adj cost) = 111.0.
    # Total Cost = 201 + 557.5 + 111.0 = 869.5.
    # Loss = 556.5 - 869.5 = -313.0.
    # No repurchases within +/- 2 months of Sell 2 (2023-10-10), so loss is allowable.
    expected_loss_sell2 = -313.0

    assert buy_1_result[GPP] == 0.0
    assert buy_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_2_blocking_result[GPP] == 0.0
    assert buy_2_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_adj_buy2)
    assert sell_1_deferred_result[GPP] == 0.0 # Fully deferred
    assert sell_1_deferred_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_3_blocking_result[GPP] == 0.0
    assert buy_3_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_adj_buy3)
    assert sell_2_loss_result[GPP] == pytest.approx(expected_loss_sell2)
    assert sell_2_loss_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_single_buy_blocks_two_separate_sells_capacity_logic(sample_transactions_base):
    """
    Test scenario where one TYPE_BUY (70 shares) blocks two 'sells' (50 shares each).
    Assumes the buy's blocking capacity (70) is consumed chronologically.
    """
    data = {
        DATETIME: ['2023-01-10', '2023-06-10', '2023-07-10', '2023-08-10'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL, TYPE_BUY, TYPE_SELL],
        TICKER: ['XYZ', 'XYZ', 'XYZ', 'XYZ'],
        SHARES: [100.0, -50.0, 70.0, -50.0], # Sell shares negative
        SHARE_PRICE: [10, 8, 9, 7],
        COMISION: [5.0, 2.5, 3.5, 2.5] # Added commissions
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_1_result = processed_df.iloc[0]
    sell_1_result = processed_df.iloc[1]
    buy_blocking_result = processed_df.iloc[2]
    sell_2_result = processed_df.iloc[3]

    # Sell 1: 50@8. Proceeds = 397.5. Cost 50@10.05 = 502.5. Loss -105 (-2.1/sh).
    # Buy Blocking (70 shares capacity) blocks 50 shares from Sell 1.
    # Deferred Value (Sell 1) = 50 shares * |-2.1| = 105. Allowable Loss (Sell 1) = 0.
    # Buy Blocking capacity used = 50. Remaining capacity = 70 - 50 = 20.
    allowable_sell1 = 0.0
    deferred_from_sell1 = 105.0

    # Sell 2: 50@7. Proceeds = 347.5. Cost 50@10.05 = 502.5. Loss -155 (-3.1/sh).
    # Buy Blocking (20 shares remaining capacity) blocks 20 shares from Sell 2.
    # Deferred Value (Sell 2, from this buy) = 20 shares * |-3.1| = 62.
    # Buy Blocking capacity used = 20. Remaining capacity = 20 - 20 = 0.
    # Allowable Loss (Sell 2) = -155 + 62 = -93.
    deferred_from_sell2_by_this_buy = 62.0
    expected_allowable_sell2 = -93.0

    # Total Adjustment on Buy Blocking = Deferred from Sell 1 + Deferred from Sell 2 = 105 + 62 = 167.
    expected_total_adjustment_on_buy = 167.0

    assert buy_1_result[GPP] == 0.0
    assert buy_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_1_result[GPP] == pytest.approx(allowable_sell1)
    assert sell_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_blocking_result[GPP] == 0.0
    assert buy_blocking_result[RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(expected_total_adjustment_on_buy)
    assert sell_2_result[GPP] == pytest.approx(expected_allowable_sell2)
    assert sell_2_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_handles_non_buy_sell_transactions(sample_transactions_base):
    """ Test Buy has 0.0, others have None. """
    data = {
        DATETIME: ['2023-01-10', '2023-03-15'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_DIVIDEND], # No TYPE_SELL
        TICKER: ['XYZ', 'XYZ'],
        SHARES: [100.0, 50.0], # Dividend shares 50
        SHARE_PRICE: [10, 2], # Dividend price 2
        COMISION: [5.0, 1.0] # Added commissions
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_result = processed_df.iloc[0]
    dividend_row_result = processed_df.iloc[1]

    assert buy_row_result[GPP] == 0
    assert buy_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0
    assert pd.isna(dividend_row_result[GPP])
    assert pd.isna(dividend_row_result[RESULT_DEFERRED_ADJUSTMENT])

def test_irpf_edge_case_exactly_two_months_before_exclusive(sample_transactions_base):
    """ Test loss NOT deferred (exact 2mo before): Sell has loss, Buy has 0.0. """
    data = {
        DATETIME: ['2023-01-15', '2023-04-15', '2023-06-15'], # Buy exactly 2 months before sell
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY, TYPE_SELL],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [50.0, 50.0, -100.0], # Sell shares negative
        SHARE_PRICE: [9, 9, 8],
        COMISION: [2.5, 2.5, 4.0] # Added commissions
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_1_result = processed_df.iloc[0]
    buy_row_2_potential_blocker = processed_df.iloc[1]
    sell_row_result = processed_df.iloc[2]

    # Buy 1 Cost = 50*9 + 2.5 = 452.5 (Cost/Sh = 9.05)
    # Buy 2 Cost = 50*9 + 2.5 = 452.5 (Cost/Sh = 9.05)
    # Sell Proceeds = 100*8 - 4 = 796.
    # Cost Basis = (50 * 9.05) + (50 * 9.05) = 905.
    # Loss = 796 - 905 = -109.
    # Buy 2 is exactly 2 months before, so window is > date, loss not deferred.
    expected_loss = -109.0

    assert buy_row_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_row_2_potential_blocker[GPP] == 0.0
    assert buy_row_2_potential_blocker[RESULT_DEFERRED_ADJUSTMENT] == 0.0 # Not blocked
    assert sell_row_result[GPP] == pytest.approx(expected_loss)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_edge_case_exactly_two_months_after_exclusive(sample_transactions_base):
    """ Test loss NOT deferred (exact 2mo after): Sell has loss, Buy has 0.0. """
    data = {
        DATETIME: ['2023-01-15', '2023-06-15', '2023-08-15'], # Buy exactly 2 months after sell
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL, TYPE_BUY],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        SHARES: [100.0, -100.0, 50.0], # Sell shares negative
        SHARE_PRICE: [10, 8, 9],
        COMISION: [5.0, 4.0, 2.5] # Added commissions
    }
    processed_df = run_irpf_calc_direct(data, sample_transactions_base)

    buy_row_1_result = processed_df.iloc[0]
    sell_row_result = processed_df.iloc[1]
    buy_row_2_potential_blocker = processed_df.iloc[2] # The potential blocker

    # Buy 1 Cost = 100*10 + 5 = 1005 (Cost/Sh = 10.05)
    # Sell Proceeds = 100*8 - 4 = 796.
    # Cost Basis = 100 * 10.05 = 1005.
    # Loss = 796 - 1005 = -209.
    # Buy 2 is exactly 2 months after, so window is < date, loss not deferred.
    expected_loss = -209.0

    assert buy_row_1_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert sell_row_result[GPP] == pytest.approx(expected_loss)
    assert sell_row_result[RESULT_DEFERRED_ADJUSTMENT] == 0.0
    assert buy_row_2_potential_blocker[GPP] == 0.0
    assert buy_row_2_potential_blocker[RESULT_DEFERRED_ADJUSTMENT] == 0.0 # Not blocked

# --- Validation Tests (calling calculate_table directly) ---

def test_calculate_table_raises_error_if_df_missing():
    """ Test ValueError if calculate_table called with non-DataFrame. """
    calculator = IrpfEarningsCalculator()
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        calculator.calculate_table(None) # type: ignore
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        calculator.calculate_table([1, 2, 3]) # type: ignore

def test_calculate_table_raises_error_if_missing_required_columns(sample_transactions_base):
    """ Test ValueError from calculate_table if DataFrame is missing essential columns. """
    # Add COMISION to valid data and required_cols check
    valid_data = {
        DATETIME: [datetime(2023,1,1)], TRANSACTION_TYPE: [TYPE_BUY], TICKER: ['T'],
        SHARES: [1.0], SHARE_PRICE: [1.0], COMISION: [0.1]
    }
    transactions_ok = pd.DataFrame(valid_data)
    transactions_ok[DATETIME] = pd.to_datetime(transactions_ok[DATETIME], format='mixed', dayfirst=True)

    calculator = IrpfEarningsCalculator()

    # Check missing columns needed by the calculator
    required_cols = [DATETIME, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE, COMISION]
    for col in required_cols:
        if col not in transactions_ok.columns: continue
        transactions_bad = transactions_ok.drop(columns=[col])
        with pytest.raises(ValueError, match=f"DataFrame must contain columns:.*{col}"):
             calculator.calculate_table(transactions_bad)

def test_calculate_table_raises_error_if_date_column_not_convertible(sample_transactions_base):
    """ Test ValueError from calculate_table if date column cannot be converted. """
    data = {
        DATETIME: ['10-01-2023', 'invalid-date-string'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL], TICKER: ['XYZ', 'XYZ'],
        SHARES: [100.0, -100.0], SHARE_PRICE: [10, 12], COMISION: [5.0, 6.0]
    }
    
    transactions = pd.concat([sample_transactions_base, pd.DataFrame(data)], ignore_index=True)    

    calculator = IrpfEarningsCalculator()
    with pytest.raises(ValueError, match=f"Could not convert"):        
        calculator.calculate_table(transactions)

def test_calculate_table_converts_date_column_if_possible(sample_transactions_base):
    """ Test that calculate_table converts date column if it's not already datetime. """
    data = {
        DATETIME: ['10-01-2023', '15-03-2023'], # Dates as strings
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL], TICKER: ['XYZ', 'XYZ'],
        SHARES: [100.0, -100.0], SHARE_PRICE: [10, 12], COMISION: [5.0, 6.0]
    }
    # Add COMISION to base fixture for concat
    base_fixture_with_comision = sample_transactions_base.copy()
    if COMISION not in base_fixture_with_comision.columns:
        base_fixture_with_comision[COMISION] = pd.Series([], dtype=float)

    transactions = pd.concat([base_fixture_with_comision, pd.DataFrame(data)], ignore_index=True)

    calculator = IrpfEarningsCalculator()
    # Call calculate_table and check the result (doesn't modify input)
    result_df = calculator.calculate_table(transactions)
    # Check the output structure
    assert isinstance(result_df, pd.DataFrame)
    assert GPP in result_df.columns
    assert RESULT_DEFERRED_ADJUSTMENT in result_df.columns
    assert result_df.index.equals(transactions.index) # Check index preservation

# Add COMISION column to the base fixture used by validation tests
@pytest.fixture(autouse=True)
def add_comision_to_base_fixture(sample_transactions_base):
    if COMISION not in sample_transactions_base.columns:
        sample_transactions_base[COMISION] = pd.Series([], dtype=float)