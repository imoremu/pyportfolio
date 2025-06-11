# test/calculators/test_fifo_calculator.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from pyportfolio.calculators.fifo_calculator import FIFOCalculator, RESULT_FIFO_GAIN_LOSS
from pyportfolio.columns import (
    DATETIME,
    TICKER,
    TRANSACTION_TYPE,
    SHARES,
    SHARE_PRICE,
    COMISION,
    TYPE_BUY,
    TYPE_DIVIDEND,
    TYPE_SELL
)

# --- Base Fixture ---
@pytest.fixture
def sample_transactions_fifo_base():
    """ Provides an empty DataFrame with the expected structure including COMISION. """
    return pd.DataFrame({
        DATETIME: pd.to_datetime([]),
        TICKER: pd.Series([], dtype=str),
        TRANSACTION_TYPE: pd.Series([], dtype=str),
        SHARES: pd.Series([], dtype=float),
        SHARE_PRICE: pd.Series([], dtype=float),
        COMISION: pd.Series([], dtype=float)
    })

# --- Helper to run the calculation ---
def run_fifo_calc_direct(data, base_fixture):
    """
    Creates DataFrame, instantiates FIFOCalculator (table-wise), calls calculate_table,
    and joins the results to the input for easier testing.
    """
    df_input = pd.DataFrame(data)
    df_input[DATETIME] = pd.to_datetime(df_input[DATETIME], format='mixed', dayfirst=True)
    df_input[SHARES] = df_input[SHARES].astype(float)
    df_input[SHARE_PRICE] = df_input[SHARE_PRICE].astype(float)
    if COMISION not in df_input.columns:
        df_input[COMISION] = 0.0
    df_input[COMISION] = df_input[COMISION].astype(float)

    transactions_input = pd.concat([base_fixture, df_input], ignore_index=True)

    transactions_input = transactions_input.sort_values(
        by=[TICKER, DATETIME, TRANSACTION_TYPE], ascending=[True, True, True]
    ).reset_index(drop=True)

    calculator = FIFOCalculator()
    results_df = calculator.calculate_table(transactions_input.copy())

    merged_df = transactions_input.join(results_df)
    return merged_df

# --- Test Cases ---

def test_fifo_simple(sample_transactions_fifo_base):
    """ Simple sell consumes from the first FIFO buy, including commissions. """
    data = {
        DATETIME: ['2023-01-10', '2023-02-15', '2023-03-20'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY, TYPE_SELL],
        SHARES: [10, 5, -8],
        SHARE_PRICE: [100, 120, 150],
        COMISION: [5, 2, 3]
    }
    processed_df = run_fifo_calc_direct(data, sample_transactions_fifo_base)

    # Buy 1 Cost Basis: (10 * 100 + 5) / 10 = 100.5
    # Sell Proceeds: (8 * 150) - 3 = 1197
    # Cost Basis for Sell: 8 * 100.5 = 804
    # Gain = 1197 - 804 = 393
    expected_gain = 393.0

    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]
    buy_rows_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_BUY]

    assert sell_row_result[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_gain)
    assert buy_rows_result[RESULT_FIFO_GAIN_LOSS].isna().all()

def test_fifo_partial_consumption_across_buys(sample_transactions_fifo_base):
    """ Sell consumes shares from multiple buys, including commissions. """
    data = {
        DATETIME: ['2023-01-10', '2023-02-15', '2023-03-20'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY, TYPE_SELL],
        SHARES: [5, 5, -8],
        SHARE_PRICE: [100, 120, 150],
        COMISION: [5, 2, 3]
    }
    processed_df = run_fifo_calc_direct(data, sample_transactions_fifo_base)

    # Buy 1 Cost Basis: (5 * 100 + 5) / 5 = 101
    # Buy 2 Cost Basis: (5 * 120 + 2) / 5 = 120.4
    # Sell Proceeds: (8 * 150) - 3 = 1197
    # Cost Basis for Sell: (5 * 101) + (3 * 120.4) = 505 + 361.2 = 866.2
    # Gain = 1197 - 866.2 = 330.8
    expected_gain = 330.8

    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]
    assert sell_row_result[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_gain)

def test_fifo_sell_exactly_all_shares(sample_transactions_fifo_base):
    """ Sell consumes exactly all purchased shares, including commissions. """
    data = {
        DATETIME: ['2023-01-10', '2023-02-15', '2023-03-20'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY, TYPE_SELL],
        SHARES: [4, 6, -10],
        SHARE_PRICE: [100, 110, 130],
        COMISION: [4, 6, 10]
    }
    processed_df = run_fifo_calc_direct(data, sample_transactions_fifo_base)

    # Buy 1 Cost Basis: (4 * 100 + 4) / 4 = 101
    # Buy 2 Cost Basis: (6 * 110 + 6) / 6 = 111
    # Sell Proceeds: (10 * 130) - 10 = 1290
    # Cost Basis for Sell: (4 * 101) + (6 * 111) = 404 + 666 = 1070
    # Gain = 1290 - 1070 = 220
    expected_gain = 220.0

    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]
    assert sell_row_result[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_gain)

def test_fifo_sell_more_than_available_raises_error(sample_transactions_fifo_base):
    """ Attempting to sell more shares than available should raise ValueError. """
    data = {
        DATETIME: ['2023-01-10', '2023-02-15', '2023-03-20'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY, TYPE_SELL],
        SHARES: [3, 2, -10],
        SHARE_PRICE: [100, 120, 150],
        COMISION: [1, 1, 1]
    }
    df_input = pd.DataFrame(data)
    df_input[DATETIME] = pd.to_datetime(df_input[DATETIME], format='mixed', dayfirst=True)
    df_input[SHARES] = df_input[SHARES].astype(float)
    df_input[SHARE_PRICE] = df_input[SHARE_PRICE].astype(float)
    df_input[COMISION] = df_input[COMISION].astype(float)
    transactions_input = pd.concat([sample_transactions_fifo_base, df_input], ignore_index=True)

    calculator = FIFOCalculator()

    with pytest.raises(ValueError) as exc_info:
        calculator.calculate_table(transactions_input)

    assert "Cannot sell 10.0000 shares of XYZ" in str(exc_info.value)
    assert "only 5.0000 are available" in str(exc_info.value)

def test_fifo_no_sell_transaction(sample_transactions_fifo_base):
    """ If there are no sells, the result column should be NA. """
    data = {
        DATETIME: ['2023-01-10', '2023-02-15'],
        TICKER: ['XYZ', 'XYZ'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY],
        SHARES: [10, 5],
        SHARE_PRICE: [100, 120],
        COMISION: [5, 2]
    }
    processed_df = run_fifo_calc_direct(data, sample_transactions_fifo_base)

    assert RESULT_FIFO_GAIN_LOSS in processed_df.columns
    assert processed_df[RESULT_FIFO_GAIN_LOSS].isna().all()

def test_fifo_multiple_sells_in_sequence(sample_transactions_fifo_base):
    """ Multiple sells processed correctly in order, including commissions. """
    data = {
        DATETIME: ['2023-01-10', '2023-02-15', '2023-03-20', '2023-04-25'],
        TICKER: ['XYZ', 'XYZ', 'XYZ', 'XYZ'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY, TYPE_SELL, TYPE_SELL],
        SHARES: [10, 5, -8, -5],
        SHARE_PRICE: [50, 60, 80, 100],
        COMISION: [10, 5, 8, 5]
    }
    processed_df = run_fifo_calc_direct(data, sample_transactions_fifo_base)

    sell_rows = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL]
    sell_1_result = sell_rows.iloc[0]
    sell_2_result = sell_rows.iloc[1]

    # Buy 1 Cost Basis: (10 * 50 + 10) / 10 = 51
    # Buy 2 Cost Basis: (5 * 60 + 5) / 5 = 61
    # Sell 1 Proceeds: (8 * 80) - 8 = 632
    # Cost Basis 1 = 8 * 51 = 408
    # Gain 1 = 632 - 408 = 224
    expected_gain1 = 224.0
    assert sell_1_result[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_gain1)

    # Sell 2 Proceeds: (5 * 100) - 5 = 495
    # Cost Basis 2 = (2 * 51) + (3 * 61) = 102 + 183 = 285
    # Gain 2 = 495 - 285 = 210
    expected_gain2 = 210.0
    assert sell_2_result[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_gain2)

def test_fifo_sell_ignores_future_buys_and_raises_error(sample_transactions_fifo_base):
    """ Sell cannot use future buys; raises error if not enough previous shares. """
    data = {
        DATETIME: ['2023-01-10', '2023-03-20', '2023-04-15'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL, TYPE_BUY],
        SHARES: [5, -6, 10],
        SHARE_PRICE: [100, 130, 80],
        COMISION: [2, 3, 4]
    }
    df_input = pd.DataFrame(data)
    df_input[DATETIME] = pd.to_datetime(df_input[DATETIME], format='mixed', dayfirst=True)
    df_input[SHARES] = df_input[SHARES].astype(float)
    df_input[SHARE_PRICE] = df_input[SHARE_PRICE].astype(float)
    df_input[COMISION] = df_input[COMISION].astype(float)
    transactions_input = pd.concat([sample_transactions_fifo_base, df_input], ignore_index=True)

    calculator = FIFOCalculator()

    with pytest.raises(ValueError) as exc_info:
        calculator.calculate_table(transactions_input)

    assert "Cannot sell 6.0000 shares of XYZ" in str(exc_info.value)
    assert "only 5.0000 are available" in str(exc_info.value)

def test_fifo_multiple_tickers(sample_transactions_fifo_base):
    """ Ensures that FIFO calculation is isolated per ticker, including commissions. """
    data_corrected = {
        DATETIME: ['2023-01-10', '2023-01-15', '2023-02-10', '2023-02-20', '2023-03-10', '2023-03-15'],
        TICKER: ['AAA', 'BBB', 'AAA', 'BBB', 'AAA', 'BBB'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY, TYPE_BUY, TYPE_SELL, TYPE_SELL, TYPE_SELL],
        SHARES: [10, 20, 5, -15, -12, -5],
        SHARE_PRICE: [10, 5, 12, 8, 15, 4],
        COMISION: [1, 2, 0.5, 1.5, 1.2, 0.5]
    }
    processed_df_corrected = run_fifo_calc_direct(data_corrected, sample_transactions_fifo_base)

    sell_aaa = processed_df_corrected[(processed_df_corrected[TICKER] == 'AAA') & (processed_df_corrected[TRANSACTION_TYPE] == TYPE_SELL)].iloc[0]
    sell_bbb_1 = processed_df_corrected[(processed_df_corrected[TICKER] == 'BBB') & (processed_df_corrected[TRANSACTION_TYPE] == TYPE_SELL)].iloc[0]
    sell_bbb_2_corrected = processed_df_corrected[(processed_df_corrected[TICKER] == 'BBB') & (processed_df_corrected[TRANSACTION_TYPE] == TYPE_SELL)].iloc[1]

    # AAA Buy 1 Cost Basis: (10 * 10 + 1) / 10 = 10.1
    # AAA Buy 2 Cost Basis: (5 * 12 + 0.5) / 5 = 12.1
    # AAA Sell Proceeds: (12 * 15) - 1.2 = 178.8
    # AAA Cost Basis = (10 * 10.1) + (2 * 12.1) = 101 + 24.2 = 125.2
    # AAA Gain = 178.8 - 125.2 = 53.6
    expected_gain_aaa = 53.6
    assert sell_aaa[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_gain_aaa)

    # BBB Buy 1 Cost Basis: (20 * 5 + 2) / 20 = 5.1
    # BBB Sell 1 Proceeds: (15 * 8) - 1.5 = 118.5
    # BBB Cost Basis 1 = 15 * 5.1 = 76.5
    # BBB Gain 1 = 118.5 - 76.5 = 42.0
    expected_gain_bbb1 = 42.0
    assert sell_bbb_1[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_gain_bbb1)

    # BBB Sell 2 Proceeds: (5 * 4) - 0.5 = 19.5
    # BBB Cost Basis 2 = 5 * 5.1 = 25.5
    # BBB Loss 2 = 19.5 - 25.5 = -6.0
    expected_loss_bbb2 = -6.0
    assert sell_bbb_2_corrected[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_loss_bbb2)


def test_fifo_handles_other_transaction_types(sample_transactions_fifo_base):
    """ Other transaction types should have NA result. """
    data = {
        DATETIME: ['2023-01-10', '2023-02-15', '2023-03-20'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_DIVIDEND, TYPE_SELL],
        SHARES: [10, np.nan, -5],
        SHARE_PRICE: [100, np.nan, 150],
        COMISION: [5, 0, 2]
    }
    processed_df = run_fifo_calc_direct(data, sample_transactions_fifo_base)

    buy_row = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_BUY].iloc[0]
    dividend_row = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_DIVIDEND ].iloc[0]
    sell_row = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]

    assert pd.isna(buy_row[RESULT_FIFO_GAIN_LOSS])
    assert pd.isna(dividend_row[RESULT_FIFO_GAIN_LOSS])

    # Buy 1 Cost Basis: (10 * 100 + 5) / 10 = 100.5
    # Sell Proceeds: (5 * 150) - 2 = 748
    # Cost Basis = 5 * 100.5 = 502.5
    # Gain = 748 - 502.5 = 245.5
    expected_gain = 245.5
    assert sell_row[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_gain)

def test_fifo_sell_with_zero_or_positive_shares_is_ignored(sample_transactions_fifo_base):
    """ Sell rows with zero or positive shares should be ignored (result NA). """
    data = {
        DATETIME: ['2023-01-10', '2023-02-15', '2023-03-20', '2023-04-10'],
        TICKER: ['XYZ', 'XYZ', 'XYZ', 'XYZ'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL, TYPE_SELL, TYPE_SELL],
        SHARES: [10, 0, 5, -2],
        SHARE_PRICE: [100, 110, 120, 130],
        COMISION: [10, 0, 0, 2]
    }
    processed_df = run_fifo_calc_direct(data, sample_transactions_fifo_base)

    sell_rows = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL]
    sell_zero_shares = sell_rows.iloc[0]
    sell_positive_shares = sell_rows.iloc[1]
    sell_valid = sell_rows.iloc[2]

    assert pd.isna(sell_zero_shares[RESULT_FIFO_GAIN_LOSS])
    assert pd.isna(sell_positive_shares[RESULT_FIFO_GAIN_LOSS])

    # Buy 1 Cost Basis: (10 * 100 + 10) / 10 = 101
    # Valid Sell Proceeds: (2 * 130) - 2 = 258
    # Cost Basis = 2 * 101 = 202
    # Gain = 258 - 202 = 56
    expected_gain = 56.0
    assert sell_valid[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_gain)


def test_fifo_simple(sample_transactions_fifo_base):
    """ Simple sell consumes from the first FIFO buy, including commissions. """
    data = {
        DATETIME: ['2023-01-10', '2023-02-15', '2023-03-20'],
        TICKER: ['XYZ', 'XYZ', 'XYZ'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_BUY, TYPE_SELL],
        SHARES: [10, 5, -8],
        SHARE_PRICE: [100, 120, 150],
        COMISION: [5, 2, 3]
    }
    processed_df = run_fifo_calc_direct(data, sample_transactions_fifo_base)

    # Buy 1 Cost Basis: (10 * 100 + 5) / 10 = 100.5
    # Sell Proceeds: (8 * 150) - 3 = 1197
    # Cost Basis for Sell: 8 * 100.5 = 804
    # Gain = 1197 - 804 = 393
    expected_gain = 393.0

    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]
    buy_rows_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_BUY]

    assert sell_row_result[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_gain)
    assert buy_rows_result[RESULT_FIFO_GAIN_LOSS].isna().all()

def test_fifo_precission(sample_transactions_fifo_base):
    """ Sell consumes shares from multiple buys, including commissions. """
    data = {
        DATETIME: ['2023-02-01', '2024-12-01'],
        TICKER: ['XYZ', 'XYZ'],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL],
        SHARES: [20, -20],
        SHARE_PRICE: [100/1.1, 90/1.04],
        COMISION: [1/1.1, 1/1.04]
    }
    
    # Numerically:
    # Buy Total Cost = (20 * 90.90909090909091) + 0.9090909090909091 = 1818.181818181818 + 0.9090909090909091 = 1819.090909090909
    # Buy Cost per Share = 1819.090909090909 / 20 = 90.95454545454545
    # Sell Proceeds = (20 * 86.53846153846153) - 0.9615384615384615 = 1730.7692307692307 - 0.9615384615384615 = 1729.8076923076924

    # expected_gain =  1729.8076923076924 - (20 * 90.95454545454545) = 1729.8076923076924 - 1819.090909090909 = -89.2832167832166

    processed_df = run_fifo_calc_direct(data, sample_transactions_fifo_base)

    buy_shares = 20
    buy_price = 100 / 1.1
    buy_commission = 1 / 1.1
    sell_shares = 20
    sell_price = 90 / 1.04
    sell_commission = 1 / 1.04

    buy_total_cost = (buy_shares * buy_price) + buy_commission
    buy_cost_per_share = buy_total_cost / buy_shares

    sell_proceeds = (sell_shares * sell_price) - sell_commission

    cost_basis_sell = sell_shares * buy_cost_per_share

    gain = sell_proceeds - cost_basis_sell

    expected_gain = -89.28321678

    sell_row_result = processed_df[processed_df[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]

    assert sell_row_result[RESULT_FIFO_GAIN_LOSS] == pytest.approx(expected_gain, rel=1e-9, abs=1e-9)


    