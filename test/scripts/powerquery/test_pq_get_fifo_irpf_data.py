import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Function to test
from pyportfolio.scripts.powerquery.pq_get_fifo_irpf_data import pq_get_fifo_irpf_data

# Column constants used in the script and for assertions
from pyportfolio.columns import (
    DATETIME, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE, COMISION,
    TYPE_BUY, TYPE_SELL, TYPE_DIVIDEND
)
from pyportfolio.calculators.fifo_calculator import RESULT_FIFO_GAIN_LOSS
from pyportfolio.calculators.irpf_earnings_calculator import (
    RESULT_TAXABLE_GAIN_LOSS,
    RESULT_DEFERRED_ADJUSTMENT
)

# --- Constants used by the script under test ---
SCRIPT_TEST_COLUMN_NAME = 'Test'
SCRIPT_RCM_COLUMN_NAME = "Rendimientos Capital Mobiliario" # As defined in pq_get_fifo_irpf_data

@pytest.fixture
def base_input_columns():
    """Defines the basic input columns for test DataFrames."""
    return [DATETIME, SCRIPT_TEST_COLUMN_NAME, TICKER, TRANSACTION_TYPE, SHARES, SHARE_PRICE, COMISION]

def create_test_df(data_list: list, columns: list) -> pd.DataFrame:
    """Helper to create DataFrames for tests."""
    df = pd.DataFrame(data_list, columns=columns)
    if DATETIME in df.columns:
        df[DATETIME] = pd.to_datetime(df[DATETIME], format='%Y-%m-%d')
    if SHARES in df.columns:
        df[SHARES] = df[SHARES].astype(float)
    if SHARE_PRICE in df.columns:
        df[SHARE_PRICE] = df[SHARE_PRICE].astype(float)
    if COMISION in df.columns:
        df[COMISION] = df[COMISION].astype(float)
    return df

# --- Test Cases ---

def test_empty_input_dataframe(base_input_columns):
    """Test with an empty input DataFrame."""
    empty_df = create_test_df([], base_input_columns)
    result_df = pq_get_fifo_irpf_data(empty_df)

    assert result_df.empty
    expected_calculated_cols = [RESULT_FIFO_GAIN_LOSS, SCRIPT_RCM_COLUMN_NAME, RESULT_TAXABLE_GAIN_LOSS, RESULT_DEFERRED_ADJUSTMENT]
    for col in base_input_columns + expected_calculated_cols:
        assert col in result_df.columns

def test_missing_test_column(base_input_columns):
    """Test when the mandatory SCRIPT_TEST_COLUMN_NAME is missing."""
    columns_without_test = [col for col in base_input_columns if col != SCRIPT_TEST_COLUMN_NAME]
    data = [
        ['2023-01-01', 'TICKA', TYPE_BUY, 10.0, 100.0, 1.0],
    ]
    df_missing_col = create_test_df(data, columns_without_test)
    result_df = pq_get_fifo_irpf_data(df_missing_col)

    assert not result_df.empty
    assert 'Error' in result_df.columns
    assert result_df['Error'].iloc[0] == f"Missing test column: {SCRIPT_TEST_COLUMN_NAME}"
    # Check that calculated columns exist and are NA
    expected_calculated_cols = [RESULT_FIFO_GAIN_LOSS, SCRIPT_RCM_COLUMN_NAME, RESULT_TAXABLE_GAIN_LOSS, RESULT_DEFERRED_ADJUSTMENT]
    for col in expected_calculated_cols:
        assert col in result_df.columns
        assert result_df[col].isna().all()


def test_simple_gain_and_dividend_scenario(base_input_columns, caplog):
    """Test a scenario with a buy, a dividend, and a sell resulting in a gain."""
    # This test assumes DividendCalculator is instantiated correctly in the script
    # e.g., dividend_calculator = DividendCalculator(dataset_sorted)
    data = [
        ['2023-01-01', 'Test1', 'TICKA', TYPE_BUY, 10.0, 100.0, 1.0],  # Buy 10 @ 100, com 1. Cost/sh = (1000+1)/10 = 100.1
        ['2023-01-15', 'Test1', 'TICKA', TYPE_DIVIDEND, 10.0, 2.0, 0.5], # Dividend 10*2 - 0.5 = 19.5
        ['2023-02-01', 'Test1', 'TICKA', TYPE_SELL, -10.0, 120.0, 2.0], # Sell 10 @ 120, com 2. Proceeds = 1200-2 = 1198
                                                                    # FIFO Gain = 1198 - (10 * 100.1) = 1198 - 1001 = 197
                                                                    # IRPF GPP = 197 (no deferral)
    ]
    input_df = create_test_df(data, base_input_columns)
    
    result_df = pq_get_fifo_irpf_data(input_df)
    
    assert 'Error' not in result_df.columns or result_df['Error'].isna().all()

    # Row 0 (BUY)
    assert pd.isna(result_df.loc[0, RESULT_FIFO_GAIN_LOSS])
    assert pd.isna(result_df.loc[0, SCRIPT_RCM_COLUMN_NAME])
    assert result_df.loc[0, RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert result_df.loc[0, RESULT_DEFERRED_ADJUSTMENT] == 0.0

    # Row 1 (DIVIDEND)
    assert pd.isna(result_df.loc[1, RESULT_FIFO_GAIN_LOSS])
    assert result_df.loc[1, SCRIPT_RCM_COLUMN_NAME] == 20 # Note: Spanish legislaton does not allow to deduct comissions from dividends
    assert pd.isna(result_df.loc[1, RESULT_TAXABLE_GAIN_LOSS]) # IRPF calc ignores dividends for GPP
    assert pd.isna(result_df.loc[1, RESULT_DEFERRED_ADJUSTMENT])

    # Row 2 (SELL)
    assert result_df.loc[2, RESULT_FIFO_GAIN_LOSS] == pytest.approx(197.0)
    assert pd.isna(result_df.loc[2, SCRIPT_RCM_COLUMN_NAME])
    assert result_df.loc[2, RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(197.0)
    assert result_df.loc[2, RESULT_DEFERRED_ADJUSTMENT] == 0.0

def test_irpf_loss_deferral_scenario(base_input_columns):
    """Test IRPF loss deferral with a blocking buy."""
    data = [
        # Test Group: Test_Defer, Ticker: LDEF
        ['2023-01-01', 'Test_Defer', 'LDEF', TYPE_BUY, 100.0, 10.0, 5.0],  # Buy1: 100 sh @ 10, com 5. Cost/sh = (1000+5)/100 = 10.05
        ['2023-06-01', 'Test_Defer', 'LDEF', TYPE_SELL, -100.0, 8.0, 4.0], # Sell1: 100 sh @ 8, com 4. Proceeds = 800-4 = 796.
                                                                        # FIFO Loss = 796 - (100*10.05) = 796 - 1005 = -209. Loss/sh = -2.09
        ['2023-07-01', 'Test_Defer', 'LDEF', TYPE_BUY, 50.0, 9.0, 2.0],   # Buy2 (Blocker): 50 sh @ 9, com 2. Blocks 50 shares of Sell1.
                                                                        # Deferred Loss for Sell1 = 50 * 2.09 = 104.5
                                                                        # Allowable Loss for Sell1 = -209 + 104.5 = -104.5
                                                                        # Adjustment on Buy2 = 104.5
    ]
    input_df = create_test_df(data, base_input_columns)
    result_df = pq_get_fifo_irpf_data(input_df)

    assert 'Error' not in result_df.columns or result_df['Error'].isna().all()

    # Row 0 (Initial BUY)
    assert result_df.loc[0, RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert result_df.loc[0, RESULT_DEFERRED_ADJUSTMENT] == 0.0

    # Row 1 (SELL with deferred loss)
    assert result_df.loc[1, RESULT_FIFO_GAIN_LOSS] == pytest.approx(-209.0)
    assert result_df.loc[1, RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(-104.5)
    assert result_df.loc[1, RESULT_DEFERRED_ADJUSTMENT] == 0.0

    # Row 2 (Blocking BUY)
    assert result_df.loc[2, RESULT_TAXABLE_GAIN_LOSS] == 0.0
    assert result_df.loc[2, RESULT_DEFERRED_ADJUSTMENT] == pytest.approx(104.5)

def test_multiple_groups_and_tickers(base_input_columns):
    """Test calculations are isolated across different groups and tickers."""
    data = [
        # Group: G1, Ticker: AAA
        ['2023-01-01', 'G1', 'AAA', TYPE_BUY, 10.0, 10.0, 1.0],   # Cost/sh = 10.1
        ['2023-05-05', 'G1', 'AAA', TYPE_SELL, -5.0, 12.0, 0.5],  # Proceeds = 59.5. Cost = 5*10.1=50.5. Gain = 9.0
        # Group: G1, Ticker: BBB
        ['2023-01-02', 'G1', 'BBB', TYPE_BUY, 20.0, 5.0, 2.0],    # Cost/sh = 5.1
        ['2023-06-06', 'G1', 'BBB', TYPE_SELL, -10.0, 4.0, 1.0],  # Proceeds = 39. Cost = 10*5.1=51. Loss = -12.0
        # Group: G2, Ticker: AAA
        ['2023-02-01', 'G2', 'AAA', TYPE_BUY, 8.0, 15.0, 0.8],    # Cost/sh = 15.1
        ['2023-05-05', 'G2', 'AAA', TYPE_SELL, -8.0, 18.0, 0.8],  # Proceeds = 143.2. Cost = 8*15.1=120.8. Gain = 22.4
    ]
    input_df = create_test_df(data, base_input_columns)
    result_df = pq_get_fifo_irpf_data(input_df)

    assert 'Error' not in result_df.columns or result_df['Error'].isna().all()

    # G1, AAA, SELL (index 1)
    assert result_df.loc[1, RESULT_FIFO_GAIN_LOSS] == pytest.approx(9.0)
    assert result_df.loc[1, RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(9.0)

    # G1, BBB, SELL (index 3)
    assert result_df.loc[3, RESULT_FIFO_GAIN_LOSS] == pytest.approx(-12.0)
    assert result_df.loc[3, RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(-12.0) # Assuming no deferral

    # G2, AAA, SELL (index 5)
    assert result_df.loc[5, RESULT_FIFO_GAIN_LOSS] == pytest.approx(22.4)
    assert result_df.loc[5, RESULT_TAXABLE_GAIN_LOSS] == pytest.approx(22.4)

