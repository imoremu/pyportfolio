# c:\Users\imore\SynologyDrive\Proyectos\pyportfolio\test\calculators\test_irpf_earnings_calculator_v2.py 
# Renamed to avoid confusion with previous versions
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Assuming IrpfEarningsCalculator is updated to implement the new logic
from pyportfolio.calculators.irpf_earnings_calculator import IrpfEarningsCalculator 
# from pyportfolio.calculators.base_calculator import BaseCalculator # If needed for type hints

# --- Fixtures for common data ---
@pytest.fixture
def sample_transactions_base():
    """ Base DataFrame structure to be extended in tests """
    return pd.DataFrame({
        'Date': pd.to_datetime([]),
        'Transaction Type': pd.Series([], dtype=str),
        'Ticker': pd.Series([], dtype=str),
        'Quantity': pd.Series([], dtype=float),
        'Price': pd.Series([], dtype=float),
        'FIFO_Gain_Loss': pd.Series([], dtype=float) # Pre-calculated FIFO results
    })

# Helper to create transactions and calculator
def create_calc_from_data(data, base_fixture):
    """ Creates DataFrame, ensures sorting, and initializes calculator """
    df = pd.DataFrame(data)
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # Combine with base if needed
    transactions = pd.concat([base_fixture, df], ignore_index=True)
    # Ensure sorting by date and type (buys before sells on same day), crucial for logic
    transactions = transactions.sort_values(
        by=['Date', 'Transaction Type'], 
        ascending=[True, True],
        key=lambda col: col.map({'buy': 0, 'sell': 1}) if col.name == 'Transaction Type' else col # Ensure buys process first on same day if needed by calc logic
    ).reset_index(drop=True)
    # IMPORTANT: The IrpfEarningsCalculator needs to be adapted to handle this new logic
    # It will likely need to perform lookups within the calculate method
    calculator = IrpfEarningsCalculator(transactions_df=transactions) 
    return calculator, transactions

# --- Test Cases ---

def test_irpf_gain_is_returned_directly(sample_transactions_base):
    """ Test gain: Sell returns (gain, 0.0), Buy returns (0.0, 0.0). """
    data = {
        'Date': ['2023-01-10', '2023-03-15'],
        'Transaction Type': ['buy', 'sell'],
        'Ticker': ['XYZ', 'XYZ'],
        'Quantity': [100, 100],
        'Price': [10, 12],
        'FIFO_Gain_Loss': [np.nan, 200.0] # Positive gain
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row = transactions[transactions['Transaction Type'] == 'buy'].iloc[0]
    sell_row = transactions[transactions['Transaction Type'] == 'sell'].iloc[0]
    
    result_buy = calculator.calculate(buy_row)
    result_sell = calculator.calculate(sell_row)

    assert result_buy == (0.0, 0.0), "Buy row should have (0.0, 0.0)"
    assert result_sell == (200.0, 0.0), "Sell row should have (Gain, 0.0)"

def test_irpf_loss_no_repurchase_within_window(sample_transactions_base):
    """ Test loss (no deferral): Sell returns (loss, 0.0), Buys return (0.0, 0.0). """
    data = {
        'Date': ['2023-01-10', '2023-06-15', '2023-10-01'],
        'Transaction Type': ['buy', 'sell', 'buy'],
        'Ticker': ['XYZ', 'XYZ', 'XYZ'],
        'Quantity': [100, 100, 50],
        'Price': [10, 8, 7],
        'FIFO_Gain_Loss': [np.nan, -200.0, np.nan] # Negative loss
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row_1 = transactions[transactions['Transaction Type'] == 'buy'].iloc[0]
    sell_row = transactions[transactions['Transaction Type'] == 'sell'].iloc[0]
    buy_row_2 = transactions[transactions['Transaction Type'] == 'buy'].iloc[1]

    result_buy_1 = calculator.calculate(buy_row_1)
    result_sell = calculator.calculate(sell_row)
    result_buy_2 = calculator.calculate(buy_row_2)

    # Loss occurred on 2023-06-15. Next buy is 2023-10-01 (> 2 months after).
    assert result_buy_1 == (0.0, 0.0), "Initial Buy row should be (0.0, 0.0)"
    assert result_sell == (-200.0, 0.0), "Sell row should have (Loss, 0.0) as not deferred"
    assert result_buy_2 == (0.0, 0.0), "Later Buy row should be (0.0, 0.0)"

def test_irpf_loss_deferred_due_to_repurchase_within_2_months_after(sample_transactions_base):
    """ Test loss partial deferral (buy after): Sell=(Allowed Loss, 0.0), Blocking Buy=(0.0, +DeferredLoss). """
    data = {
        'Date': ['2023-01-10', '2023-06-15', '2023-07-20'], # Repurchase within 2 months after
        'Transaction Type': ['buy', 'sell', 'buy'],
        'Ticker': ['XYZ', 'XYZ', 'XYZ'],
        'Quantity': [100, 100, 50], # Sell 100, Repurchase 50
        'Price': [10, 8, 9],
        'FIFO_Gain_Loss': [np.nan, -200.0, np.nan] # Loss = -200
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row_initial = transactions[transactions['Transaction Type'] == 'buy'].iloc[0]
    sell_row = transactions[transactions['Transaction Type'] == 'sell'].iloc[0]
    buy_row_blocking = transactions[transactions['Transaction Type'] == 'buy'].iloc[1]

    result_buy_initial = calculator.calculate(buy_row_initial)
    result_sell = calculator.calculate(sell_row)
    result_buy_blocking = calculator.calculate(buy_row_blocking)

    # Loss occurred on 2023-06-15 (-200 for 100 shares = -2/share). 
    # Repurchase on 2023-07-20 (50 shares) blocks 50 shares.
    # Deferred Loss = 50 * -2 = -100. Adjustment = 100.
    # Allowed Loss = (100 - 50) * -2 = -100.
    
    assert result_buy_initial == (0.0, 0.0), "Initial Buy row should be (0.0, 0.0)"
    assert result_sell == (-100.0, 0.0), "Sell row loss is partially deferred, should return (Allowed Loss, 0.0)"
    # The blocking buy gets the positive deferred loss value for the 50 shares it blocks
    assert result_buy_blocking == (0.0, 100.0), "Blocking Buy row (50 shares) should have (0.0, +DeferredLoss Portion)"

def test_irpf_loss_deferred_due_to_repurchase_within_2_months_before(sample_transactions_base):
    """ Test loss partial deferral (buy before): Sell=(Allowed Loss, 0.0), Blocking Buy=(0.0, +DeferredLoss). """
    data = {
        'Date': ['2023-01-10', '2023-05-20', '2023-06-15'], # Buy before within 2 months
        'Transaction Type': ['buy', 'buy', 'sell'], 
        'Ticker': ['XYZ', 'XYZ', 'XYZ'],
        'Quantity': [100, 50, 100], # Sell 100, Repurchase 50 before
        'Price': [10, 9, 8],
        'FIFO_Gain_Loss': [np.nan, np.nan, -200.0] # Sell 100@8 (50@9, 50@10) -> Loss = -150
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row_initial = transactions[transactions['Transaction Type'] == 'buy'].iloc[0]
    buy_row_blocking = transactions[transactions['Transaction Type'] == 'buy'].iloc[1]
    sell_row = transactions[transactions['Transaction Type'] == 'sell'].iloc[0]
    
    result_buy_initial = calculator.calculate(buy_row_initial)
    result_buy_blocking = calculator.calculate(buy_row_blocking)
    result_sell = calculator.calculate(sell_row)

    # Loss occurred on 2023-06-15 (-200 for 100 shares = -2/share).
    # Repurchase on 2023-05-20 (50 shares) blocks 50 shares.
    # Deferred Loss = 50 * -2 = -100. Adjustment = 100.
    # Allowed Loss = (100 - 50) * -2 = -100.

    assert result_buy_initial == (0.0, 0.0), "Initial Buy row should be (0.0, 0.0)"
    # The blocking buy gets the positive deferred loss value for the 50 shares it blocks
    assert result_buy_blocking == (0.0, 100.0), "Blocking Buy row (50 shares) should have (0.0, +DeferredLoss Portion)"
    assert result_sell == (-100.0, 0.0), "Sell row loss is partially deferred, should return (Allowed Loss, 0.0)"


def test_irpf_loss_not_deferred_if_repurchase_is_different_ticker(sample_transactions_base):
    """ Test loss not deferred (diff ticker): Sell returns (loss, 0.0), Buys return (0.0, 0.0). """
    data = {
        'Date': ['2023-01-10', '2023-06-15', '2023-07-20'],
        'Transaction Type': ['buy', 'sell', 'buy'],
        'Ticker': ['XYZ', 'XYZ', 'ABC'], # Repurchase is 'ABC', sale is 'XYZ'
        'Quantity': [100, 100, 50],
        'Price': [10, 8, 9],
        'FIFO_Gain_Loss': [np.nan, -200.0, np.nan]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row_xyz = transactions[transactions['Ticker'] == 'XYZ'].iloc[0]
    sell_row_xyz = transactions[transactions['Ticker'] == 'XYZ'].iloc[1]
    buy_row_abc = transactions[transactions['Ticker'] == 'ABC'].iloc[0]

    result_buy_xyz = calculator.calculate(buy_row_xyz)
    result_sell_xyz = calculator.calculate(sell_row_xyz)
    result_buy_abc = calculator.calculate(buy_row_abc)


    # Loss occurred on 2023-06-15 (XYZ). Repurchase on 2023-07-20 is 'ABC'.
    assert result_buy_xyz == (0.0, 0.0), "Initial XYZ Buy row should be (0.0, 0.0)"
    assert result_sell_xyz == (-200.0, 0.0), "XYZ Sell row loss NOT deferred, should return (Loss, 0.0)"
    assert result_buy_abc == (0.0, 0.0), "ABC Buy row is irrelevant, should be (0.0, 0.0)"

def test_irpf_loss_deferred_and_gain_realized(sample_transactions_base):
    """ Test sequence: Deferred loss (-160), then Gain (190). Blocking buys get proportional adjustment. """
    data = {
        'Date': ['2023-01-10', '2023-05-20', '2023-06-15', '2023-07-20', '2023-09-10'],
        'Transaction Type': ['buy', 'buy', 'sell', 'buy', 'sell'],
        'Ticker': ['XYZ', 'XYZ', 'XYZ', 'XYZ', 'XYZ'],
        'Quantity': [100, 50, 80, 60, 70], # Sell 80, blocked by Buy 50 (before) and Buy 60 (after)
        'Price': [10, 9, 8, 10, 12],
        'FIFO_Gain_Loss': [np.nan, np.nan, -160.0, np.nan, 190.0] # Loss = -160 on 80 shares (-2/share)
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    # Get rows by index after sorting
    buy_1_row = transactions.iloc[0] # 2023-01-10
    buy_2_blocking_row = transactions.iloc[1] # 2023-05-20 (Blocks 50 shares)
    sell_1_deferred_row = transactions.iloc[2] # 2023-06-15 (Loss deferred)
    buy_3_blocking_row = transactions.iloc[3] # 2023-07-20 (Blocks remaining 30 shares)
    sell_2_gain_row = transactions.iloc[4] # 2023-09-10 (Gain)

    result_buy_1 = calculator.calculate(buy_1_row)
    result_buy_2_blocking = calculator.calculate(buy_2_blocking_row)
    result_sell_1_deferred = calculator.calculate(sell_1_deferred_row)
    result_buy_3_blocking = calculator.calculate(buy_3_blocking_row)
    result_sell_2_gain = calculator.calculate(sell_2_gain_row)

    loss_per_share = -160.0 / 80.0 # = -2.0
    shares_sold = 80.0

    # Buy 2 blocks min(50, shares_sold) = 50 shares
    adj_buy2 = abs(loss_per_share * 50) # = 100.0
    remaining_shares_to_block = shares_sold - 50 # = 30

    # Buy 3 blocks min(60, remaining_shares_to_block) = 30 shares
    adj_buy3 = abs(loss_per_share * 30) # = 60.0

    assert result_buy_1 == (0.0, 0.0), "Initial Buy"
    # Buy 2 blocks 50 shares, gets proportional adjustment
    assert result_buy_2_blocking == (0.0, adj_buy2), f"Blocking Buy (before, 50 shares) gets adjustment {adj_buy2}"
    assert result_sell_1_deferred == (0.0, 0.0), "Deferred Sell"
    # Buy 3 blocks remaining 30 shares, gets proportional adjustment
    assert result_buy_3_blocking == (0.0, adj_buy3), f"Blocking Buy (after, blocks 30 shares) gets adjustment {adj_buy3}"
    assert result_sell_2_gain == (190.0, 0.0), "Gain Sell"

def test_irpf_loss_deferred_and_loss_realized(sample_transactions_base):
    """ Test sequence: Deferred loss (-160), then Gain (190). Blocking buys get proportional adjustment. """
    data = {
        'Date': ['2023-01-10', '2023-05-20', '2023-06-15', '2023-07-20', '2023-10-10'],
        'Transaction Type': ['buy', 'buy', 'sell', 'buy', 'sell'],
        'Ticker': ['XYZ', 'XYZ', 'XYZ', 'XYZ', 'XYZ'],
        'Quantity': [100, 50, 80, 60, 80], # Sell 80, blocked by Buy 50 (before) and Buy 60 (after)
        'Price': [10, 9, 8, 10, 7],
        'FIFO_Gain_Loss': [np.nan, np.nan, -160.0, np.nan, .0] # Loss = -160 on 80 shares (-2/share)
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    # Get rows by index after sorting
    buy_1_row = transactions.iloc[0] # 2023-01-10
    buy_2_blocking_row = transactions.iloc[1] # 2023-05-20 (Blocks 50 shares)
    sell_1_deferred_row = transactions.iloc[2] # 2023-06-15 (Loss deferred)
    buy_3_blocking_row = transactions.iloc[3] # 2023-07-20 (Blocks remaining 30 shares)
    sell_2_gain_row = transactions.iloc[4] # 2023-09-10 (Gain)

    result_buy_1 = calculator.calculate(buy_1_row)
    result_buy_2_blocking = calculator.calculate(buy_2_blocking_row)
    result_sell_1_deferred = calculator.calculate(sell_1_deferred_row)
    result_buy_3_blocking = calculator.calculate(buy_3_blocking_row)
    result_sell_2_loss = calculator.calculate(sell_2_gain_row)

    loss_per_share = -160.0 / 80.0 # = -2.0
    shares_sold = 80.0

    # Buy 2 blocks min(50, shares_sold) = 50 shares
    adj_buy2 = abs(loss_per_share * 50) # = 100.0
    remaining_shares_to_block = shares_sold - 50 # = 30

    # Buy 3 blocks min(60, remaining_shares_to_block) = 30 shares
    adj_buy3 = abs(loss_per_share * remaining_shares_to_block) # = 60.0

    assert result_buy_1 == (0.0, 0.0), "Initial Buy"
    # Buy 2 blocks 50 shares, gets proportional adjustment
    assert result_buy_2_blocking == (0.0, adj_buy2), f"Blocking Buy (before, 50 shares) gets adjustment {adj_buy2}"
    assert result_sell_1_deferred == (0.0, 0.0), "Deferred Sell"    
    # Buy 3 blocks remaining 30 shares, gets proportional adjustment
    assert result_buy_3_blocking == (0.0, adj_buy3), f"Blocking Buy (after, blocks 30 shares) gets adjustment {adj_buy3}"

    # Buy 2 cost = 450 + 100 (loss deferred)
    # Buy 3 cost = 600 + 60 (loss deferred)
    # Sell FIFO: 20 from buy 1 (200) + All from Buy 2 (550) + 10 from Buy 3 (660 / 6 = 110) = 860
    # Benefit = 8 * 70 - 860 = 300
    assert result_sell_2_loss == (-300.0, 0.0), "Loss Sell"


def test_irpf_returns_correct_tuples_for_non_sell_transactions(sample_transactions_base):
    """ Test Buy returns (0.0, 0.0), others return (None, None). """
    data = {
        'Date': ['2023-01-10', '2023-03-15'],
        'Transaction Type': ['buy', 'dividend'], # No 'sell'
        'Ticker': ['XYZ', 'XYZ'],
        'Quantity': [100, np.nan],
        'Price': [10, np.nan],
        'FIFO_Gain_Loss': [np.nan, np.nan]
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row = transactions.iloc[0]
    dividend_row = transactions.iloc[1]

    result_buy = calculator.calculate(buy_row)
    result_dividend = calculator.calculate(dividend_row)

    assert result_buy == (0.0, 0.0), "Buy row should return (0.0, 0.0)"
    assert result_dividend == (None, None), "Non buy/sell should return (None, None)"

def test_irpf_handles_nan_fifo_gain_loss_for_sell(sample_transactions_base):
    """ Test Sell with NaN FIFO returns (None, None). """
    data = {
        'Date': ['2023-01-10', '2023-03-15'],
        'Transaction Type': ['buy', 'sell'],
        'Ticker': ['XYZ', 'XYZ'],
        'Quantity': [100, 100],
        'Price': [10, 12],
        'FIFO_Gain_Loss': [np.nan, np.nan] # FIFO result is NaN
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row = transactions.iloc[0]
    sell_row = transactions.iloc[1]

    result_buy = calculator.calculate(buy_row)
    result_sell = calculator.calculate(sell_row)

    assert result_buy == (0.0, 0.0)
    # Should return (None, None) if it cannot determine gain/loss
    assert result_sell == (None, None) 

def test_irpf_edge_case_exactly_two_months_before_exclusive(sample_transactions_base):
    """ Test loss NOT deferred (exact 2mo before): Sell=(loss, 0.0), Buy=(0.0, 0.0). """
    data = {
        'Date': ['2023-04-15', '2023-06-15'], # Buy exactly 2 months before
        'Transaction Type': ['buy', 'sell'], 
        'Ticker': ['XYZ', 'XYZ'],
        'Quantity': [50, 100],
        'Price': [9, 8],
        'FIFO_Gain_Loss': [np.nan, -200.0] 
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)

    buy_row = transactions.iloc[0] 
    sell_row = transactions.iloc[1] 
    
    result_buy = calculator.calculate(buy_row)
    result_sell = calculator.calculate(sell_row)
    
    assert result_buy == (0.0, 0.0)
    # Assuming strict < > window, loss is NOT deferred
    assert result_sell == (-200.0, 0.0) 

def test_irpf_edge_case_exactly_two_months_after_exclusive(sample_transactions_base):
    """ Test loss NOT deferred (exact 2mo after): Sell=(loss, 0.0), Buy=(0.0, 0.0). """
    data = {
        'Date': ['2023-06-15', '2023-08-15'], # Buy exactly 2 months after
        'Transaction Type': ['sell', 'buy'], 
        'Ticker': ['XYZ', 'XYZ'],
        'Quantity': [100, 50],
        'Price': [8, 9],
        'FIFO_Gain_Loss': [-200.0, np.nan] 
    }
    calculator, transactions = create_calc_from_data(data, sample_transactions_base)
    
    sell_row = transactions.iloc[0]
    buy_row = transactions.iloc[1]

    result_sell = calculator.calculate(sell_row)
    result_buy = calculator.calculate(buy_row)

    # Assuming strict < > window, loss is NOT deferred
    assert result_sell == (-200.0, 0.0) 
    assert result_buy == (0.0, 0.0)


# --- Initialization Tests (remain largely the same) ---

def test_init_raises_error_if_df_missing(): # Removed fixture dependency
    """ Test ValueError if initialized with non-DataFrame. """
    with pytest.raises(ValueError, match="transactions_df must be a pandas DataFrame"):
        IrpfEarningsCalculator(transactions_df=None)
    with pytest.raises(ValueError, match="transactions_df must be a pandas DataFrame"):
        IrpfEarningsCalculator(transactions_df=[1, 2, 3])

def test_init_raises_error_if_missing_required_columns(sample_transactions_base):
    """ Test ValueError if DataFrame is missing essential columns. """
    valid_data = {
        'Date': [datetime(2023,1,1)], 'Transaction Type': ['buy'], 'Ticker': ['T'],
        'Quantity': [1], 'Price': [1], 'FIFO_Gain_Loss': [np.nan]
    }
    transactions_ok = pd.DataFrame(valid_data)
    transactions_ok['Date'] = pd.to_datetime(transactions_ok['Date'])

    # Test missing columns
    # Added 'Quantity', 'Price' as they might be needed by the new logic implicitly
    required_cols = ['Date', 'Transaction Type', 'Ticker', 'FIFO_Gain_Loss', 'Quantity', 'Price'] 
    for col in required_cols:
        if col not in transactions_ok.columns: continue # Skip if base fixture doesn't have it
        transactions_bad = transactions_ok.drop(columns=[col])
        with pytest.raises(ValueError, match=f"DataFrame must contain columns:.*{col}"):
             IrpfEarningsCalculator(transactions_df=transactions_bad)

def test_init_raises_error_if_date_column_not_convertible(sample_transactions_base):
    """ Test ValueError if the date column cannot be converted to datetime. """
    data = {
        'Date': ['2023-01-10', 'invalid-date-string'], # Add a non-convertible date
        'Transaction Type': ['buy', 'sell'], 'Ticker': ['XYZ', 'XYZ'],
        'Quantity': [100, 100], 'Price': [10, 12], 'FIFO_Gain_Loss': [np.nan, 200.0]
    }
    transactions = pd.concat([sample_transactions_base, pd.DataFrame(data)], ignore_index=True)

    with pytest.raises(ValueError, match="Could not convert date column 'Date' to datetime"):
        IrpfEarningsCalculator(transactions_df=transactions)

def test_init_converts_date_column_if_possible(sample_transactions_base):
    """ Test that the date column is converted to datetime during init if it's not already. """
    data = {
        'Date': ['2023-01-10', '2023-03-15'], # Dates as strings
        'Transaction Type': ['buy', 'sell'], 'Ticker': ['XYZ', 'XYZ'],
        'Quantity': [100, 100], 'Price': [10, 12], 'FIFO_Gain_Loss': [np.nan, 200.0]
    }
    transactions = pd.concat([sample_transactions_base, pd.DataFrame(data)], ignore_index=True)

    # Should not raise error if calculator handles conversion
    calculator = IrpfEarningsCalculator(transactions_df=transactions) 

    # Check dtype is converted in the internal DataFrame
    assert pd.api.types.is_datetime64_any_dtype(calculator.transactions_df['Date'])
    # Check internal buys_df if it exists and is accessible
    if hasattr(calculator, 'buys_df') and isinstance(calculator.buys_df, pd.DataFrame):
         assert pd.api.types.is_datetime64_any_dtype(calculator.buys_df['Date'])