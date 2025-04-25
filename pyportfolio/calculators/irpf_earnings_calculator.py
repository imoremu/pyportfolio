# pyportfolio/calculators/irpf_earnings_calculator.py

import pandas as pd
from pandas.tseries.offsets import DateOffset # More robust for month calculations
from typing import Any, Optional
import numpy as np

from .base_calculator import BaseCalculator

class IrpfEarningsCalculator(BaseCalculator):
    """
    Calculates the capital gain or loss for Spanish IRPF purposes,
    applying the "two-month rule" (regla de los dos meses) to defer losses
    if identical securities are repurchased within 2 months before or after the sale.

    Assumes a prior calculation (e.g., FIFOCalculator) has populated a
    'FIFO_Gain_Loss' column.
    """

    def __init__(self, transactions_df: pd.DataFrame, 
                 fifo_column: str = 'FIFO_Gain_Loss',
                 date_column: str = 'Date',
                 type_column: str = 'Transaction Type',
                 ticker_column: str = 'Ticker'):
        """
        Initializes the calculator with the full transaction DataFrame.

        Args:
            transactions_df: The complete DataFrame with all transactions.
            fifo_column: Name of the column containing pre-calculated FIFO gains/losses.
            date_column: Name of the column containing transaction dates (datetime objects).
            type_column: Name of the column containing the transaction type ('buy', 'sell', etc.).
            ticker_column: Name of the column containing the security ticker/symbol.
        """
        if not isinstance(transactions_df, pd.DataFrame):
            raise ValueError("transactions_df must be a pandas DataFrame")
        
        required_columns = [fifo_column, date_column, type_column, ticker_column]
        if not all(col in transactions_df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
            
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(transactions_df[date_column]):
             try:
                 transactions_df[date_column] = pd.to_datetime(transactions_df[date_column])
             except Exception as e:
                 raise ValueError(f"Could not convert date column '{date_column}' to datetime: {e}")

        self.transactions_df = transactions_df
        self.fifo_column = fifo_column
        self.date_column = date_column
        self.type_column = type_column
        self.ticker_column = ticker_column
        
        # Pre-filter for faster lookups later (optional optimization)
        self.buys_df = self.transactions_df[
            self.transactions_df[self.type_column].str.lower() == 'buy'
        ].copy() # Use .copy() to avoid SettingWithCopyWarning if modifying later


    def calculate(self, row: pd.Series) -> Optional[float]:
        """
        Calculates the IRPF gain/loss for a given transaction row.

        Args:
            row: A pandas Series representing a single transaction row.

        Returns:
            The taxable gain (positive float), allowable loss (negative float),
            0.0 if the loss is deferred due to the two-month rule,
            or None if the transaction is not a relevant sale.
        """
        transaction_type = str(row.get(self.type_column, '')).lower()
        
        # Only calculate for sales
        if transaction_type != 'sell':
            return None

        fifo_gain_loss = row.get(self.fifo_column)

        # If FIFO calculation resulted in NaN or None, cannot proceed
        if pd.isna(fifo_gain_loss):
            # Or raise an error, depending on desired behaviour
            return None 
            
        # If it's a gain, it's generally taxable directly
        if fifo_gain_loss >= 0:
            return float(fifo_gain_loss)

        # --- It's a Loss - Apply the Two-Month Rule ---
        loss_amount = float(fifo_gain_loss)
        sale_date = row[self.date_column]
        ticker = row[self.ticker_column]

        # Define the two-month window before and after the sale
        # Note: Spanish rule is often interpreted as exactly 2 months, 
        # calendar-wise, not 60 days. DateOffset handles this better.
        two_months_before = sale_date - DateOffset(months=2)
        two_months_after = sale_date + DateOffset(months=2)

        # Look for repurchases of the *same ticker* within the window
        repurchases_in_window = self.buys_df[
            (self.buys_df[self.ticker_column] == ticker) &
            (self.buys_df[self.date_column] > two_months_before) & # Exclusive of start date? Check legislation nuances
            (self.buys_df[self.date_column] < two_months_after)   # Exclusive of end date? Check legislation nuances
        ]

        if not repurchases_in_window.empty:
            # Repurchase found within the window - Loss is deferred
            # We return 0.0 to indicate no *currently* allowable loss for IRPF.
            # The actual loss value might be needed later when the repurchased shares are sold.
            return 0.0 
        else:
            # No repurchase found - Loss is allowable for IRPF in this period
            return loss_amount


# --- Example Usage (Conceptual - how it might integrate) ---
# 
# Assume 'transactions' is your DataFrame after FIFOCalculator ran
# 
# from pyportfolio.transaction_manager import TransactionManager
# 
# # 1. Initial DataFrame (example)
# transactions = pd.DataFrame({
#     'Date': pd.to_datetime(['2023-01-15', '2023-04-01', '2023-05-01', '2023-08-01']),
#     'Transaction Type': ['buy', 'sell', 'buy', 'sell'],
#     'Ticker': ['TEF', 'TEF', 'TEF', 'TEF'],
#     'Quantity': [100, 50, 30, 80],
#     'Price': [10, 12, 9, 8],
#     'FIFO_Gain_Loss': [np.nan, 100.0, np.nan, -120.0] # Example FIFO results
# })
# 
# # 2. Initialize TransactionManager
# manager = TransactionManager(transactions)
# 
# # 3. Initialize and Register the IRPF Calculator
# #    Crucially, pass the *entire* DataFrame to the calculator's constructor
# irpf_calculator = IrpfEarningsCalculator(transactions_df=manager.transactions) 
# manager.register_calculation("IRPF_Gain_Loss", irpf_calculator, dtype='float')
# 
# # 4. Process
# manager.process_all()
# 
# # 5. Inspect Results
# print(manager.transactions)
# 
# # Expected Output Snippet (conceptual):
# # Date        Transaction Type Ticker ... FIFO_Gain_Loss  IRPF_Gain_Loss
# # 2023-01-15       buy           TEF  ...        NaN             NaN   # No calculation for buys
# # 2023-04-01       sell          TEF  ...       100.0           100.0   # Gain is realized
# # 2023-05-01       buy           TEF  ...        NaN             NaN   # No calculation for buys
# # 2023-08-01       sell          TEF  ...      -120.0             0.0   # Loss deferred (buy on 2023-05-01 is within 2 months before)

