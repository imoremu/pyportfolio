# pyportfolio/calculators/irpf_earnings_calculator.py

import pandas as pd
from pandas.tseries.offsets import DateOffset
from typing import Optional, Tuple, List
import logging

# Import the correct base class
from .base_calculator import BaseTableCalculator

# --- Constants ---
from pyportfolio.columns import (
    SHARE_PRICE,
    SHARES,
    TRANSACTION_TYPE,
    TICKER,
    DATE
)

# Transaction Types
_TRANSACTION_TYPE_BUY = 'buy'
_TRANSACTION_TYPE_SELL = 'sell'

# Result Column Names
RESULT_TAXABLE_GAIN_LOSS = 'IRPF - Ganancia / Pérdida Imputable'
RESULT_DEFERRED_ADJUSTMENT = 'IRPF - Ajuste Diferido'
_RESULT_COLUMNS = [RESULT_TAXABLE_GAIN_LOSS, RESULT_DEFERRED_ADJUSTMENT]

# Internal State Columns (prefix _ indicates internal use)
_INTERNAL_AVAILABLE_SHARES = '_IrpfAvailableShares'
_INTERNAL_DEFERRED_ADJUSTMENT_STATE = '_IrpfDeferredAdjustmentState'
_INTERNAL_ADJUSTED_COST_PER_SHARE = '_IrpfAdjustedCostPerShare'
_INTERNAL_BLOCKING_CAPACITY_REMAINING = '_IrpfBlockingCapacityRemaining'

# --- Logger ---
logger = logging.getLogger(__name__)

class IrpfEarningsCalculator(BaseTableCalculator):
    """
    Calculates the capital gain or loss for Spanish IRPF purposes as a table-wise operation.

    Applies the "two-month rule" (regla de los dos meses) and adjusts
    cost basis for deferred losses, using a "consuming capacity" logic for blocking buys.
    Performs its own adjusted FIFO calculation internally.
    """

    def __init__(self):
        """
        Initializes the table-wise calculator. No pre-calculation is done here.
        """
        super().__init__()
        logger.debug("IrpfEarningsCalculator initialized (Table-wise).")

    def _validate_input_dataframe(self, df: pd.DataFrame):
        """ Performs validation checks on the input DataFrame. """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        required_columns = [DATE, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame must contain columns: {required_columns}. Missing: {missing_cols}")
        if not pd.api.types.is_datetime64_any_dtype(df[DATE]):
            try:
                # Check conversion without modifying original df passed to function
                pd.to_datetime(df[DATE])
            except Exception as e:
                raise ValueError(f"Could not convert date column '{DATE}' to datetime: {e}")

    def calculate_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs the IRPF calculation on the entire DataFrame.

        Args:
            df: The transaction DataFrame provided by the TransactionManager.

        Returns:
            A new DataFrame containing only the calculated result columns
            (RESULT_TAXABLE_GAIN_LOSS, RESULT_DEFERRED_ADJUSTMENT)
            with the same index as the input df.
        """
        logger.info("Starting IRPF calculate_table...")
        self._validate_input_dataframe(df)

        # Create an internal copy to avoid modifying the original DataFrame from the manager
        internal_df = df.copy(deep=True)

        if not pd.api.types.is_datetime64_any_dtype(internal_df[DATE]):
             internal_df[DATE] = pd.to_datetime(internal_df[DATE])
        internal_df = internal_df.sort_values(
            by=[DATE, TRANSACTION_TYPE], ascending=[True, True]
        ).reset_index(drop=True) # Reset index for 0..N iteration

        internal_df[_INTERNAL_AVAILABLE_SHARES] = 0.0
        internal_df[_INTERNAL_DEFERRED_ADJUSTMENT_STATE] = 0.0
        internal_df[_INTERNAL_ADJUSTED_COST_PER_SHARE] = 0.0
        internal_df[_INTERNAL_BLOCKING_CAPACITY_REMAINING] = 0.0

        buy_mask = internal_df[TRANSACTION_TYPE].str.lower() == _TRANSACTION_TYPE_BUY
        internal_df.loc[buy_mask, _INTERNAL_AVAILABLE_SHARES] = internal_df.loc[buy_mask, SHARES]
        internal_df.loc[buy_mask, _INTERNAL_ADJUSTED_COST_PER_SHARE] = internal_df.loc[buy_mask, SHARE_PRICE]
        internal_df.loc[buy_mask, _INTERNAL_BLOCKING_CAPACITY_REMAINING] = internal_df.loc[buy_mask, SHARES]

        internal_df[RESULT_TAXABLE_GAIN_LOSS] = pd.NA
        internal_df[RESULT_DEFERRED_ADJUSTMENT] = pd.NA
        internal_df[RESULT_TAXABLE_GAIN_LOSS] = internal_df[RESULT_TAXABLE_GAIN_LOSS].astype('Float64')
        internal_df[RESULT_DEFERRED_ADJUSTMENT] = internal_df[RESULT_DEFERRED_ADJUSTMENT].astype('Float64')

        logger.info("Starting IRPF chronological processing loop within calculate_table...")
        for index, row in internal_df.iterrows():
            transaction_type = str(row.get(TRANSACTION_TYPE, '')).lower()

            try:
                if transaction_type == _TRANSACTION_TYPE_SELL:
                    result_tuple = self._process_sell_and_update_buys(row, index, internal_df)
                    if result_tuple is not None:
                        internal_df.loc[index, RESULT_TAXABLE_GAIN_LOSS] = result_tuple[0]
                        internal_df.loc[index, RESULT_DEFERRED_ADJUSTMENT] = result_tuple[1] # Should be 0.0 for sells

                elif transaction_type == _TRANSACTION_TYPE_BUY:
                    internal_df.loc[index, RESULT_TAXABLE_GAIN_LOSS] = 0.0
                    internal_df.loc[index, RESULT_DEFERRED_ADJUSTMENT] = 0.0 # Placeholder
                    pass
                else:
                    internal_df.loc[index, RESULT_TAXABLE_GAIN_LOSS] = None
                    internal_df.loc[index, RESULT_DEFERRED_ADJUSTMENT] = None

            except Exception as e:
                 logger.error(f"Error processing row {index} during IRPF calculation: {e}. Row: {row.to_dict()}", exc_info=True)
                 internal_df.loc[index, RESULT_TAXABLE_GAIN_LOSS] = None
                 internal_df.loc[index, RESULT_DEFERRED_ADJUSTMENT] = None

        # Transfer the accumulated adjustment state to the result column for buys
        logger.info("Populating final deferred adjustment results for buy transactions...")
        buy_mask_final = internal_df[TRANSACTION_TYPE].str.lower() == _TRANSACTION_TYPE_BUY
        internal_df.loc[buy_mask_final, RESULT_DEFERRED_ADJUSTMENT] = internal_df.loc[buy_mask_final, _INTERNAL_DEFERRED_ADJUSTMENT_STATE]

        logger.info("Finished IRPF calculate_table.")

        # Ensure the index matches the original input DataFrame 'df'
        internal_df.index = df.index

        results_df = internal_df[_RESULT_COLUMNS].copy()
        return results_df


    def _process_sell_and_update_buys(self, sell_row: pd.Series, sell_index: int, internal_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
         """
         Processes a sell transaction: calculates adjusted FIFO gain/loss,
         handles loss deferral by finding blockers and updating their state
         (capacity, adjustment, cost basis) directly in internal_df.
         Returns the (allowable_loss, 0.0) tuple for the sell row.

         Args:
             sell_row: The row representing the sell transaction.
             sell_index: The index of the sell row in internal_df (after sorting).
             internal_df: The DataFrame holding the current calculation state.
         """
         ticker = sell_row.get(TICKER)
         shares_to_sell = sell_row.get(SHARES)
         sell_price = sell_row.get(SHARE_PRICE)
         sell_date = sell_row.get(DATE)

         if pd.isna(ticker) or pd.isna(shares_to_sell) or shares_to_sell <= 0 or pd.isna(sell_price) or pd.isna(sell_date):
             logger.warning(f"Sell row {sell_index} missing essential data. Cannot calculate.")
             return (None, None)

         # Find previous buy lots with available shares *in the current state*
         previous_buys = internal_df[
             (internal_df.index < sell_index) &
             (internal_df[TICKER] == ticker) &
             (internal_df[TRANSACTION_TYPE].str.lower() == _TRANSACTION_TYPE_BUY) &
             (internal_df[_INTERNAL_AVAILABLE_SHARES] > 1e-9) # Use tolerance for float comparison
         ]

         total_available_before = previous_buys[_INTERNAL_AVAILABLE_SHARES].sum()

         if shares_to_sell > total_available_before + 1e-9: # Use tolerance
             logger.error(f"Overselling detected for sell {sell_index}. Available: {total_available_before}, Selling: {shares_to_sell}")
             return (None, None)

         total_adjusted_cost = 0.0
         remaining_to_sell = shares_to_sell

         # Consume shares from previous buys in FIFO order using adjusted cost
         # Iterate over index to safely use .at for updates
         for buy_idx in previous_buys.index:
             available_in_lot = internal_df.at[buy_idx, _INTERNAL_AVAILABLE_SHARES]
             adjusted_cost_per_share = internal_df.at[buy_idx, _INTERNAL_ADJUSTED_COST_PER_SHARE]

             if pd.isna(adjusted_cost_per_share):
                  logger.error(f"NaN adjusted cost per share in buy lot {buy_idx} when processing sell {sell_index}.")
                  return (None, None)

             consume_from_lot = min(available_in_lot, remaining_to_sell)
             total_adjusted_cost += consume_from_lot * adjusted_cost_per_share
             internal_df.at[buy_idx, _INTERNAL_AVAILABLE_SHARES] -= consume_from_lot
             remaining_to_sell -= consume_from_lot

             if remaining_to_sell < 1e-9: break

         sell_proceeds = sell_price * shares_to_sell
         adjusted_gain_loss = sell_proceeds - total_adjusted_cost

         if adjusted_gain_loss < -1e-9: # Check if it's a loss (with tolerance)
             loss_per_share = adjusted_gain_loss / shares_to_sell
             repurchases = self._find_repurchases(ticker, sell_date, internal_df)

             if repurchases.empty:
                 logger.debug(f"Sell {sell_index}: No repurchases found. Allowable loss = {adjusted_gain_loss:.2f}")
                 return (adjusted_gain_loss, 0.0)
             else:
                 sorted_repurchases = repurchases.sort_values(by=DATE)
                 actual_shares_blocked_for_this_sell = 0.0
                 remaining_shares_from_sell_to_block = shares_to_sell
                 logger.debug(f"Sell {sell_index}: Loss={adjusted_gain_loss:.2f}. Found {len(sorted_repurchases)} potential blockers. Need to block {remaining_shares_from_sell_to_block} shares.")

                 # Iterate through potential blockers to allocate blocking & update their state
                 for internal_blocker_index in sorted_repurchases.index:
                     logger.debug(f"--- Sell {sell_index}: Checking potential blocker index {internal_blocker_index} Date: {internal_df.at[internal_blocker_index, DATE]} ---")

                     blocker_total_qty = internal_df.at[internal_blocker_index, SHARES]
                     blocker_remaining_capacity = internal_df.at[internal_blocker_index, _INTERNAL_BLOCKING_CAPACITY_REMAINING]
                     logger.debug(f"Sell {sell_index}: Blocker {internal_blocker_index} - RemainingCapacity={blocker_remaining_capacity}")

                     if blocker_remaining_capacity < 1e-9:
                         logger.debug(f"Sell {sell_index}: Blocker {internal_blocker_index} has no capacity left.")
                         continue

                     shares_this_blocker_blocks_now = min(remaining_shares_from_sell_to_block, blocker_remaining_capacity)
                     logger.debug(f"Sell {sell_index}: Blocker {internal_blocker_index} will block {shares_this_blocker_blocks_now} shares now.")

                     if shares_this_blocker_blocks_now > 1e-9:
                         new_remaining_capacity = blocker_remaining_capacity - shares_this_blocker_blocks_now
                         internal_df.at[internal_blocker_index, _INTERNAL_BLOCKING_CAPACITY_REMAINING] = new_remaining_capacity

                         loss_adj_for_this_blocker = shares_this_blocker_blocks_now * abs(loss_per_share)
                         current_adj = internal_df.at[internal_blocker_index, _INTERNAL_DEFERRED_ADJUSTMENT_STATE]
                         new_total_adj = current_adj + loss_adj_for_this_blocker
                         internal_df.at[internal_blocker_index, _INTERNAL_DEFERRED_ADJUSTMENT_STATE] = new_total_adj

                         original_price = internal_df.at[internal_blocker_index, SHARE_PRICE]
                         if blocker_total_qty > 1e-9:
                              new_adjusted_cost = original_price + (new_total_adj / blocker_total_qty)
                              internal_df.at[internal_blocker_index, _INTERNAL_ADJUSTED_COST_PER_SHARE] = new_adjusted_cost
                              logger.debug(f"Sell {sell_index}: Updated Blocker {internal_blocker_index} - NewRemainingCapacity={new_remaining_capacity:.2f}, NewTotalAdj={new_total_adj:.2f}, NewAdjCost={new_adjusted_cost:.4f}")
                         else:
                              internal_df.at[internal_blocker_index, _INTERNAL_ADJUSTED_COST_PER_SHARE] = original_price
                              logger.debug(f"Sell {sell_index}: Updated Blocker {internal_blocker_index} - NewRemainingCapacity={new_remaining_capacity:.2f}, NewTotalAdj={new_total_adj:.2f}, Cost N/A (zero qty)")

                         actual_shares_blocked_for_this_sell += shares_this_blocker_blocks_now
                         remaining_shares_from_sell_to_block -= shares_this_blocker_blocks_now

                     if remaining_shares_from_sell_to_block < 1e-9:
                         logger.debug(f"Sell {sell_index}: All shares blocked. Breaking blocker loop.")
                         break

                 total_deferred_loss_amount = actual_shares_blocked_for_this_sell * abs(loss_per_share)
                 allowable_loss = adjusted_gain_loss + total_deferred_loss_amount
                 allowable_loss = min(allowable_loss, 0.0) # Ensure allowable loss is not positive
                 logger.debug(f"Sell {sell_index}: Finished blocking. ActualTotalBlocked={actual_shares_blocked_for_this_sell:.2f}, Final Allowable={allowable_loss:.2f}")
                 return (allowable_loss, 0.0)

         else:
             adjusted_gain_loss = max(adjusted_gain_loss, 0.0) # Ensure not slightly negative
             logger.debug(f"Sell {sell_index}: Gain calculated = {adjusted_gain_loss:.2f}")
             return (adjusted_gain_loss, 0.0)

    def _find_repurchases(self, ticker: str, current_date: pd.Timestamp, lookup_df: pd.DataFrame) -> pd.DataFrame:
        """
        Finds buy transactions (repurchases) of the same ticker within the strict
        +/- 2 month window of a given date in the lookup DataFrame.

        Args:
            ticker: The ticker symbol to match.
            current_date: The date of the sell transaction.
            lookup_df: The DataFrame to search within (should be the internal_df).
        """
        two_months_before = current_date - DateOffset(months=2)
        two_months_after = current_date + DateOffset(months=2)

        if not pd.api.types.is_datetime64_any_dtype(lookup_df[DATE]):
             lookup_df_dates = pd.to_datetime(lookup_df[DATE])
        else:
             lookup_df_dates = lookup_df[DATE]

        ticker_match = lookup_df[TICKER] == ticker
        is_buy = lookup_df[TRANSACTION_TYPE].str.lower() == _TRANSACTION_TYPE_BUY
        # Strict inequality for the window check based on common interpretation
        after_window_start = lookup_df_dates > two_months_before
        before_window_end = lookup_df_dates < two_months_after
        valid_quantity = ~pd.isna(lookup_df[SHARES]) & (lookup_df[SHARES] > 1e-9) # Use tolerance

        repurchases = lookup_df[
            ticker_match & is_buy & after_window_start & before_window_end & valid_quantity
        ]
        return repurchases
