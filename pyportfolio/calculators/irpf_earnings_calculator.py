import pandas as pd
from pandas.tseries.offsets import DateOffset
from typing import Optional, Tuple
import logging

from .base_calculator import BaseCalculator

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

# Internal State Columns
_INTERNAL_AVAILABLE_SHARES = '_IrpfAvailableShares' # For FIFO tracking within this calculator
_INTERNAL_DEFERRED_ADJUSTMENT = '_IrpfDeferredAdjustment' # Accumulated monetary adjustment on Buys
_INTERNAL_ADJUSTED_COST_PER_SHARE = '_IrpfAdjustedCostPerShare' # Adjusted cost basis for FIFO
_INTERNAL_BLOCKING_CAPACITY_REMAINING = '_IrpfBlockingCapacityRemaining' # Tracks remaining shares a buy can block

# Result Columns stored in the internal DataFrame after calculation
_RESULT_TAXABLE_GAIN_LOSS = '_IrpfTaxableGainLoss'
_RESULT_DEFERRED_ADJUSTMENT = '_IrpfDeferredAdjustmentResult' # Final adjustment result for buys

# --- Logger ---
logger = logging.getLogger(__name__)

class IrpfEarningsCalculator(BaseCalculator):
    """
    Calculates the capital gain or loss for Spanish IRPF purposes,
    applying the "two-month rule" (regla de los dos meses) and adjusting
    cost basis for deferred losses, using a "consuming capacity" logic for blocking buys.

    This calculator uses a chronological pass during initialization to pre-calculate
    results. Sell transactions with losses find relevant blocking buy transactions
    (past or future) and update their state (deferred adjustment, adjusted cost,
    and remaining blocking capacity) directly.

    It performs its own adjusted FIFO calculation internally.
    """

    def __init__(self, transactions_df: pd.DataFrame):
        """
        Initializes the calculator, validates input, sets up internal state,
        and runs the pre-calculation loop.

        Args:
            transactions_df: The complete DataFrame with all transactions.
                             Must contain columns defined in pyportfolio.columns
                             (DATE, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE).
        """
        self._validate_input_dataframe(transactions_df)

        internal_df = transactions_df.copy(deep=True)

        # Ensure date column is datetime for reliable sorting and comparisons
        if not pd.api.types.is_datetime64_any_dtype(internal_df[DATE]):
             internal_df[DATE] = pd.to_datetime(internal_df[DATE])

        # Sort chronologically - essential for stateful calculations
        internal_df = internal_df.sort_values(
            by=[DATE, TRANSACTION_TYPE], ascending=[True, True]
        ).reset_index(drop=True)

        # --- Initialize State Columns ---
        internal_df[_INTERNAL_AVAILABLE_SHARES] = 0.0
        internal_df[_INTERNAL_DEFERRED_ADJUSTMENT] = 0.0
        internal_df[_INTERNAL_ADJUSTED_COST_PER_SHARE] = 0.0
        internal_df[_INTERNAL_BLOCKING_CAPACITY_REMAINING] = 0.0 # Default 0

        buy_mask = internal_df[TRANSACTION_TYPE].str.lower() == _TRANSACTION_TYPE_BUY
        # Initialize state for buy transactions
        internal_df.loc[buy_mask, _INTERNAL_AVAILABLE_SHARES] = internal_df.loc[buy_mask, SHARES]
        internal_df.loc[buy_mask, _INTERNAL_ADJUSTED_COST_PER_SHARE] = internal_df.loc[buy_mask, SHARE_PRICE]
        internal_df.loc[buy_mask, _INTERNAL_BLOCKING_CAPACITY_REMAINING] = internal_df.loc[buy_mask, SHARES]

        # --- Initialize Result Columns ---
        internal_df[_RESULT_TAXABLE_GAIN_LOSS] = pd.NA
        internal_df[_RESULT_DEFERRED_ADJUSTMENT] = pd.NA
        internal_df[_RESULT_TAXABLE_GAIN_LOSS] = internal_df[_RESULT_TAXABLE_GAIN_LOSS].astype('Float64')
        internal_df[_RESULT_DEFERRED_ADJUSTMENT] = internal_df[_RESULT_DEFERRED_ADJUSTMENT].astype('Float64')

        # Store original df for finding blockers across full timeline, independent of internal state changes
        self.original_transactions_df = transactions_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.original_transactions_df[DATE]):
             self.original_transactions_df[DATE] = pd.to_datetime(self.original_transactions_df[DATE])

        # --- Main Chronological Processing Loop ---
        logger.info("Starting IRPF pre-calculation loop (Sell-Centric Updates)...")
        
        for index, row in internal_df.iterrows():
            transaction_type = str(row.get(TRANSACTION_TYPE, '')).lower()

            try:
                if transaction_type == _TRANSACTION_TYPE_SELL:
                    # Process sell, calculate its result, and update relevant buy states
                    result_tuple = self._process_sell_and_update_buys(row, index, internal_df)
                    if result_tuple is not None:
                        internal_df.loc[index, _RESULT_TAXABLE_GAIN_LOSS] = result_tuple[0]
                        internal_df.loc[index, _RESULT_DEFERRED_ADJUSTMENT] = result_tuple[1] # Should be 0.0 for sells

                elif transaction_type == _TRANSACTION_TYPE_BUY:
                    # Set placeholder results for buys; final adjustment result populated after loop
                    internal_df.loc[index, _RESULT_TAXABLE_GAIN_LOSS] = 0.0
                    internal_df.loc[index, _RESULT_DEFERRED_ADJUSTMENT] = 0.0 # Placeholder
                    # Buy state (_INTERNAL_DEFERRED_ADJUSTMENT, _INTERNAL_ADJUSTED_COST_PER_SHARE,
                    # _INTERNAL_BLOCKING_CAPACITY_REMAINING) is updated by the _process_sell_and_update_buys method
                    pass

            except Exception as e:
                 logger.error(f"Error processing row {index} during IRPF init: {e}. Row: {row.to_dict()}", exc_info=True)
                 # Mark results as NA/None on error for this row
                 internal_df.loc[index, _RESULT_TAXABLE_GAIN_LOSS] = None
                 internal_df.loc[index, _RESULT_DEFERRED_ADJUSTMENT] = None

        # --- Final Pass to Populate Buy Results ---
        # Transfer the final accumulated adjustment state to the result column for buys
        logger.info("Populating final results for buy transactions...")
        buy_mask_final = internal_df[TRANSACTION_TYPE].str.lower() == _TRANSACTION_TYPE_BUY
        internal_df.loc[buy_mask_final, _RESULT_DEFERRED_ADJUSTMENT] = internal_df.loc[buy_mask_final, _INTERNAL_DEFERRED_ADJUSTMENT]

        logger.debug(f"Finished IRPF pre-calculation loop.")
        self.internal_transactions = internal_df # Store the processed DataFrame

    # --- Validation Helper ---
    def _validate_input_dataframe(self, df: pd.DataFrame):
        """ Performs validation checks on the input DataFrame. """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("transactions_df must be a pandas DataFrame")
        # Check for essential columns needed by the calculator
        required_columns = [DATE, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame must contain columns: {required_columns}. Missing: {missing_cols}")
        # Check if date column is convertible to prevent errors later
        if not pd.api.types.is_datetime64_any_dtype(df[DATE]):
            try:
                pd.to_datetime(df[DATE])
            except Exception as e:
                raise ValueError(f"Could not convert date column '{DATE}' to datetime: {e}")

    # --- Calculate Method (Lookup) ---
    def calculate(self, row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        """
        Retrieves the pre-calculated IRPF gain/loss tuple for the given row
        by matching it against the internally processed transactions.

        Args:
            row: A pandas Series representing a single transaction row from the
                 *original* DataFrame used during initialization.

        Returns:
            A tuple `(taxable_gain_loss, deferred_loss_adjustment)`.
            Returns (None, None) if the row cannot be matched or an error occurs.
        """
        try:
            # Attempt to match the input row based on key identifying fields
            match_cols = [DATE, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE]
            conditions = []
            for col in match_cols:
                row_val = row.get(col)
                if pd.isna(row_val):
                    conditions.append(self.internal_transactions[col].isna())
                else:
                    # Use tolerance for float comparison
                    if pd.api.types.is_float_dtype(row_val) and col in [SHARES, SHARE_PRICE]:
                         conditions.append(abs(self.internal_transactions[col] - row_val) < 1e-9)
                    else:
                         conditions.append(self.internal_transactions[col] == row_val)

            matches = self.internal_transactions[pd.concat(conditions, axis=1).all(axis=1)]

            # Fallback matching if primary fails (e.g., for dividends missing quantity/price)
            if len(matches) == 0:
                match_cols_basic = [DATE, TRANSACTION_TYPE, TICKER]
                conditions_basic = []
                for col in match_cols_basic:
                     row_val = row.get(col)
                     if pd.isna(row_val):
                          conditions_basic.append(self.internal_transactions[col].isna())
                     else:
                          conditions_basic.append(self.internal_transactions[col] == row_val)
                matches = self.internal_transactions[pd.concat(conditions_basic, axis=1).all(axis=1)]
                if len(matches) == 0:
                    logger.warning(f"Lookup failed in calculate for row: {row.to_dict()}. Returning (None, None).")
                    return (None, None)

            # Handle potential duplicate matches
            if len(matches) > 1:
                 # Prefer matching by original index if available and unique within matches
                 if row.name in matches.index:
                      internal_index = row.name
                 else:
                      logger.warning(f"Multiple matches found in calculate for row: {row.to_dict()}. Using first match index {matches.index[0]}.")
                      internal_index = matches.index[0]
            else:
                 internal_index = matches.index[0]

            # Retrieve pre-calculated results from the matched internal row
            taxable_gain_loss = self.internal_transactions.loc[internal_index, _RESULT_TAXABLE_GAIN_LOSS]
            deferred_adjustment = self.internal_transactions.loc[internal_index, _RESULT_DEFERRED_ADJUSTMENT]

            # Convert pandas NA/NaN to Python None for consistent return type
            taxable_gain_loss = None if pd.isna(taxable_gain_loss) else float(taxable_gain_loss)
            deferred_adjustment = None if pd.isna(deferred_adjustment) else float(deferred_adjustment)
            return (taxable_gain_loss, deferred_adjustment)

        except Exception as e:
            logger.error(f"Error during lookup in calculate for row {row.name if hasattr(row, 'name') else 'UNKNOWN'}: {e}. Row Data: {row.to_dict()}", exc_info=True)
            return (None, None)

    # --- Main Sell Processing Logic ---
    def _process_sell_and_update_buys(self, sell_row: pd.Series, sell_index: int, internal_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
         """
         Processes a sell transaction: calculates adjusted FIFO gain/loss,
         handles loss deferral by finding blockers and updating their state
         (capacity, adjustment, cost basis) directly in internal_df.
         Returns the (allowable_loss, 0.0) tuple for the sell row.
         """
         ticker = sell_row.get(TICKER)
         shares_to_sell = sell_row.get(SHARES)
         sell_price = sell_row.get(SHARE_PRICE)
         sell_date = sell_row.get(DATE)

         # --- Basic Validation ---
         if pd.isna(ticker) or pd.isna(shares_to_sell) or shares_to_sell <= 0 or pd.isna(sell_price) or pd.isna(sell_date):
             logger.warning(f"Sell row {sell_index} missing essential data. Cannot calculate.")
             return (None, None)

         # --- Perform Adjusted FIFO Calculation ---
         # Find previous buy lots with available shares *in the current state*
         previous_buys = internal_df[
             (internal_df.index < sell_index) &
             (internal_df[TICKER] == ticker) &
             (internal_df[TRANSACTION_TYPE].str.lower() == _TRANSACTION_TYPE_BUY) &
             (internal_df[_INTERNAL_AVAILABLE_SHARES] > 1e-9)
         ].copy() # Using copy for safety during iteration, though .at is used for updates
         total_available_before = previous_buys[_INTERNAL_AVAILABLE_SHARES].sum()

         # Check for overselling before proceeding
         if shares_to_sell > total_available_before + 1e-9:
             logger.error(f"Overselling detected for sell {sell_index}.")
             return (None, None)

         total_adjusted_cost = 0.0
         remaining_to_sell = shares_to_sell

         # Consume shares from previous buys in FIFO order using adjusted cost
         for buy_idx, buy_row in previous_buys.iterrows():
             available_in_lot = buy_row[_INTERNAL_AVAILABLE_SHARES]
             adjusted_cost_per_share = buy_row[_INTERNAL_ADJUSTED_COST_PER_SHARE]
             if pd.isna(adjusted_cost_per_share):
                  logger.error(f"NaN adjusted cost per share in buy lot {buy_idx}.")
                  return (None, None) # Cannot proceed without valid cost

             consume_from_lot = min(available_in_lot, remaining_to_sell)
             total_adjusted_cost += consume_from_lot * adjusted_cost_per_share
             # Update available shares state immediately using .at for efficiency
             internal_df.at[buy_idx, _INTERNAL_AVAILABLE_SHARES] -= consume_from_lot
             remaining_to_sell -= consume_from_lot

             if remaining_to_sell < 1e-9: break # Exit loop once all needed shares are consumed

         # --- Calculate Gain/Loss ---
         sell_proceeds = sell_price * shares_to_sell
         adjusted_gain_loss = sell_proceeds - total_adjusted_cost

         # --- Handle Loss Deferral and Update Buys ---
         if adjusted_gain_loss < -1e-9: # Check if it's a loss (with tolerance)
             loss_per_share = adjusted_gain_loss / shares_to_sell
             # Find potential repurchases (blockers) using the original DataFrame
             repurchases = self._find_repurchases(ticker, sell_date, self.original_transactions_df)

             if repurchases.empty:
                 # No repurchases found, loss is fully allowable
                 logger.debug(f"Sell {sell_index}: No repurchases found. Allowable loss = {adjusted_gain_loss:.2f}")
                 return (adjusted_gain_loss, 0.0)
             else:
                 # Sort potential blockers chronologically to apply capacity limits correctly
                 sorted_repurchases = repurchases.sort_values(by=DATE)
                 actual_shares_blocked_for_this_sell = 0.0
                 remaining_shares_from_sell_to_block = shares_to_sell
                 logger.debug(f"Sell {sell_index}: Loss={adjusted_gain_loss:.2f}. Found {len(sorted_repurchases)} potential blockers. Need to block {remaining_shares_from_sell_to_block} shares.")

                 # Iterate through potential blockers to allocate blocking & update their state
                 for blocker_orig_idx, blocker_row_orig in sorted_repurchases.iterrows():
                     logger.debug(f"--- Sell {sell_index}: Checking potential blocker original index {blocker_orig_idx} Date: {blocker_row_orig[DATE]} ---")

                     # Find the corresponding blocker row in internal_df state using robust matching
                     try:
                         match_cols = [DATE, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE]
                         conditions = []
                         for col in match_cols:
                             row_val = blocker_row_orig.get(col)
                             if pd.isna(row_val): conditions.append(internal_df[col].isna())
                             else:
                                 if pd.api.types.is_float_dtype(row_val) and col in [SHARES, SHARE_PRICE]:
                                      conditions.append(abs(internal_df[col] - row_val) < 1e-9)
                                 else: conditions.append(internal_df[col] == row_val)
                         matches = internal_df[pd.concat(conditions, axis=1).all(axis=1)]
                         if len(matches) == 0:
                              logger.warning(f"Sell {sell_index}: Could not find internal state for blocker {blocker_orig_idx}. Skipping.")
                              continue
                         internal_blocker_index = matches.index[0]

                     except Exception as e:
                          logger.error(f"Sell {sell_index}: Error matching blocker {blocker_orig_idx}: {e}. Skipping.")
                          continue

                     # Check Blocker's Remaining Capacity from internal state
                     blocker_total_qty = internal_df.at[internal_blocker_index, SHARES]
                     blocker_remaining_capacity = internal_df.at[internal_blocker_index, _INTERNAL_BLOCKING_CAPACITY_REMAINING]
                     logger.debug(f"Sell {sell_index}: Blocker {internal_blocker_index} - RemainingCapacity={blocker_remaining_capacity}")

                     if blocker_remaining_capacity < 1e-9:
                         logger.debug(f"Sell {sell_index}: Blocker {internal_blocker_index} has no capacity left.")
                         continue # Skip if no capacity

                     # Determine Shares Blocked by This Blocker for This Sell
                     shares_this_blocker_blocks_now = min(remaining_shares_from_sell_to_block, blocker_remaining_capacity)
                     logger.debug(f"Sell {sell_index}: Blocker {internal_blocker_index} will block {shares_this_blocker_blocks_now} shares now.")

                     # Update Blocker State Directly in internal_df if shares are blocked
                     if shares_this_blocker_blocks_now > 1e-9:
                         # 1. Decrement Remaining Capacity
                         new_remaining_capacity = blocker_remaining_capacity - shares_this_blocker_blocks_now
                         internal_df.at[internal_blocker_index, _INTERNAL_BLOCKING_CAPACITY_REMAINING] = new_remaining_capacity

                         # 2. Calculate and Accumulate Deferred Adjustment
                         loss_adj_for_this_blocker = shares_this_blocker_blocks_now * abs(loss_per_share)
                         current_adj = internal_df.at[internal_blocker_index, _INTERNAL_DEFERRED_ADJUSTMENT]
                         new_total_adj = current_adj + loss_adj_for_this_blocker
                         internal_df.at[internal_blocker_index, _INTERNAL_DEFERRED_ADJUSTMENT] = new_total_adj

                         # 3. Recalculate Adjusted Cost Basis for the blocker
                         original_price = internal_df.at[internal_blocker_index, SHARE_PRICE]
                         if blocker_total_qty > 1e-9:
                              new_adjusted_cost = original_price + (new_total_adj / blocker_total_qty)
                              internal_df.at[internal_blocker_index, _INTERNAL_ADJUSTED_COST_PER_SHARE] = new_adjusted_cost
                              logger.debug(f"Sell {sell_index}: Updated Blocker {internal_blocker_index} - NewRemainingCapacity={new_remaining_capacity}, NewTotalAdj={new_total_adj}, NewAdjCost={new_adjusted_cost:.4f}")
                         else:
                              logger.debug(f"Sell {sell_index}: Updated Blocker {internal_blocker_index} - NewRemainingCapacity={new_remaining_capacity}, NewTotalAdj={new_total_adj}, Cost N/A (zero qty)")


                         # Update loop variables for the current sell
                         actual_shares_blocked_for_this_sell += shares_this_blocker_blocks_now
                         remaining_shares_from_sell_to_block -= shares_this_blocker_blocks_now

                     # Check if all shares of the current sell have been blocked
                     if remaining_shares_from_sell_to_block < 1e-9:
                         logger.debug(f"Sell {sell_index}: All shares blocked. Breaking blocker loop.")
                         break

                 # Calculate Final Allowable Loss for This Sell based on actual shares blocked
                 total_deferred_loss_amount = actual_shares_blocked_for_this_sell * abs(loss_per_share)
                 allowable_loss = adjusted_gain_loss + total_deferred_loss_amount
                 allowable_loss = min(allowable_loss, 0.0) # Ensure allowable loss is not positive
                 logger.debug(f"Sell {sell_index}: Finished blocking. ActualTotalBlocked={actual_shares_blocked_for_this_sell:.2f}, Final Allowable={allowable_loss:.2f}")
                 return (allowable_loss, 0.0) # Return allowable loss for the sell, adjustment is 0

         else:
             # Gain or zero loss - no deferral applies
             adjusted_gain_loss = max(adjusted_gain_loss, 0.0)
             logger.debug(f"Sell {sell_index}: Gain calculated = {adjusted_gain_loss:.2f}")
             return (adjusted_gain_loss, 0.0) # Return gain for the sell, adjustment is 0

    # --- Find Repurchases Helper ---
    def _find_repurchases(self, ticker: str, current_date: pd.Timestamp, lookup_df: pd.DataFrame) -> pd.DataFrame:
        """
        Finds buy transactions (repurchases) of the same ticker within the strict
        +/- 2 month window of a given date in the lookup DataFrame.
        """
        two_months_before = current_date - DateOffset(months=2)
        two_months_after = current_date + DateOffset(months=2)
        ticker_match = lookup_df[TICKER] == ticker
        is_buy = lookup_df[TRANSACTION_TYPE].str.lower() == _TRANSACTION_TYPE_BUY
        # Strict inequality for the window check based on common interpretation
        after_window_start = lookup_df[DATE] > two_months_before
        before_window_end = lookup_df[DATE] < two_months_after
        valid_quantity = ~pd.isna(lookup_df[SHARES]) & (lookup_df[SHARES] > 0)
        repurchases = lookup_df[
            ticker_match & is_buy & after_window_start & before_window_end & valid_quantity
        ]
        return repurchases

