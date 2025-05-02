import pandas as pd
from pandas.tseries.offsets import DateOffset
from typing import Optional, Tuple, List
import logging
import numpy as np

# Import the correct base class
from .base_calculator import BaseTableCalculator

# --- Constants ---
from pyportfolio.columns import (
    SHARE_PRICE,
    SHARES,
    TRANSACTION_TYPE,
    TICKER,
    DATETIME,
    COMISION,
    TYPE_BUY, # Use TYPE_BUY constant
    TYPE_SELL # Use TYPE_SELL constant
)

# Result Column Names
RESULT_TAXABLE_GAIN_LOSS = 'IRPF - Ganancia / Pérdida Imputable'
RESULT_DEFERRED_ADJUSTMENT = 'IRPF - Ajuste Diferido'
_RESULT_COLUMNS = [RESULT_TAXABLE_GAIN_LOSS, RESULT_DEFERRED_ADJUSTMENT]

# Internal State Columns (prefix _ indicates internal use)
_INTERNAL_AVAILABLE_SHARES = '_IrpfAvailableShares'
_INTERNAL_DEFERRED_ADJUSTMENT_STATE = '_IrpfDeferredAdjustmentState'
_INTERNAL_ADJUSTED_COST_PER_SHARE = '_IrpfAdjustedCostPerShare' # Cost basis including commission AND deferred losses
_INTERNAL_BLOCKING_CAPACITY_REMAINING = '_IrpfBlockingCapacityRemaining'
_INTERNAL_ORIGINAL_COST_PER_SHARE_WITH_COMMISSION = '_IrpfOriginalCostPerShareWithCommission' # Cost basis including commission ONLY
_ORIGINAL_INDEX_COL = '_original_index' # Column to store original index

# --- Logger ---
logger = logging.getLogger(__name__)

class IrpfEarningsCalculator(BaseTableCalculator):
    """
    Calculates the capital gain or loss for Spanish IRPF purposes as a table-wise operation,
    including commission costs and handling negative SHARES for sells.

    Applies the "two-month rule" (regla de los dos meses) and adjusts
    cost basis for deferred losses, using a "consuming capacity" logic for blocking buys.
    Performs its own adjusted FIFO calculation internally, incorporating commissions into
    the cost basis and subtracting them from sell proceeds.
    Assumes positive SHARES for buys and negative SHARES for sells.
    """

    def __init__(self):
        """
        Initializes the table-wise calculator. No pre-calculation is done here.
        """
        super().__init__()
        logger.debug("IrpfEarningsCalculator initialized (Table-wise, with commissions, negative shares for sells).")

    def _validate_input_dataframe(self, df: pd.DataFrame):
        """ Performs validation checks on the input DataFrame. """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        required_columns = [DATETIME, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE, COMISION]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame must contain columns: {required_columns}. Missing: {missing_cols}")
        if not pd.api.types.is_datetime64_any_dtype(df[DATETIME]):
            try:
                pd.to_datetime(df[DATETIME], format='mixed', dayfirst=True)
            except Exception as e:
                raise ValueError(f"Could not convert {df[DATETIME]} from date column '{DATETIME}' to datetime: {e}")

    def calculate_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs the IRPF calculation on the entire DataFrame, including commissions
        and handling negative SHARES for sells.

        Args:
            df: The transaction DataFrame provided by the TransactionManager.
                Expected columns: DATETIME, TICKER, TRANSACTION_TYPE, SHARES, SHARE_PRICE, COMISION.
                Expects SHARES to be positive for buys and negative for sells.
                Expects COMISION to be positive for both buys (cost) and sells (expense).

        Returns:
            A new DataFrame containing only the calculated result columns
            (RESULT_TAXABLE_GAIN_LOSS, RESULT_DEFERRED_ADJUSTMENT)
            with the same index as the input df.
        """
        logger.info("Starting IRPF calculate_table (with commissions, negative shares for sells)")

        self._validate_input_dataframe(df)

        # --- Save original index object and name for final reindex ---
        original_input_index = df.index
        original_input_index_name = df.index.name
        # ---

        internal_df = df.copy(deep=True)

        if not pd.api.types.is_datetime64_any_dtype(internal_df[DATETIME]):
            try:
                internal_df[DATETIME] = pd.to_datetime(internal_df[DATETIME], format='mixed', dayfirst=True)
            except Exception as e:
                raise ValueError(f"Could not convert {internal_df[DATETIME]} from date column '{DATETIME}' to datetime: {e}")       
            

        internal_df[COMISION] = pd.to_numeric(internal_df[COMISION], errors='coerce').fillna(0.0)
        internal_df[SHARES] = pd.to_numeric(internal_df[SHARES], errors='coerce').fillna(0.0) # Ensure SHARES is numeric

        # --- Sort and reset index, keeping old index in a column ---
        internal_df = internal_df.sort_values(
            by=[DATETIME, TRANSACTION_TYPE], ascending=[True, True]
        ).reset_index() # Keep old index

        # Rename the old index column
        if 'index' in internal_df.columns:
             internal_df = internal_df.rename(columns={'index': _ORIGINAL_INDEX_COL})
        else:
             # Should not happen with reset_index(), but as fallback
             internal_df[_ORIGINAL_INDEX_COL] = internal_df.index
             internal_df = internal_df.reset_index(drop=True)
        # ---

        # --- Initialize internal and result columns ---
        internal_df[_INTERNAL_AVAILABLE_SHARES] = 0.0
        internal_df[_INTERNAL_DEFERRED_ADJUSTMENT_STATE] = 0.0
        internal_df[_INTERNAL_ADJUSTED_COST_PER_SHARE] = 0.0
        internal_df[_INTERNAL_ORIGINAL_COST_PER_SHARE_WITH_COMMISSION] = 0.0
        internal_df[_INTERNAL_BLOCKING_CAPACITY_REMAINING] = 0.0

        internal_df[RESULT_TAXABLE_GAIN_LOSS] = pd.NA
        internal_df[RESULT_DEFERRED_ADJUSTMENT] = pd.NA
        internal_df[RESULT_TAXABLE_GAIN_LOSS] = internal_df[RESULT_TAXABLE_GAIN_LOSS].astype('Float64')
        internal_df[RESULT_DEFERRED_ADJUSTMENT] = internal_df[RESULT_DEFERRED_ADJUSTMENT].astype('Float64')
        # ---

        # Initialize state for buy transactions (positive shares)
        buy_mask = (internal_df[TRANSACTION_TYPE].str.lower() == TYPE_BUY) & (internal_df[SHARES] > 1e-9)

        total_cost_buy = (internal_df.loc[buy_mask, SHARES] * internal_df.loc[buy_mask, SHARE_PRICE]) + internal_df.loc[buy_mask, COMISION]
        # Avoid division by zero if SHARES is somehow zero on a buy row
        shares_for_cost_calc = internal_df.loc[buy_mask, SHARES].replace(0, np.nan)
        cost_per_share_buy_with_commission = total_cost_buy / shares_for_cost_calc
        cost_per_share_buy_with_commission = cost_per_share_buy_with_commission.replace([np.inf, -np.inf], 0).fillna(0)

        internal_df.loc[buy_mask, _INTERNAL_AVAILABLE_SHARES] = internal_df.loc[buy_mask, SHARES]
        internal_df.loc[buy_mask, _INTERNAL_BLOCKING_CAPACITY_REMAINING] = internal_df.loc[buy_mask, SHARES]
        internal_df.loc[buy_mask, _INTERNAL_ADJUSTED_COST_PER_SHARE] = cost_per_share_buy_with_commission
        internal_df.loc[buy_mask, _INTERNAL_ORIGINAL_COST_PER_SHARE_WITH_COMMISSION] = cost_per_share_buy_with_commission


        logger.info("Starting IRPF chronological processing loop within calculate_table...")
        # --- Loop using the 0..N-1 index ---
        for index, row in internal_df.iterrows():
            transaction_type = str(row.get(TRANSACTION_TYPE, '')).lower()
            shares_value = row.get(SHARES, 0.0)
            logger.debug(f"Processing row {index}:. Type: {transaction_type}, Shares: {shares_value}")

            try:
                # Process sells based on negative shares OR transaction type 'sell'
                if shares_value < -1e-9 or transaction_type == TYPE_SELL:
                    if shares_value >= -1e-9 and transaction_type == TYPE_SELL:
                         logger.warning(f"Row {index} has Transaction Type '{TYPE_SELL}' but non-negative SHARES ({shares_value}). Processing as sell based on type.")
                    elif shares_value < -1e-9 and transaction_type != TYPE_SELL:
                         logger.warning(f"Row {index} has negative SHARES ({shares_value}) but Transaction Type is not '{TYPE_SELL}' (it's '{transaction_type}'). Processing as sell based on shares.")

                    logger.debug(f"Processing SELL row {index}: {row.get(DATETIME)} / Ticker: {row.get(TICKER)})")
                    result_tuple = self._process_sell_and_update_buys(row, index, internal_df)
                    if result_tuple is not None:
                        internal_df.loc[index, RESULT_TAXABLE_GAIN_LOSS] = result_tuple[0]
                        internal_df.loc[index, RESULT_DEFERRED_ADJUSTMENT] = result_tuple[1] # Should be 0.0 for sells

                # Process buys based on positive shares OR transaction type 'buy'
                elif shares_value > 1e-9 or transaction_type == TYPE_BUY:
                    if shares_value <= 1e-9 and transaction_type == TYPE_BUY:
                         logger.warning(f"Row {index} has Transaction Type '{TYPE_BUY}' but non-positive SHARES ({shares_value}). Skipping cost basis init for this row.")
                    elif shares_value > 1e-9 and transaction_type != TYPE_BUY:
                         logger.warning(f"Row {index} has positive SHARES ({shares_value}) but Transaction Type is not '{TYPE_BUY}' (it's '{transaction_type}'). Processing as buy based on shares.")

                    logger.debug(f"Processing BUY row {index}: {row.get(DATETIME)} / Ticker: {row.get(TICKER)})")
                    internal_df.loc[index, RESULT_TAXABLE_GAIN_LOSS] = 0.0
                    # Placeholder, will be overwritten later for buys
                    internal_df.loc[index, RESULT_DEFERRED_ADJUSTMENT] = 0.0

                else:
                    # Handle other transaction types (e.g., dividend) or zero shares
                    logger.debug(f"Ignoring row {index} with type '{transaction_type}' and shares {shares_value}.")
                    internal_df.loc[index, RESULT_TAXABLE_GAIN_LOSS] = None
                    internal_df.loc[index, RESULT_DEFERRED_ADJUSTMENT] = None

            except Exception as e:
                 logger.error(f"Error processing row {index} during IRPF calculation: {e}. Row: {row.to_dict()}", exc_info=True)
                 internal_df.loc[index, RESULT_TAXABLE_GAIN_LOSS] = None
                 internal_df.loc[index, RESULT_DEFERRED_ADJUSTMENT] = None
        # --- End Loop ---

        logger.info("Populating final deferred adjustment results for buy transactions...")
        buy_mask_final = (internal_df[TRANSACTION_TYPE].str.lower() == TYPE_BUY) & (internal_df[SHARES] > 1e-9)
        internal_df.loc[buy_mask_final, RESULT_DEFERRED_ADJUSTMENT] = internal_df.loc[buy_mask_final, _INTERNAL_DEFERRED_ADJUSTMENT_STATE]

        # --- Restore original index and prepare results ---
        try:
            internal_df = internal_df.set_index(_ORIGINAL_INDEX_COL)
            internal_df.index.name = original_input_index_name # Restore original index name if it existed
        except KeyError:
             logger.error(f"Internal error: Column '{_ORIGINAL_INDEX_COL}' not found after processing.")
             # Fallback: Try to use the original index directly, though alignment might be wrong
             internal_df.index = original_input_index
             internal_df.index.name = original_input_index_name

        # Create results DataFrame (now has original index values, but maybe not order)
        results_df = internal_df[_RESULT_COLUMNS].copy()

        # Explicitly reindex using the original input index object to ensure correct order
        results_df = results_df.reindex(original_input_index)
        # ---

        logger.info("Finished IRPF calculate_table (with commissions, negative shares for sells).")
        return results_df


    def _process_sell_and_update_buys(self, sell_row: pd.Series, sell_index: int, internal_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
         """
         Processes a sell transaction (negative SHARES): calculates adjusted FIFO gain/loss,
         handles loss deferral by finding blockers and updating their state.
         Returns the (allowable_gain_loss, 0.0) tuple for the sell row.
         Uses the 0..N-1 index ('sell_index') for internal operations on 'internal_df'.
         """
         ticker = sell_row.get(TICKER)
         original_shares_value = sell_row.get(SHARES)
         sell_price = sell_row.get(SHARE_PRICE)
         sell_date = sell_row.get(DATETIME)
         commission_sell = sell_row.get(COMISION, 0.0)

         # Use absolute value for quantity calculations
         shares_to_sell = np.abs(original_shares_value) if pd.notna(original_shares_value) else 0.0

         logger.debug(f"Processing sell row {sell_index}. Ticker: {ticker}, Orig Shares: {original_shares_value}, Shares Qty: {shares_to_sell}, Price: {sell_price}, Date: {sell_date}, Commission: {commission_sell}")

         # Validate essential data for a sell
         if pd.isna(ticker) or pd.isna(original_shares_value) or original_shares_value >= -1e-9 or pd.isna(sell_price) or pd.isna(sell_date):
             logger.warning(f"Sell row {sell_index} missing essential data or has non-negative SHARES. Cannot calculate.")
             return (None, None)

         # Find previous buys using the 0..N-1 index
         previous_buys = internal_df[
             (internal_df.index < sell_index) & # Use 0..N-1 index for comparison
             (internal_df[TICKER] == ticker) &
             (internal_df[TRANSACTION_TYPE].str.lower() == TYPE_BUY) &
             (internal_df[_INTERNAL_AVAILABLE_SHARES] > 1e-9)
         ]

         total_available_before = previous_buys[_INTERNAL_AVAILABLE_SHARES].sum()

         # Check overselling using the absolute quantity
         if shares_to_sell > total_available_before + 1e-9:
             logger.error(f"Overselling detected for sell {sell_index} (Ticker: {ticker}, Date: {sell_date}). Available: {total_available_before:.4f}, Selling Qty: {shares_to_sell:.4f}")
             # Consider raising an error or returning specific error code
             return (None, None) # Or raise ValueError

         total_adjusted_cost_basis = 0.0
         remaining_to_sell = shares_to_sell

         # Iterate through the 0..N-1 indices of previous buys
         for buy_idx in previous_buys.index:
             available_in_lot = internal_df.at[buy_idx, _INTERNAL_AVAILABLE_SHARES]
             adjusted_cost_per_share = internal_df.at[buy_idx, _INTERNAL_ADJUSTED_COST_PER_SHARE]

             if pd.isna(adjusted_cost_per_share):
                  logger.error(f"NaN adjusted cost per share in buy lot {buy_idx} when processing sell {sell_index}.")
                  return (None, None) # Or raise ValueError

             consume_from_lot = min(available_in_lot, remaining_to_sell)
             total_adjusted_cost_basis += consume_from_lot * adjusted_cost_per_share
             internal_df.at[buy_idx, _INTERNAL_AVAILABLE_SHARES] -= consume_from_lot
             remaining_to_sell -= consume_from_lot

             logger.debug(f"  Sell {sell_index}: Consumed {consume_from_lot:.4f} from buy lot {buy_idx} @ adjusted cost {adjusted_cost_per_share:.4f}. Remaining to sell: {remaining_to_sell:.4f}. Lot available now: {internal_df.at[buy_idx, _INTERNAL_AVAILABLE_SHARES]:.4f}")

             if remaining_to_sell < 1e-9: break

         # Calculate proceeds using absolute quantity, subtracting commission
         sell_proceeds = (sell_price * shares_to_sell) - commission_sell
         adjusted_gain_loss = sell_proceeds - total_adjusted_cost_basis

         logger.debug(f"Sell {sell_index}: Gross Proceeds={sell_price * shares_to_sell:.4f}, Sell Commission={commission_sell:.4f}, Net Proceeds={sell_proceeds:.4f}, Total Adjusted Cost Basis={total_adjusted_cost_basis:.4f}, Initial Gain/Loss={adjusted_gain_loss:.4f}")

         if adjusted_gain_loss < -1e-9:
             # Calculate loss per share using absolute quantity
             loss_per_share = adjusted_gain_loss / shares_to_sell if shares_to_sell > 1e-9 else 0.0

             # Find blockers using the 0..N-1 indexed internal_df
             repurchases = self._find_repurchases(ticker, sell_date, internal_df)

             if repurchases.empty:
                 logger.debug(f"Sell {sell_index}: No repurchases found. Allowable loss = {adjusted_gain_loss:.2f}")
                 return (adjusted_gain_loss, 0.0)
             else:
                 # repurchases DataFrame still has 0..N-1 index
                 sorted_repurchases = repurchases.sort_values(by=DATETIME)
                 actual_shares_blocked_for_this_sell = 0.0
                 # Start blocking capacity check with absolute quantity
                 remaining_shares_from_sell_to_block = shares_to_sell
                 logger.debug(f"Sell {sell_index}: Loss={adjusted_gain_loss:.2f} (Loss per share={loss_per_share:.4f}). Found {len(sorted_repurchases)} potential blockers. Need to block {remaining_shares_from_sell_to_block} shares.")

                 # Iterate through the 0..N-1 indices of the blockers
                 for internal_blocker_index in sorted_repurchases.index:
                     logger.debug(f"--- Sell {sell_index}: Checking potential blocker index {internal_blocker_index} Date: {internal_df.at[internal_blocker_index, DATETIME]} ---")

                     blocker_total_qty = internal_df.at[internal_blocker_index, SHARES] # Original positive quantity
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

                         original_cost_with_commission_per_share = internal_df.at[internal_blocker_index, _INTERNAL_ORIGINAL_COST_PER_SHARE_WITH_COMMISSION]

                         if blocker_total_qty > 1e-9:
                              increase_in_cost_per_share = new_total_adj / blocker_total_qty
                              new_adjusted_cost = original_cost_with_commission_per_share + increase_in_cost_per_share
                              internal_df.at[internal_blocker_index, _INTERNAL_ADJUSTED_COST_PER_SHARE] = new_adjusted_cost
                              logger.debug(f"Sell {sell_index}: Updated Blocker {internal_blocker_index} - NewRemainingCapacity={new_remaining_capacity:.2f}, NewTotalDeferredAdj={new_total_adj:.2f}, OrigCostPerShare={original_cost_with_commission_per_share:.4f}, IncreasePerShare={increase_in_cost_per_share:.4f}, NewAdjCostPerShare={new_adjusted_cost:.4f}")
                         else:
                              internal_df.at[internal_blocker_index, _INTERNAL_ADJUSTED_COST_PER_SHARE] = original_cost_with_commission_per_share
                              logger.warning(f"Sell {sell_index}: Blocker {internal_blocker_index} has zero total quantity. Setting adjusted cost to original cost with commission.")

                         actual_shares_blocked_for_this_sell += shares_this_blocker_blocks_now
                         remaining_shares_from_sell_to_block -= shares_this_blocker_blocks_now

                     if remaining_shares_from_sell_to_block < 1e-9:
                         logger.debug(f"Sell {sell_index}: All shares from sell loss blocked. Breaking blocker loop.")
                         break

                 total_deferred_loss_amount = actual_shares_blocked_for_this_sell * abs(loss_per_share)
                 allowable_loss = adjusted_gain_loss + total_deferred_loss_amount
                 allowable_loss = min(allowable_loss, 0.0) # Ensure loss doesn't become positive due to float issues
                 logger.debug(f"Sell {sell_index}: Finished blocking. ActualTotalBlocked={actual_shares_blocked_for_this_sell:.2f}, TotalDeferredAmount={total_deferred_loss_amount:.2f}, Final Allowable Loss={allowable_loss:.2f}")
                 return (allowable_loss, 0.0)

         else: # Gain or zero
             # Ensure gain is not negative due to float issues
             adjusted_gain_loss = max(adjusted_gain_loss, 0.0)
             logger.debug(f"Sell {sell_index}: Gain calculated = {adjusted_gain_loss:.2f}")
             return (adjusted_gain_loss, 0.0)

    def _find_repurchases(self, ticker: str, current_date: pd.Timestamp, lookup_df: pd.DataFrame) -> pd.DataFrame:
        """
        Finds buy transactions (repurchases) of the same ticker within the strict
        +/- 2 month window of a given date in the lookup DataFrame.
        Looks for TYPE_BUY and positive SHARES.
        Assumes lookup_df uses 0..N-1 index.
        """
        two_months_before = current_date - DateOffset(months=2)
        two_months_after = current_date + DateOffset(months=2)

        # Ensure date column in lookup_df is datetime
        if not pd.api.types.is_datetime64_any_dtype(lookup_df[DATETIME]):
             lookup_df_dates = pd.to_datetime(lookup_df[DATETIME], format='mixed', dayfirst=True)
        else:
             lookup_df_dates = lookup_df[DATETIME]

        ticker_match = lookup_df[TICKER] == ticker
        # Explicitly check for TYPE_BUY and positive shares for repurchases
        is_buy = lookup_df[TRANSACTION_TYPE].str.lower() == TYPE_BUY
        is_positive_shares = ~pd.isna(lookup_df[SHARES]) & (lookup_df[SHARES] > 1e-9)

        after_window_start = lookup_df_dates > two_months_before
        before_window_end = lookup_df_dates < two_months_after

        repurchases = lookup_df[
            ticker_match & is_buy & is_positive_shares & after_window_start & before_window_end
        ]
        return repurchases

