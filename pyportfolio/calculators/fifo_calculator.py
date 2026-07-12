# pyportfolio/calculators/fifo_calculator.py

import pandas as pd
import logging
import numpy as np

from pyportfolio.calculators.base_calculator import BaseTableCalculator, is_overselling

# --- Constants ---
from pyportfolio.columns import (
    SHARE_PRICE,
    SHARES,
    TRANSACTION_TYPE,
    TICKER,
    DATETIME,
    COMISION, # Added COMISION
    TYPE_BUY,
    TYPE_SELL,
    TYPE_TRANSFER
)

RESULT_FIFO_GAIN_LOSS = "FIFO Beneficio / Pérdida"

# Internal state columns (_ prefix indicates internal use)
_INTERNAL_FIFO_AVAILABLE_SHARES = '_FifoAvailableShares'
_INTERNAL_FIFO_COST_PER_SHARE_WITH_COMMISSION = '_FifoCostPerShareWithCommission' # Added for cost basis

# --- Logger ---
logger = logging.getLogger(__name__)

class FIFOCalculator(BaseTableCalculator):
    """
    Calculates the FIFO (First-In, First-Out) capital gain or loss for sell transactions
    as a table-wise operation, including commission costs.

    Handles sell transactions where the SHARES column contains negative values.
    Processes transactions chronologically for each ticker, tracking available shares
    and their commission-adjusted cost basis from buy lots. Assumes commissions are positive.
    """

    def __init__(self):
        """
        Initializes the table-wise calculator.
        """
        super().__init__()
        logger.debug("FIFOCalculator initialized (Table-wise).")


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
                pd.to_datetime(df[DATETIME], format='mixed', dayfirst=True) # Check conversion feasibility
            except Exception as e:
                raise ValueError(f"Could not convert date column '{DATETIME}' to datetime: {e}")        

    def calculate_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs the FIFO gain/loss calculation on the entire DataFrame, including commissions.

        Args:
            df: The transaction DataFrame provided by the TransactionManager.
                Expected columns: DATETIME, TICKER, TRANSACTION_TYPE, SHARES, SHARE_PRICE, COMISION.
                Expects SHARES to be positive for buys and negative for sells.
                Expects COMISION to be positive for both buys (cost) and sells (expense).

        Returns:
            A new DataFrame containing only the calculated result column
            (RESULT_FIFO_GAIN_LOSS) with the same index as the input df.
            The result is the gain/loss for sell transactions, and pd.NA otherwise.
        """
        logger.info("Starting FIFO calculate_table (with commissions)")
        self._validate_input_dataframe(df)

        internal_df = df.copy(deep=True)

        if not pd.api.types.is_datetime64_any_dtype(internal_df[DATETIME]):
            try:
                internal_df[DATETIME] = pd.to_datetime(internal_df[DATETIME], format='mixed',  dayfirst=True)
            except Exception as e:
                raise ValueError(f"Could not convert internal date {internal_df[DATETIME]} column '{DATETIME}' to datetime: {e}")     
             

        # Ensure commission is numeric, fill NaN with 0
        internal_df[COMISION] = pd.to_numeric(internal_df[COMISION], errors='coerce').fillna(0.0)
                

        internal_df = internal_df.sort_values(
            by=[TICKER, DATETIME, TRANSACTION_TYPE], ascending=[True, True, True]
        ).reset_index()

        logging.debug(f"Data: internal_df[[TICKER, DATETIME, TRANSACTION_TYPE]].to_string()")

        original_index_name = 'original_index'
        if 'index' in internal_df.columns:
             internal_df = internal_df.rename(columns={'index': original_index_name})
        else:
             internal_df[original_index_name] = internal_df.index
             internal_df = internal_df.reset_index(drop=True)


        internal_df[_INTERNAL_FIFO_AVAILABLE_SHARES] = 0.0
        internal_df[_INTERNAL_FIFO_COST_PER_SHARE_WITH_COMMISSION] = 0.0
        internal_df[RESULT_FIFO_GAIN_LOSS] = pd.NA
        internal_df[RESULT_FIFO_GAIN_LOSS] = internal_df[RESULT_FIFO_GAIN_LOSS].astype('Float64')

        # Initialize available shares and cost basis for every row that ADDS shares to
        # the FIFO stock: buys and transfer-ins (traspaso with positive shares).
        # For a transfer-in, SHARE_PRICE is expected to carry the cost basis per share
        # brought over from the origin fund (set by the global carried-cost pre-pass);
        # commissions are normally 0 on a traspaso.
        type_lower = internal_df[TRANSACTION_TYPE].str.lower()
        entry_mask = type_lower.isin([TYPE_BUY, TYPE_TRANSFER]) & (internal_df[SHARES] > 1e-9)

        # Calculate total cost (price*shares + commission) for entries
        total_cost_entry = (internal_df.loc[entry_mask, SHARES] * internal_df.loc[entry_mask, SHARE_PRICE]) + internal_df.loc[entry_mask, COMISION]

        # Calculate cost per share including commission, handle potential division by zero
        cost_per_share_entry = total_cost_entry / internal_df.loc[entry_mask, SHARES]
        cost_per_share_entry = cost_per_share_entry.replace([np.inf, -np.inf], 0).fillna(0) # Handle division by zero or NaN shares

        internal_df.loc[entry_mask, _INTERNAL_FIFO_AVAILABLE_SHARES] = internal_df.loc[entry_mask, SHARES]
        internal_df.loc[entry_mask, _INTERNAL_FIFO_COST_PER_SHARE_WITH_COMMISSION] = cost_per_share_entry

        logger.info("Starting FIFO chronological processing loop within calculate_table...")

        for index, row in internal_df.iterrows():
            transaction_type = str(row.get(TRANSACTION_TYPE, '')).lower()
            ticker = row.get(TICKER)
            original_shares_value = row.get(SHARES)

            # Consuming events remove shares from the FIFO stock:
            #   - a sell (venta), which realises a gain/loss, and
            #   - a transfer-out (traspaso with negative shares), which consumes lots
            #     (carrying their cost forward) but realises NO gain (deferred taxation).
            is_sell = transaction_type == TYPE_SELL
            is_transfer_out = (
                transaction_type == TYPE_TRANSFER
                and pd.notna(original_shares_value)
                and original_shares_value < -1e-9
            )

            if not (is_sell or is_transfer_out):
                # Buys and transfer-ins were already added to the stock above; nothing to consume.
                if transaction_type == TYPE_BUY and (pd.isna(original_shares_value) or original_shares_value < 1e-9):
                    logger.warning(f"Buy row at index {index} has non-positive SHARES ({original_shares_value}). Check data integrity.")
                continue

            event_label = "SELL" if is_sell else "TRANSFER-OUT"
            logger.debug(f"Processing {event_label} row {index}: {row.get(DATETIME)} / Ticker: {ticker}")

            shares_to_consume = np.abs(original_shares_value) if pd.notna(original_shares_value) else 0.0
            sell_price = row.get(SHARE_PRICE)

            # Validate essential data for a consuming event
            if pd.isna(ticker) or pd.isna(original_shares_value) or original_shares_value >= -1e-9:
                logger.warning(
                    f"{event_label} row at index {index} missing essential data or has non-negative SHARES. "
                    f"(Ticker: {ticker}, Shares: {original_shares_value}). Skipping."
                )
                continue
            if is_sell and pd.isna(sell_price):
                logger.warning(f"Sell row at index {index} missing price. (Ticker: {ticker}). Skipping.")
                continue

            # Available lots come from previous buys AND transfer-ins of the same ticker.
            previous_entries = internal_df[
                (internal_df.index < index) &
                (internal_df[TICKER] == ticker) &
                (internal_df[TRANSACTION_TYPE].str.lower().isin([TYPE_BUY, TYPE_TRANSFER])) &
                (internal_df[_INTERNAL_FIFO_AVAILABLE_SHARES] > 1e-9)
            ]

            total_available_before = previous_entries[_INTERNAL_FIFO_AVAILABLE_SHARES].sum()

            if is_overselling(shares_to_consume, total_available_before):
                if is_sell:
                    logger.error(f"Overselling detected for {event_label} at index {index} (Ticker: {ticker}, Date: {row.get(DATETIME)}) . Available: {total_available_before:.4f}, Consuming (abs): {shares_to_consume:.4f}")
                    raise ValueError(
                        f"Cannot sell {shares_to_consume:.4f} shares of {ticker} at {row.get(DATETIME)}; "
                        f"only {total_available_before:.4f} are available from previous buys."
                    )
                # Transfer-out short of stock: tolerate. Real datasets contain "Traspaso" rows for
                # cash accounts (no FIFO stock) and for dividend mirror rows; aborting the whole
                # calculation for those would break every other product. Consume what exists,
                # warn, and let genuine fund data errors surface at the next real sell.
                logger.warning(
                    f"Transfer-out at index {index} (Ticker: {ticker}, Date: {row.get(DATETIME)}) requests "
                    f"{shares_to_consume:.4f} but only {total_available_before:.4f} are available. "
                    f"Consuming the available amount only (cash-account or unmatched traspaso?)."
                )
                shares_to_consume = total_available_before
            else:
                # Within float-rounding tolerance: clamp any tiny overshoot so the
                # origin balance settles cleanly to zero instead of leaving noise.
                shares_to_consume = min(shares_to_consume, total_available_before)

            total_cost_basis = 0.0
            remaining_to_consume = shares_to_consume

            for buy_idx in previous_entries.index:
                available_in_lot = internal_df.at[buy_idx, _INTERNAL_FIFO_AVAILABLE_SHARES]
                cost_per_share_with_commission = internal_df.at[buy_idx, _INTERNAL_FIFO_COST_PER_SHARE_WITH_COMMISSION]

                if pd.isna(cost_per_share_with_commission):
                     logger.error(f"NaN cost per share in lot {buy_idx} when processing {event_label} {index}.")
                     raise ValueError(f"Found entry lot (index {buy_idx}) with NaN cost basis needed for FIFO calculation.")

                consume_from_lot = min(available_in_lot, remaining_to_consume)
                total_cost_basis += consume_from_lot * cost_per_share_with_commission

                internal_df.at[buy_idx, _INTERNAL_FIFO_AVAILABLE_SHARES] -= consume_from_lot
                remaining_to_consume -= consume_from_lot

                logger.debug(f"  Consumed {consume_from_lot:.4f} from lot {buy_idx} @ cost {cost_per_share_with_commission:.4f}. Remaining: {remaining_to_consume:.4f}. Lot available now: {internal_df.at[buy_idx, _INTERNAL_FIFO_AVAILABLE_SHARES]:.4f}")

                if remaining_to_consume < 1e-9:
                    break

            if is_sell:
                # Calculate proceeds subtracting the (assumed positive) sell commission
                commission_sell = row.get(COMISION, 0.0)
                sell_proceeds = (sell_price * shares_to_consume) - commission_sell
                fifo_gain_loss = sell_proceeds - total_cost_basis
                internal_df.loc[index, RESULT_FIFO_GAIN_LOSS] = fifo_gain_loss
                logger.debug(f"Sell row {index}: Proceeds={sell_proceeds:.4f}, Total Cost Basis={total_cost_basis:.4f}, FIFO Gain/Loss={fifo_gain_loss:.4f}")
            else:
                # Transfer-out: lots consumed and their cost carried forward, but no realised
                # gain (deferred). RESULT_FIFO_GAIN_LOSS stays NA for this row.
                logger.debug(f"Transfer-out row {index}: consumed {shares_to_consume:.4f} shares of {ticker}, cost basis {total_cost_basis:.4f} carried forward (no realised gain).")

        logger.info("Finished FIFO chronological processing loop.")

        internal_df = internal_df.set_index(original_index_name)
        internal_df.index.name = df.index.name

        #results_df = internal_df[[RESULT_FIFO_GAIN_LOSS]].copy()
        #results_df = results_df.reindex(df.index)

        results_df = internal_df

        logger.info("Finished FIFO calculate_table (with commissions).")
        return results_df