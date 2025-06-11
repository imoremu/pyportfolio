import pandas as pd
import logging
from typing import List, Optional # Added for type hinting

# Configuración básica del logger
logging.basicConfig(filename='logfile.log', level=logging.ERROR, # Changed to INFO for more detail
                    format='%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s')

# Assuming these are imported correctly from your project structure
from pyportfolio.transaction_manager import TransactionManager
from pyportfolio.calculators.fifo_calculator import FIFOCalculator
from pyportfolio.calculators.dividend_calculator import DividendCalculator
from pyportfolio.calculators.irpf_earnings_calculator import (
    IrpfEarningsCalculator,
    RESULT_TAXABLE_GAIN_LOSS,
    RESULT_DEFERRED_ADJUSTMENT
)
from pyportfolio.columns import FIFO, DATETIME, TRANSACTION_TYPE, TICKER, DIVIDEND_EARNINGS # Added DIVIDEND_EARNINGS

logger = logging.getLogger(__name__)

# --- Configuration ---
TEST_COLUMN_NAME = 'Test' 

fifo_result_column = FIFO
dividend_result_column = DIVIDEND_EARNINGS # Use the constant
irpf_result_columns = [RESULT_TAXABLE_GAIN_LOSS, RESULT_DEFERRED_ADJUSTMENT]
calculated_columns = [fifo_result_column, dividend_result_column] + irpf_result_columns
fifo_result_dtype = 'Float64'
dividend_result_dtype = 'Float64' 

# List to store results from each group processing
all_results: List[pd.DataFrame] = []
final_results_df: Optional[pd.DataFrame] = None

# --- Main Processing Logic ---
try:
    # --- Input Validation ---
    if not isinstance(dataset, pd.DataFrame):
         raise TypeError("Input 'dataset' is not a pandas DataFrame.")
    if dataset.empty:
        logger.warning("Input dataset is empty. No calculations will be performed.")
        # Create an empty DataFrame with expected columns based on input if possible
        expected_cols = dataset.columns.tolist() + calculated_columns
        final_results_df = pd.DataFrame(columns=list(dict.fromkeys(expected_cols)), index=dataset.index)

    elif TEST_COLUMN_NAME not in dataset.columns:
        logger.error(f"Mandatory test column '{TEST_COLUMN_NAME}' not found in the input dataset.")
        # Create an error DataFrame similar to the general exception block
        original_cols = list(dataset.columns)
        all_expected_cols = list(dict.fromkeys(original_cols + calculated_columns))
        final_results_df = pd.DataFrame(index=dataset.index, columns=all_expected_cols)
        for col in all_expected_cols:
             if col not in final_results_df.columns: final_results_df[col] = pd.NA
             elif final_results_df[col].isnull().all(): final_results_df[col] = pd.NA
        final_results_df['Error'] = f"Missing test column: {TEST_COLUMN_NAME}"

    else:
        # --- Grouped Processing ---
        # Group by the test column AND the ticker for calculation context
        # If calculations are strictly per-test regardless of ticker, remove TICKER here.
        grouping_columns = [TEST_COLUMN_NAME]
        if TICKER in dataset.columns:
            grouping_columns.append(TICKER)
        else:
            logger.warning(f"Column '{TICKER}' not found. Grouping only by '{TEST_COLUMN_NAME}'. Calculations might mix tickers within a test.")

        logger.info(f"Starting processing, grouping by: {grouping_columns}")
        grouped_data = dataset.groupby(grouping_columns, observed=True, group_keys=False) # group_keys=False avoids adding group keys as index

        logger.info(f"Grupos: {grouped_data.size()}")
        for group_keys, group_df in grouped_data:
            # Ensure group_keys is a tuple for consistent logging
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            group_id_str = ", ".join(map(str, group_keys))
            logger.info(f"--- Processing Group: {group_id_str} ({len(group_df)} rows) ---")

            try:
                # --- Calculations for the current group ---
                # 1. Sort the group data
                if DATETIME in group_df.columns and TRANSACTION_TYPE in group_df.columns:
                    dataset_sorted = group_df.sort_values(
                        by=[DATETIME, TRANSACTION_TYPE],
                        ascending=[True, True],
                        key=lambda col: col.map({'buy': 0, 'sell': 1}) if col.name == TRANSACTION_TYPE else col
                    ).copy() # Use copy to avoid SettingWithCopyWarning on the slice
                    logger.debug(f"Group {group_id_str}: Sorted by Date and Transaction Type.")
                else:
                    logger.warning(f"Group {group_id_str}: Missing '{DATETIME}' or '{TRANSACTION_TYPE}'. Cannot guarantee order.")
                    dataset_sorted = group_df.copy() # Use copy here as well

                # 2. Initialize TransactionManager for the group
                tm = TransactionManager(dataset_sorted)
                logger.debug(f"Group {group_id_str}: Initialized TransactionManager.")

                # 3. Register Calculators (Create new instances for each group for safety)
                fifo_calculator = FIFOCalculator()
                tm.register_calculation(
                    calculator=fifo_calculator,
                    column=fifo_result_column,
                    dtype=fifo_result_dtype
                )
                logger.debug(f"Group {group_id_str}: Registered FIFOCalculator.")

                dividend_calculator = DividendCalculator(dataset_sorted)
                tm.register_calculation(
                    calculator=dividend_calculator,
                    column=dividend_result_column, # Use the constant
                    dtype=dividend_result_dtype
                )
                logger.debug(f"Group {group_id_str}: Registered DividendCalculator.")

                irpf_calculator = IrpfEarningsCalculator()
                tm.register_calculation(
                    calculator=irpf_calculator
                )                
                logger.debug(f"Group {group_id_str}: Registered IrpfEarningsCalculator.")

                # 4. Process calculations for the group
                logger.info(f"Group {group_id_str}: Processing calculations...")
                tm.process_all()
                logger.info(f"Group {group_id_str}: Finished processing calculations.")

                # 5. Store the results for this group
                group_results_df = tm.transactions.copy() # Get the processed data
                all_results.append(group_results_df)
                logger.info(f"Group {group_id_str}: Results stored.")

            except Exception as group_e:
                logger.error(f"Error processing group {group_id_str}: {group_e}", exc_info=True)
                # Create an error DataFrame for this specific group
                original_cols = list(group_df.columns)
                all_expected_cols = list(dict.fromkeys(original_cols + calculated_columns))
                error_df = pd.DataFrame(index=group_df.index, columns=all_expected_cols)
                # Fill original columns with original data if possible
                for col in original_cols:
                    if col in group_df:
                        error_df[col] = group_df[col]
                # Fill calculated columns with NA
                for col in calculated_columns:
                     error_df[col] = pd.NA
                     # Try setting dtype for known result columns if they exist
                     if col in error_df.columns:
                         try:
                             if col == fifo_result_column: error_df[col] = error_df[col].astype(fifo_result_dtype)
                             elif col == dividend_result_column: error_df[col] = error_df[col].astype(dividend_result_dtype)
                             # IRPF columns are already Float64 by default in the calculator
                         except Exception as dtype_e:
                             logger.warning(f"Could not set dtype for {col} in error df: {dtype_e}")

                error_df['Error'] = f"Error in group {group_id_str}: {str(group_e)}"
                all_results.append(error_df)
                logger.info(f"Group {group_id_str}: Created error DataFrame.")

        # --- Combine results after the loop ---
        if all_results:
            logger.info("Concatenating results from all processed groups...")
            # Concatenate results from all groups
            final_results_df = pd.concat(all_results)
            # Restore the original index order from the input dataset
            final_results_df = final_results_df.reindex(dataset.index)
            logger.info("Final results DataFrame created and reindexed successfully.")
        elif final_results_df is None: # Only if not already handled by empty input case
             logger.warning("No groups were processed (e.g., empty groups after filtering). Creating empty DataFrame.")
             expected_cols = dataset.columns.tolist() + calculated_columns
             final_results_df = pd.DataFrame(columns=list(dict.fromkeys(expected_cols)), index=dataset.index)


except Exception as e:
    logger.error(f"Critical error during script execution: {e}", exc_info=True)
    # Create DataFrame vacío con columnas originales (si es posible) + calculadas + error
    try:
        original_cols = list(dataset.columns)
        input_index = dataset.index
    except Exception as fallback_e:
        logger.error(f"Could not access original dataset columns/index in fallback error handler: {fallback_e}")
        original_cols = []
        input_index = None # Cannot guarantee index preservation

    all_expected_cols = list(dict.fromkeys(original_cols + calculated_columns))

    # Use input_index if available, otherwise, it will be None (default range index)
    final_results_df = pd.DataFrame(index=input_index, columns=all_expected_cols)

    # Rellenar con NA por si acaso
    for col in all_expected_cols:
        if col not in final_results_df.columns: # Should not happen, but safety check
             final_results_df[col] = pd.NA
        # Ensure columns exist before trying to fill NA (they should, based on columns=...)
        elif final_results_df[col].isnull().all():
             final_results_df[col] = pd.NA # Ensure it's pd.NA for consistency
             # Try setting dtype for known result columns if they exist
             if col in final_results_df.columns:
                 try:
                     if col == fifo_result_column: final_results_df[col] = final_results_df[col].astype(fifo_result_dtype)
                     elif col == dividend_result_column: final_results_df[col] = final_results_df[col].astype(dividend_result_dtype)
                 except Exception as dtype_e:
                     logger.warning(f"Could not set dtype for {col} in final error df: {dtype_e}")


    final_results_df['Error'] = f"Critical error: {str(e)}"
    logger.info("Created final error DataFrame due to critical failure.")


# --- Final Assignment ---
result = final_results_df if final_results_df is not None else pd.DataFrame()
