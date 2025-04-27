# pyportfolio/transaction_manager.py

import pandas as pd
from typing import Dict, Optional, Tuple, List, Union, Sequence, Any, Callable
# Import base classes
from pyportfolio.calculators.base_calculator import BaseRowCalculator, BaseTableCalculator

import logging

logger = logging.getLogger(__name__)

class TransactionManager:
    """
    Generic transaction manager. Allows registering multiple calculations
    for specific columns or entire DataFrame transformations.

    Calculators can operate in two modes:
    1. Row-wise (inheriting from BaseRowCalculator): Implement `calculate_row(self, row: pd.Series)`.
       Registered with specific output column(s) and optional dtype(s).
    2. Table-wise (inheriting from BaseTableCalculator): Implement `calculate_table(self, df: pd.DataFrame)`.
       Receives the current state of the DataFrame and returns a DataFrame
       containing the calculated/updated columns. The returned DataFrame's index
       MUST align with the input DataFrame's index. When registering a table-wise
       calculator, the `column` and `dtype` arguments are ignored.
    """

    def __init__(self, transactions: pd.DataFrame):
        """
        Initializes the TransactionManager with a Pandas DataFrame.

        Parameters:
        -----------
        transactions : pd.DataFrame
            A DataFrame representing the transaction data. It will be modified in place.
        """
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame")
        self.transactions = transactions        
        self._registrations: List[Tuple[Union[BaseRowCalculator, BaseTableCalculator], Optional[List[str]], Optional[List[Optional[str]]], str]] = []

    def _validate_row_params(self, column, dtype) -> Tuple[List[str], List[Optional[str]]]:
        """ Validates 'column' and 'dtype' arguments for row-wise calculators. """
        columns_list: List[str]
        dtypes_list: List[Optional[str]]

        if isinstance(column, str):
            columns_list = [column]
        elif isinstance(column, (list, tuple)):
            if not all(isinstance(c, str) for c in column):
                raise TypeError("If 'column' is a sequence (for row-wise), all elements must be strings.")
            if not column:
                raise ValueError("If 'column' is a sequence (for row-wise), it cannot be empty.")
            columns_list = list(column)
        else:
            raise TypeError("For row-wise calculators, 'column' must be a string or a sequence of strings")

        num_columns = len(columns_list)

        if isinstance(dtype, str) or dtype is None:
            dtypes_list = [dtype] * num_columns
        elif isinstance(dtype, (list, tuple)):
            if len(dtype) != num_columns:
                raise ValueError(f"If 'dtype' is a sequence (for row-wise), its length ({len(dtype)}) must match the number of columns ({num_columns}).")
            if not all(isinstance(d, (str, type(None))) for d in dtype):
                 raise TypeError("If 'dtype' is a sequence (for row-wise), all elements must be strings or None.")
            dtypes_list = list(dtype)
        else:
            raise TypeError("For row-wise calculators, 'dtype' must be None, a string, or a sequence of None/string")

        return columns_list, dtypes_list

    def register_calculation(
        self,
        calculator: Union[BaseRowCalculator, BaseTableCalculator],
        column: Union[str, Sequence[str], None] = None,
        dtype: Union[Optional[str], Sequence[Optional[str]]] = None
    ):
        """
        Registers a calculation to be performed, detecting if it's row-wise or table-wise.

        Parameters:
        -----------
        calculator : Union[BaseRowCalculator, BaseTableCalculator]
            An object inheriting from BaseRowCalculator or BaseTableCalculator.
        column : Union[str, Sequence[str], None], optional
            Required for row-wise calculators (BaseRowCalculator). The name of the
            column or a sequence of column names where the result(s) will be stored.
            Ignored for table-wise calculators.
        dtype : Union[Optional[str], Sequence[Optional[str]]], optional
            Optional target dtype(s) for the column(s) for row-wise calculators.
            Ignored for table-wise calculators.
        """
        if isinstance(calculator, BaseTableCalculator):
            mode = 'table'
            logger.debug(f"Registering {type(calculator).__name__} as table-wise calculator.")
            if column is not None or dtype is not None:
                logger.warning(f"'column' and 'dtype' arguments are ignored when registering a BaseTableCalculator ({type(calculator).__name__}).")
            self._registrations.append((calculator, None, None, mode))

        elif isinstance(calculator, BaseRowCalculator):
            mode = 'row'
            if column is None:
                raise ValueError("Argument 'column' is required for row-wise calculators (BaseRowCalculator).")
            columns_list, dtypes_list = self._validate_row_params(column, dtype)
            logger.debug(f"Registering {type(calculator).__name__} as row-wise calculator for column(s): {columns_list}.")
            self._registrations.append((calculator, columns_list, dtypes_list, mode))
        else:
            raise TypeError(f"Calculator must inherit from BaseRowCalculator or BaseTableCalculator, got {type(calculator).__name__}")


    def process_all(self):
        """
        Applies each registered calculation to the DataFrame based on its mode
        (row-wise or table-wise). Modifies the internal DataFrame in place.
        """
        is_initially_empty = self.transactions.empty

        if is_initially_empty:
            logger.info("DataFrame is initially empty. Initializing columns for registered row-wise calculators.")
            for _, columns, dtypes, mode in self._registrations:
                if mode == 'row' and columns:
                    effective_dtypes = dtypes if dtypes else [None] * len(columns)
                    for col, dt in zip(columns, effective_dtypes):
                        if col not in self.transactions.columns:
                            na_val = pd.NA if dt and (pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_datetime64_any_dtype(dt) or pd.api.types.is_bool_dtype(dt)) else None
                            self.transactions[col] = pd.Series(na_val, index=self.transactions.index, dtype=dt)
            logger.debug(f"Columns after row-wise init for empty df: {self.transactions.columns.tolist()}")


        for calculator, columns, dtypes, mode in self._registrations:
            calc_name = type(calculator).__name__
            try:
                if mode == 'table':
                    if not isinstance(calculator, BaseTableCalculator):
                         logger.error(f"Internal Error: Expected BaseTableCalculator, got {calc_name}")
                         continue

                    logger.info(f"Processing table-wise calculation with {calc_name}...")
                    result_df = calculator.calculate_table(self.transactions)

                    if not isinstance(result_df, pd.DataFrame):
                        raise TypeError(f"Table calculator {calc_name} must return a pandas DataFrame, got {type(result_df)}.")

                    # Check index alignment (optional but recommended)
                    if not is_initially_empty and not self.transactions.index.equals(result_df.index):
                         logger.warning(f"Index of DataFrame returned by table calculator {calc_name} does not match the original index. Results might be misaligned.")
                         # Attempt to reindex - might introduce NaNs if indices differ significantly
                         result_df = result_df.reindex(self.transactions.index)
                    elif is_initially_empty and not result_df.empty and not self.transactions.index.equals(result_df.index):
                         logger.warning(f"Index of DataFrame returned by table calculator {calc_name} does not match original empty index. Reindexing.")
                         result_df = result_df.reindex(self.transactions.index)

                    logger.debug(f"Result DF data {calc_name}: {result_df.to_string()}")
                    
                    for col in result_df.columns:
                        if col in self.transactions.columns:
                            logger.debug(f"Table calculator {calc_name} overwriting column '{col}'.")
                        if is_initially_empty and col not in self.transactions.columns:
                             self.transactions[col] = pd.Series(result_df[col], index=self.transactions.index)
                        else:
                             self.transactions[col] = result_df[col]

                    logger.info(f"Finished table-wise calculation with {calc_name}.")
                    logger.debug(f"Result Transactions data {calc_name}: {self.transactions.to_string()}")

                elif mode == 'row':
                    if not isinstance(calculator, BaseRowCalculator):
                         logger.error(f"Internal Error: Expected BaseRowCalculator, got {calc_name}")
                         continue

                    if is_initially_empty:
                         logger.debug(f"Skipping row-wise calculation processing for {calc_name} as DataFrame was initially empty.")
                         continue

                    logger.info(f"Processing row-wise calculation for columns {columns} with {calc_name}...")
                    results = self.transactions.apply(
                        lambda row: calculator.calculate_row(row), axis=1
                    )

                    if len(columns) == 1:
                        col_name = columns[0]
                        target_dtype = dtypes[0] if dtypes else None
                        self.transactions[col_name] = results
                        if target_dtype is not None:
                            try:
                                # Use convert_dtypes for better nullable type inference if possible
                                if pd.__version__ >= "1.0.0":
                                     self.transactions[col_name] = self.transactions[col_name].convert_dtypes()
                                # Then attempt specific cast if provided
                                self.transactions[col_name] = self.transactions[col_name].astype(target_dtype)
                            except Exception as e:
                                logger.warning(f"Warning: Could not convert column '{col_name}' to dtype '{target_dtype}'. Error: {e}")

                    else:
                        first_valid_result = next((item for item in results if item is not None), None)
                        if first_valid_result is not None and (not isinstance(first_valid_result, Sequence) or len(first_valid_result) != len(columns)):
                             raise ValueError(f"Row-wise calculator {calc_name} was expected to return a sequence of {len(columns)} items for columns {columns}, but returned: {first_valid_result}")

                        # Helper to handle potential Nones or incorrect structures from apply
                        def _safe_tolist(item):
                            if isinstance(item, Sequence) and len(item) == len(columns):
                                return list(item)
                            # Return list of Nones matching expected columns if structure is wrong
                            return [None] * len(columns)

                        try:
                            processed_results = results.apply(_safe_tolist)
                            if not processed_results.empty and isinstance(processed_results.iloc[0], list):
                                temp_df = pd.DataFrame(
                                    processed_results.tolist(),
                                    index=self.transactions.index,
                                    columns=columns
                                )
                            else:
                                temp_df = pd.DataFrame(None, index=self.transactions.index, columns=columns)

                        except Exception as e:
                             logger.error(f"Error creating temporary DataFrame for columns {columns} from row-wise calculator {calc_name}. Check calculator's return values consistency. Error: {e}")
                             raise e

                        for i, col_name in enumerate(columns):
                            target_dtype = dtypes[i] if dtypes else None
                            self.transactions[col_name] = temp_df[col_name]
                            if target_dtype is not None:
                                try:
                                     if pd.__version__ >= "1.0.0":
                                         self.transactions[col_name] = self.transactions[col_name].convert_dtypes()
                                     self.transactions[col_name] = self.transactions[col_name].astype(target_dtype)
                                except Exception as e:
                                    logger.warning(f"Warning: Could not convert column '{col_name}' to dtype '{target_dtype}'. Error: {e}")
                    logger.info(f"Finished row-wise calculation for columns {columns}.")

            except Exception as e:
                 logger.error(f"Error during {mode}-wise calculation with {calc_name}: {e}", exc_info=True)
                 raise e
