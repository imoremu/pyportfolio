import pandas as pd
from typing import Dict, Optional, Tuple, List, Union, Sequence, Any # Added List, Union, Sequence, Any
from pyportfolio.calculators.base_calculator import BaseCalculator

class TransactionManager:
    """
    Generic transaction manager. Allows registering multiple calculations
    for specific columns, then iterates over the DataFrame to populate them.
    A single calculator can be registered to populate one or multiple columns.
    Optionally, each calculation can specify target dtypes for the resulting columns.
    """

    def __init__(self, transactions: pd.DataFrame):
        """
        Initializes the TransactionManager with a Pandas DataFrame.

        Parameters:
        -----------
        transactions : pd.DataFrame
            A DataFrame representing the transaction data.
        """
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame")
        self.transactions = transactions
        # Store registrations as a list to maintain order and handle multi-column calcs
        # Structure: [(calculator, [col1, col2,...], [dtype1, dtype2,...]), ...]
        self._registrations: List[Tuple[BaseCalculator, List[str], List[Optional[str]]]] = []


    def register_calculation(
        self,
        column: Union[str, Sequence[str]],
        calculator: BaseCalculator,
        dtype: Union[Optional[str], Sequence[Optional[str]]] = None
    ):
        """
        Registers a calculation for one or more columns using a single calculator.

        Parameters:
        -----------
        column : Union[str, Sequence[str]]
            The name of the column or a sequence (list/tuple) of column names
            where the calculation's result(s) will be stored.
        calculator : BaseCalculator
            An object implementing the BaseCalculator interface. Its `calculate`
            method should return a single value if `column` is a string, or a
            sequence of values matching the order of `column` if it's a sequence.
        dtype : Union[Optional[str], Sequence[Optional[str]]]
            Optional target dtype(s) for the column(s).
            - If `column` is a string, `dtype` should be None or a string.
            - If `column` is a sequence:
                - `dtype` can be None (no conversion for any column).
                - `dtype` can be a single string (applied to all columns).
                - `dtype` can be a sequence of None/string matching the length
                  and order of `column`.
        """
        if not isinstance(calculator, BaseCalculator):
             raise TypeError("calculator must be an instance of BaseCalculator or its subclass")

        columns_list: List[str]
        dtypes_list: List[Optional[str]]

        # --- Normalize columns to a list ---
        if isinstance(column, str):
            columns_list = [column]
        elif isinstance(column, (list, tuple)):
            if not all(isinstance(c, str) for c in column):
                raise TypeError("If 'column' is a sequence, all elements must be strings.")
            if not column:
                raise ValueError("If 'column' is a sequence, it cannot be empty.")
            columns_list = list(column)
        else:
            raise TypeError("column must be a string or a sequence of strings")

        num_columns = len(columns_list)

        # --- Normalize dtypes to a list matching columns ---
        if isinstance(dtype, str) or dtype is None:
            # Apply single dtype to all columns
            dtypes_list = [dtype] * num_columns
        elif isinstance(dtype, (list, tuple)):
            if len(dtype) != num_columns:
                raise ValueError(f"If 'dtype' is a sequence, its length ({len(dtype)}) must match the number of columns ({num_columns}).")
            if not all(isinstance(d, (str, type(None))) for d in dtype):
                 raise TypeError("If 'dtype' is a sequence, all elements must be strings or None.")
            dtypes_list = list(dtype)
        else:
            raise TypeError("dtype must be None, a string, or a sequence of None/string")

        # --- Store the registration ---
        self._registrations.append((calculator, columns_list, dtypes_list))


    def process_all(self):
        """
        Applies each registered calculation to the DataFrame.
        Handles both single and multi-column calculations.
        If target dtypes were specified, attempts to convert the columns.
        """
        if self.transactions.empty:
            # Handle empty DataFrame: Initialize columns if needed, but skip apply
            for _, columns, dtypes in self._registrations:
                for col, dt in zip(columns, dtypes):
                    if col not in self.transactions.columns:
                        # Initialize with NaN or None based on dtype if possible
                        init_val = pd.NA if dt == 'float' or pd.api.types.is_numeric_dtype(dt) else None
                        self.transactions[col] = pd.Series(init_val, index=self.transactions.index, dtype=dt)
            return # Nothing to process

        for calculator, columns, dtypes in self._registrations:
            # --- Apply the calculation ---
            # apply returns a Series. If the calculator returns single values,
            # it's a Series of values. If the calculator returns sequences,
            # it's a Series of sequences (tuples/lists).
            results = self.transactions.apply(
                lambda row: calculator.calculate(row), axis=1
            )

            # --- Assign results to DataFrame columns ---
            if len(columns) == 1:
                # Single column assignment
                col_name = columns[0]
                target_dtype = dtypes[0]
                # Assign directly, pandas handles type inference initially
                self.transactions[col_name] = results
                # Apply desired dtype if specified
                if target_dtype is not None:
                    try:
                        self.transactions[col_name] = self.transactions[col_name].astype(target_dtype)
                    except Exception as e:
                        print(f"Warning: Could not convert column '{col_name}' to dtype '{target_dtype}'. Error: {e}")

            else:
                # Multiple column assignment
                # Check if results are sequences of the correct length
                first_valid_result = next((item for item in results if item is not None), None) # Find first non-None result to check structure
                if first_valid_result is not None and (not isinstance(first_valid_result, Sequence) or len(first_valid_result) != len(columns)):
                     raise ValueError(f"Calculator {type(calculator).__name__} was expected to return a sequence of {len(columns)} items for columns {columns}, but returned: {first_valid_result}")

                # Create a temporary DataFrame from the results (Series of sequences)
                # Handle potential Nones returned by the calculator for rows where calculation doesn't apply
                # We need to ensure these Nones become rows of Nones/NaNs in the temp DataFrame
                def _safe_tolist(item):
                    if isinstance(item, Sequence) and len(item) == len(columns):
                        return list(item)
                    # If item is None or not a sequence of correct length, return list of Nones
                    return [None] * len(columns)

                try:
                    temp_df = pd.DataFrame(
                        results.apply(_safe_tolist).tolist(), # Convert Series of sequences to list of lists
                        index=self.transactions.index,
                        columns=columns
                    )
                except Exception as e:
                     # Provide more context on failure
                     print(f"Error creating temporary DataFrame for columns {columns} from calculator {type(calculator).__name__}. Check calculator's return values.")
                     raise e


                # Assign columns from the temporary DataFrame
                for i, col_name in enumerate(columns):
                    target_dtype = dtypes[i]
                    self.transactions[col_name] = temp_df[col_name]
                    if target_dtype is not None:
                        try:
                            # Use convert_dtypes for better nullable type handling or astype
                            # self.transactions[col_name] = self.transactions[col_name].convert_dtypes() # Option 1
                            self.transactions[col_name] = self.transactions[col_name].astype(target_dtype) # Option 2 (original)
                        except Exception as e:
                            print(f"Warning: Could not convert column '{col_name}' to dtype '{target_dtype}'. Error: {e}")

