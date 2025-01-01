import pandas as pd
from typing import Dict, Optional, Tuple
from pyportfolio.calculators.base_calculator import BaseCalculator

class TransactionManager:
    """
    Generic transaction manager. Allows registering multiple calculations
    for specific columns, then iterates over the DataFrame to populate them.
    Optionally, each calculation can specify a target dtype for the resulting column.
    """

    def __init__(self, transactions: pd.DataFrame):
        """
        Initializes the TransactionManager with a Pandas DataFrame.

        Parameters:
        -----------
        transactions : pd.DataFrame
            A DataFrame representing the transaction data.
        """
        self.transactions = transactions
        # A dict mapping: column_name -> (calculator_object, optional_dtype)
        self.calculations: Dict[str, Tuple[BaseCalculator, Optional[str]]] = {}

    def register_calculation(
        self,
        column: str,
        calculator: BaseCalculator,
        dtype: Optional[str] = None
    ):
        """
        Registers a new calculation for a specific column, with an optional dtype.

        Parameters:
        -----------
        column : str
            The name of the column where the calculation’s result will be stored.
        calculator : BaseCalculator
            An object implementing the BaseCalculator interface.
        dtype : Optional[str]
            If provided, we will convert the column to this dtype after the
            calculation is applied. For example, "object" to preserve literal `None`,
            or "float" to allow numeric operations with NaN for missing values.
        """
        self.calculations[column] = (calculator, dtype)

    def process_all(self):
        """
        Applies each registered calculation to each row in the DataFrame.
        If desired_dtype is specified, attempts to convert the column to that dtype.
        """
        for column, (calculator, desired_dtype) in self.calculations.items():
            # Initialize the column with the desired dtype before applying calculations
            if desired_dtype is not None:
                self.transactions[column] = pd.Series(index=self.transactions.index, dtype=desired_dtype)

            # Apply the calculator to populate the column
            self.transactions[column] = self.transactions.apply(
                lambda row: calculator.calculate(row), axis=1
            )
