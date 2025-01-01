from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

class BaseCalculator(ABC):
    """
    Abstract base class (interface) for calculators.
    The entire transaction DataFrame can be injected in the constructor if needed.
    """

    def __init__(self, transactions: pd.DataFrame):
        """
        :param transactions: The complete DataFrame of transactions,
                             in case a global state is needed.
        """
        self.transactions = transactions

    @abstractmethod
    def calculate(self, row: pd.Series) -> Any:
        """
        Calculates the value for the given row.
        Returns None if no calculation applies.

        :param row: A Pandas Series representing a single row of the DataFrame.
        :return: The calculated value for the given row, or None.
        """
        pass
