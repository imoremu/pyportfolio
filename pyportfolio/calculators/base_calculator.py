from abc import ABC, abstractmethod
from typing import Any, Sequence, Union
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
    def calculate(self, row: pd.Series) -> Union[Any, Sequence[Any]]:
        """
        Calculates the value(s) for the given row.

        If calculating for a single column, returns the calculated value or None.
        If calculating for multiple columns, returns a sequence (e.g., tuple, list)
        of values corresponding to the registered columns, in the same order.
        If no calculation applies for this row for any of the intended columns,
        it might return None (if registered for one column) or a sequence
        of Nones (if registered for multiple columns), depending on implementation needs.

        :param row: A Pandas Series representing a single row of the DataFrame.
        :return: The calculated value or a sequence of calculated values, or None/sequence of Nones.
        """
        pass
