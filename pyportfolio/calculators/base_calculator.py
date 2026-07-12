from abc import ABC, abstractmethod
from typing import Any, Sequence, Union
import pandas as pd

# --- Shared FIFO share-quantity tolerance -----------------------------------
# Comparing "shares requested to sell" against "shares available from previous
# buys" must tolerate floating-point noise: available balances are the result of
# subtracting many lot consumptions in a single chronological pass, so their
# accumulated rounding error can exceed a tight absolute epsilon (the former
# 1e-9 produced false "overselling" aborts on perfectly valid data, e.g. selling
# 76.2550 shares when 76.25499998 remained). We use an absolute floor plus a
# magnitude-relative term so the tolerance scales with the balance size, while
# staying far below one real (fractional) share. See bug B-PYP-0001.
_SHARE_ABS_TOL = 1e-6
_SHARE_REL_TOL = 1e-6


def share_tolerance(available: float) -> float:
    """Float-noise tolerance for a share balance of the given magnitude."""
    return max(_SHARE_ABS_TOL, _SHARE_REL_TOL * abs(available))


def is_overselling(requested: float, available: float) -> bool:
    """True only when ``requested`` exceeds ``available`` beyond rounding noise.

    A genuine oversell (requesting clearly more than is available) returns True;
    a request that merely differs from the balance by float-rounding error
    returns False and should be clamped to ``available`` by the caller.
    """
    return requested > available + share_tolerance(available)

class BaseRowCalculator(ABC):
    """
    Abstract base class (interface) for calculators that operate row by row.
    The entire transaction DataFrame can be injected in the constructor if needed.
    """

    def __init__(self, transactions: pd.DataFrame):
        """
        Initializes the row-wise calculator.

        Args:
            transactions: The complete DataFrame of transactions,
                          in case a global state or context is needed.
                          Note: Modifying this DataFrame directly within the
                          calculator might lead to unexpected side effects if
                          multiple calculators are used. Prefer read-only access
                          or operate on copies if modifications are necessary.
        """
        self.transactions = transactions

    @abstractmethod
    def calculate_row(self, row: pd.Series) -> Union[Any, Sequence[Any]]:
        """
        Calculates the value(s) for the given row.

        If calculating for a single column, returns the calculated value or None.
        If calculating for multiple columns, returns a sequence (e.g., tuple, list)
        of values corresponding to the registered columns, in the same order.
        If no calculation applies for this row for any of the intended columns,
        it might return None (if registered for one column) or a sequence
        of Nones (if registered for multiple columns), depending on implementation needs.

        Args:
            row: A Pandas Series representing a single row of the DataFrame being processed.

        Returns:
            The calculated value or a sequence of calculated values, or None/sequence of Nones.
        """
        pass


class BaseTableCalculator(ABC):
    """
    Abstract base class (interface) for calculators that operate on the entire DataFrame at once.
    """

    def __init__(self):
        """
        Initializes the table-wise calculator.
        Typically, these calculators might not need the full initial DataFrame
        in the constructor, as they receive it during the `calculate_table` call.
        However, parameters for the calculation (like window sizes, column names, etc.)
        can be passed here.
        """
        # Constructor can be used for parameters, but often doesn't need the df itself.
        pass

    @abstractmethod
    def calculate_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs calculations on the entire DataFrame and returns a DataFrame
        containing the results.

        The returned DataFrame should ideally contain only the new or updated
        columns calculated by this specific calculator. It MUST have an index
        that aligns perfectly with the input DataFrame `df` to allow the
        TransactionManager to correctly merge the results.

        Args:
            df: The current state of the transaction DataFrame as processed
                by the TransactionManager up to this point.

        Returns:
            A pandas DataFrame containing the calculated columns, with an index
            matching the input DataFrame.
        """
        pass
