import pandas as pd
from typing import Any
from .base_calculator import BaseCalculator


class DividendCalculator(BaseCalculator):
    """
    Calculator that returns the total dividend if the transaction type is 'dividend'.
    Otherwise, returns None.
    """

    def calculate(self, row: pd.Series) -> Any:
        if row.get("Transaction Type", "").lower() != "dividend":
            return None
        
        # Assuming the row has a 'Dividends' key indicating the dividend amount.
        return row.get("Dividends", 0)