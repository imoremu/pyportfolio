import pandas as pd
from typing import Any
from .base_calculator import BaseRowCalculator

from pyportfolio.columns import (
    SHARE_PRICE,
    SHARES,
    TRANSACTION_TYPE,
    COMISION,
    TYPE_DIVIDEND
)

class DividendCalculator(BaseRowCalculator):
    """
    Calculator that returns the total dividend if the transaction type is 'dividend'.
    Otherwise, returns None.
    """

    def calculate_row(self, row: pd.Series) -> Any:
        if row.get(TRANSACTION_TYPE, "").lower() != TYPE_DIVIDEND:
            return None
        
        # Return the dividend amount, which is assumed to be: Number of Shares * Shares Price - Comission
        shares = row.get(SHARES, 0)
        share_price = row.get(SHARE_PRICE, 0)
        comission = row.get(COMISION, 0)
        
        return shares * share_price - comission