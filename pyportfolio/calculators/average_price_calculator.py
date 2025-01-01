import pandas as pd
from typing import Any
from .base_calculator import BaseCalculator


class AveragePriceCalculator(BaseCalculator):
    """
    Calculator that determines the average share price after each buy.
    If the row is not a 'buy' transaction, it returns None.
    """

    def __init__(self, transactions: pd.DataFrame):
        super().__init__(transactions)
        self.current_total_shares = 0.0
        self.current_total_cost = 0.0

    def calculate(self, row: pd.Series) -> Any:
        """
        If the transaction is 'buy', calculate the new average price.
        Otherwise, return None.
        """
        if row.get("Transaction Type", "").lower() != "buy":
            return None

        shares_bought = row["Shares Bought"]
        share_price = row["Share Price"]

        self.current_total_cost += shares_bought * share_price
        self.current_total_shares += shares_bought

        if self.current_total_shares > 0:
            avg_price = self.current_total_cost / self.current_total_shares
        else:
            avg_price = 0

        return avg_price
