import pandas as pd
from typing import Any

from pyportfolio.columns import (
    AVAILABLE_SHARES,
    SHARE_PRICE,
    SHARES,
    TRANSACTION_TYPE,
    TYPE_BUY,
    TYPE_SELL,
)
from .base_calculator import BaseCalculator


class FIFOCalculator(BaseCalculator):
    """
    Calculator that applies FIFO logic to 'sell' transactions,
    strictly limited to shares bought (and still available) prior to this
    sell transaction in the chronological list.

    Raises a ValueError if the requested shares exceed what is available
    from previous buys.
    """

    def __init__(self, transactions: pd.DataFrame):
        super().__init__(transactions)
        # Initialize "Available Shares" for every 'buy' row
        self.transactions.loc[
            self.transactions[TRANSACTION_TYPE].str.lower() == TYPE_BUY, AVAILABLE_SHARES
        ] = self.transactions.loc[
            self.transactions[TRANSACTION_TYPE].str.lower() == TYPE_BUY, SHARES
        ]

    def calculate(self, row: pd.Series) -> Any:
        """
        If this row is a 'sell' transaction, ensure the shares to sell
        do not exceed the total available among previous buys. Then
        consume those shares in FIFO order among the earlier transactions.

        Returns the FIFO gain if successful, or None if this row is not 'sell'.
        """
        # 1) Check if it's a 'sell' transaction
        if row[TRANSACTION_TYPE].lower() != TYPE_SELL:
            return None

        # 2) Get the index of this 'sell' transaction
        sell_index = row.name
        shares_to_sell = row[SHARES]
        sell_price = row[SHARE_PRICE]

        # 3) Calculate the total available shares before this transaction
        previous_buys = self.transactions.loc[
            (self.transactions.index < sell_index)
            & (self.transactions[TRANSACTION_TYPE].str.lower() == TYPE_BUY)
        ]
        total_available_before = previous_buys[AVAILABLE_SHARES].sum()

        # 4) If there aren't enough shares, raise an exception
        if shares_to_sell > total_available_before:
            raise ValueError(
                f"Cannot sell {shares_to_sell} shares; "
                f"only {total_available_before} are available in previous buys."
            )

        # 5) Perform FIFO consumption
        total_cost = 0.0
        remaining_to_sell = shares_to_sell

        for idx, prev_row in previous_buys.iterrows():
            avail = prev_row[AVAILABLE_SHARES]
            if avail > 0 and remaining_to_sell > 0:
                used = min(avail, remaining_to_sell)
                total_cost += used * prev_row[SHARE_PRICE]
                # Update available shares for this buy transaction
                self.transactions.at[idx, AVAILABLE_SHARES] = avail - used
                remaining_to_sell -= used

            if remaining_to_sell <= 0:
                break

        # 6) Calculate FIFO gain
        fifo_gain = (sell_price * shares_to_sell) - total_cost

        return fifo_gain
