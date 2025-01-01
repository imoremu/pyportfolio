import pytest
import pandas as pd
from pyportfolio.calculators.dividend_calculator import DividendCalculator


def test_dividend_calculator():
    transactions = pd.DataFrame([
        {"Transaction Type": "dividend", "Dividends": 100},
        {"Transaction Type": "buy", "Dividends": 200},  # Should be ignored by the calculator
    ])
    calc = DividendCalculator(transactions)

    assert calc.calculate(transactions.iloc[0]) == 100
    # The second transaction is a 'buy', so the dividend calculator should return None
    assert calc.calculate(transactions.iloc[1]) is None