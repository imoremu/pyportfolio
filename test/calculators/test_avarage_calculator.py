import pandas as pd
from pyportfolio.calculators.average_price_calculator import AveragePriceCalculator


def test_average_calculator_basic():
    transactions = pd.DataFrame([
        {"Transaction Type": "buy", "Shares Bought": 10, "Share Price": 50},
        {"Transaction Type": "buy", "Shares Bought": 10, "Share Price": 70},
        {"Transaction Type": "sell", "Shares Bought": 5, "Share Price": 80},
    ])
    calc = AveragePriceCalculator(transactions)

    # First buy
    avg1 = calc.calculate_row(transactions.iloc[0])
    # total_cost = 10*50 = 500, total_shares = 10 => average = 50
    assert avg1 == 50

    # Second buy
    avg2 = calc.calculate_row(transactions.iloc[1])
    # total_cost = 500 + (10*70) = 1200, total_shares = 20 => average = 60
    assert avg2 == 60

    # Third transaction is 'sell' => no new average => None
    avg3 = calc.calculate_row(transactions.iloc[2])
    assert avg3 is None
