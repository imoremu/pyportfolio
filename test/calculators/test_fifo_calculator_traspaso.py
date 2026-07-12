"""
Tests for fund-to-fund traspaso (transfer) handling in FIFOCalculator.

A traspaso is modelled as its own transaction type (TYPE_TRANSFER):
  - Traspaso de ENTRADA: positive SHARES -> adds a FIFO lot (like a buy), using the
    row's SHARE_PRICE as the (carried) cost basis per share.
  - Traspaso de SALIDA:  negative SHARES -> consumes FIFO lots (so the origin fund
    reduces its available shares) but realises NO gain (deferred taxation),
    leaving RESULT_FIFO_GAIN_LOSS as NA.
"""

import pytest
import pandas as pd
import numpy as np

from pyportfolio.calculators.fifo_calculator import (
    FIFOCalculator,
    RESULT_FIFO_GAIN_LOSS,
    _INTERNAL_FIFO_AVAILABLE_SHARES,
)
from pyportfolio.columns import (
    DATETIME,
    TICKER,
    TRANSACTION_TYPE,
    SHARES,
    SHARE_PRICE,
    COMISION,
    TYPE_BUY,
    TYPE_SELL,
    TYPE_TRANSFER,
)


def _run(data):
    df = pd.DataFrame(data)
    df[DATETIME] = pd.to_datetime(df[DATETIME], format="mixed", dayfirst=True)
    df[SHARES] = df[SHARES].astype(float)
    df[SHARE_PRICE] = df[SHARE_PRICE].astype(float)
    if COMISION not in df.columns:
        df[COMISION] = 0.0
    df[COMISION] = df[COMISION].astype(float)
    df = df.sort_values(by=[TICKER, DATETIME, TRANSACTION_TYPE]).reset_index(drop=True)
    return FIFOCalculator().calculate_table(df.copy())


def test_traspaso_entrada_makes_shares_available_for_sell():
    """The reported bug: buy 9 + traspaso-in 40 + traspaso-in 50, then sell 40.

    Before the fix this raised ValueError ('only 9 available'). Now the two
    transfer-ins count as available shares, so the sale succeeds.
    """
    data = {
        DATETIME: ["2023-01-10", "2023-02-10", "2023-03-10", "2023-04-10"],
        TICKER: ["FI_DEST", "FI_DEST", "FI_DEST", "FI_DEST"],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_TRANSFER, TYPE_TRANSFER, TYPE_SELL],
        SHARES: [9, 40, 50, -40],
        SHARE_PRICE: [10, 11, 12, 15],
        COMISION: [0, 0, 0, 0],
    }
    result = _run(data)

    # FIFO for the sale of 40: 9 @10 (=90) + 31 @11 (=341) = 431 cost
    # proceeds = 40 * 15 = 600  ->  gain = 600 - 431 = 169
    sell_row = result[result[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]
    assert sell_row[RESULT_FIFO_GAIN_LOSS] == pytest.approx(169.0)

    # Transfer-in rows themselves realise no gain.
    assert result[result[TRANSACTION_TYPE] == TYPE_TRANSFER][RESULT_FIFO_GAIN_LOSS].isna().all()

    # Remaining available: (40-31) + 50 = 59
    remaining = result[_INTERNAL_FIFO_AVAILABLE_SHARES].sum()
    assert remaining == pytest.approx(59.0)


def test_traspaso_salida_consumes_lots_without_realised_gain():
    """Traspaso-out consumes FIFO lots (deferred) but leaves gain as NA;
    a later real sale sees the reduced share balance."""
    data = {
        DATETIME: ["2023-01-10", "2023-02-10", "2023-03-10"],
        TICKER: ["FI_ORIG", "FI_ORIG", "FI_ORIG"],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_TRANSFER, TYPE_SELL],
        SHARES: [100, -40, -30],
        SHARE_PRICE: [10, 10, 20],
        COMISION: [0, 0, 0],
    }
    result = _run(data)

    # Traspaso-out realises no gain (deferred).
    out_row = result[result[TRANSACTION_TYPE] == TYPE_TRANSFER].iloc[0]
    assert pd.isna(out_row[RESULT_FIFO_GAIN_LOSS])

    # Sell 30 after the transfer-out: consumes 30 from remaining 60 @10.
    # proceeds = 30*20 = 600, cost = 30*10 = 300 -> gain = 300
    sell_row = result[result[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]
    assert sell_row[RESULT_FIFO_GAIN_LOSS] == pytest.approx(300.0)

    # Remaining available: 100 - 40 - 30 = 30
    assert result[_INTERNAL_FIFO_AVAILABLE_SHARES].sum() == pytest.approx(30.0)


def test_traspaso_salida_overselling_raises():
    """Transferring out more than available raises, like an oversell."""
    data = {
        DATETIME: ["2023-01-10", "2023-02-10"],
        TICKER: ["FI_ORIG", "FI_ORIG"],
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_TRANSFER],
        SHARES: [10, -20],
        SHARE_PRICE: [10, 10],
        COMISION: [0, 0],
    }
    with pytest.raises(ValueError) as exc:
        _run(data)
    assert "10.0000 are available" in str(exc.value)


def test_traspaso_entrada_carried_cost_used_as_basis():
    """A transfer-in lot uses its SHARE_PRICE as cost basis; selling it later
    computes gain against that basis (carried cost, once the pre-pass sets it)."""
    data = {
        DATETIME: ["2023-01-10", "2023-02-10"],
        TICKER: ["FI_DEST", "FI_DEST"],
        TRANSACTION_TYPE: [TYPE_TRANSFER, TYPE_SELL],
        SHARES: [100, -100],
        SHARE_PRICE: [50, 120],  # carried cost 50/share; sold at 120
        COMISION: [0, 0],
    }
    result = _run(data)
    sell_row = result[result[TRANSACTION_TYPE] == TYPE_SELL].iloc[0]
    # proceeds 100*120=12000, cost 100*50=5000 -> gain 7000
    assert sell_row[RESULT_FIFO_GAIN_LOSS] == pytest.approx(7000.0)
