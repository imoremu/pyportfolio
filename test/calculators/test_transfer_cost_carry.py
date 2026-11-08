"""
Tests for the fund-to-fund traspaso carried-cost pre-pass
(pyportfolio.calculators.transfer_cost_carry.apply_transfer_carried_cost).

The pre-pass rewrites each transfer-in row's SHARE_PRICE with the cost basis carried
over from the origin fund (FIFO), paired via the reciprocal `ID Transacción Espejo`.
"""

import pytest
import pandas as pd

from pyportfolio.calculators.transfer_cost_carry import apply_transfer_carried_cost
from pyportfolio.columns import (
    DATETIME,
    TEST,
    TICKER,
    TRANSACTION_TYPE,
    SHARES,
    SHARE_PRICE,
    COMISION,
    TRANSACTION_ID,
    MIRROR_TRANSACTION_ID,
    TYPE_BUY,
    TYPE_SELL,
    TYPE_TRANSFER,
)

COLUMNS = [DATETIME, TEST, TICKER, TRANSACTION_TYPE, SHARES, SHARE_PRICE, COMISION,
           TRANSACTION_ID, MIRROR_TRANSACTION_ID]


def _df(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df[DATETIME] = pd.to_datetime(df[DATETIME], format="%Y-%m-%d")
    for c in (SHARES, SHARE_PRICE, COMISION):
        df[c] = df[c].astype(float)
    return df


def _price(df, txn_id):
    return df.loc[df[TRANSACTION_ID] == txn_id, SHARE_PRICE].iloc[0]


def test_full_transfer_carries_origin_cost():
    # A: buy 100 @ 50 -> cost 5000. Transfer all to B, which receives 200 @ NAV 100.
    # Carried cost 5000 / 200 = 25 per share.
    df = _df([
        ["2023-01-01", "T", "A", TYPE_BUY,      100,  50, 0, "A_BUY", ""],
        ["2023-02-01", "T", "A", TYPE_TRANSFER, -100, 50, 0, "OUT1",  "IN1"],
        ["2023-02-01", "T", "B", TYPE_TRANSFER,  200, 100, 0, "IN1",  "OUT1"],
    ])
    out = apply_transfer_carried_cost(df)
    assert _price(out, "IN1") == pytest.approx(25.0)
    # Origin rows untouched
    assert _price(out, "A_BUY") == pytest.approx(50.0)


def test_partial_transfer_uses_fifo_cost_of_leaving_shares():
    # A: buy 100 @ 50. Transfer out 40 -> carried 40*50 = 2000. B receives 80 -> 2000/80 = 25.
    df = _df([
        ["2023-01-01", "T", "A", TYPE_BUY,      100, 50, 0, "A_BUY", ""],
        ["2023-02-01", "T", "A", TYPE_TRANSFER, -40, 50, 0, "OUT1",  "IN1"],
        ["2023-02-01", "T", "B", TYPE_TRANSFER,  80, 100, 0, "IN1",  "OUT1"],
    ])
    out = apply_transfer_carried_cost(df)
    assert _price(out, "IN1") == pytest.approx(25.0)


def test_partial_transfer_respects_fifo_lot_order():
    # A: buy 50 @ 10 (lot1), buy 50 @ 20 (lot2). Transfer out 60 -> FIFO consumes
    # 50@10 + 10@20 = 500 + 200 = 700. B receives 100 -> 700/100 = 7.
    df = _df([
        ["2023-01-01", "T", "A", TYPE_BUY,       50, 10, 0, "A_BUY1", ""],
        ["2023-01-05", "T", "A", TYPE_BUY,       50, 20, 0, "A_BUY2", ""],
        ["2023-02-01", "T", "A", TYPE_TRANSFER, -60, 20, 0, "OUT1",   "IN1"],
        ["2023-02-01", "T", "B", TYPE_TRANSFER, 100, 30, 0, "IN1",    "OUT1"],
    ])
    out = apply_transfer_carried_cost(df)
    assert _price(out, "IN1") == pytest.approx(7.0)


def test_chain_a_b_c_carries_cost_through():
    # A buy 100 @ 50 (5000). A->B all: B gets 200 -> 25/sh. B->C all: carried 5000 -> C gets 100 -> 50/sh.
    df = _df([
        ["2023-01-01", "T", "A", TYPE_BUY,      100,  50, 0, "A_BUY", ""],
        ["2023-02-01", "T", "A", TYPE_TRANSFER, -100, 50, 0, "OUT1",  "IN1"],
        ["2023-02-01", "T", "B", TYPE_TRANSFER,  200, 100, 0, "IN1",  "OUT1"],
        ["2023-03-01", "T", "B", TYPE_TRANSFER, -200, 100, 0, "OUT2", "IN2"],
        ["2023-03-01", "T", "C", TYPE_TRANSFER,  100, 200, 0, "IN2",  "OUT2"],
    ])
    out = apply_transfer_carried_cost(df)
    assert _price(out, "IN1") == pytest.approx(25.0)
    assert _price(out, "IN2") == pytest.approx(50.0)


def test_normal_two_day_settlement_gap():
    # The common real case: money leaves the origin on day 1 and arrives at the
    # destination 2 days later. Carried cost 5000 / 200 = 25.
    df = _df([
        ["2023-01-01", "T", "A", TYPE_BUY,      100,  50, 0, "A_BUY", ""],
        ["2023-02-01", "T", "A", TYPE_TRANSFER, -100, 50, 0, "OUT1",  "IN1"],
        ["2023-02-03", "T", "B", TYPE_TRANSFER,  200, 100, 0, "IN1",  "OUT1"],
    ])
    out = apply_transfer_carried_cost(df)
    assert _price(out, "IN1") == pytest.approx(25.0)


def test_pairing_works_when_entrada_dated_before_salida():
    # The transfer-in is dated 2023-02-05, BEFORE its transfer-out on 2023-02-10.
    # Pairing is by ID (not date) and ordering follows the pairing, so the carried cost
    # is still applied correctly (5000 / 200 = 25).
    df = _df([
        ["2023-01-01", "T", "A", TYPE_BUY,      100,  50, 0, "A_BUY", ""],
        ["2023-02-10", "T", "A", TYPE_TRANSFER, -100, 50, 0, "OUT1",  "IN1"],
        ["2023-02-05", "T", "B", TYPE_TRANSFER,  200, 100, 0, "IN1",  "OUT1"],
    ])
    out = apply_transfer_carried_cost(df)
    assert _price(out, "IN1") == pytest.approx(25.0)


def test_chain_with_mismatched_dates():
    # A->B and B->C, with each transfer-in dated a few days off its transfer-out.
    df = _df([
        ["2023-01-01", "T", "A", TYPE_BUY,      100,  50, 0, "A_BUY", ""],
        ["2023-02-10", "T", "A", TYPE_TRANSFER, -100, 50, 0, "OUT1",  "IN1"],
        ["2023-02-08", "T", "B", TYPE_TRANSFER,  200, 100, 0, "IN1",  "OUT1"],  # dated before OUT1
        ["2023-03-10", "T", "B", TYPE_TRANSFER, -200, 100, 0, "OUT2", "IN2"],
        ["2023-03-07", "T", "C", TYPE_TRANSFER,  100, 200, 0, "IN2",  "OUT2"],  # dated before OUT2
    ])
    out = apply_transfer_carried_cost(df)
    assert _price(out, "IN1") == pytest.approx(25.0)
    assert _price(out, "IN2") == pytest.approx(50.0)


def test_no_transfers_returns_unchanged():
    df = _df([
        ["2023-01-01", "T", "A", TYPE_BUY,  100, 50, 0, "A_BUY", ""],
        ["2023-02-01", "T", "A", TYPE_SELL, -50, 60, 0, "A_SELL", ""],
    ])
    out = apply_transfer_carried_cost(df)
    pd.testing.assert_frame_equal(out, df)


def test_unpaired_transfer_in_left_unchanged():
    # transfer-in with no matching transfer-out keeps its own price (graceful fallback).
    df = _df([
        ["2023-02-01", "T", "B", TYPE_TRANSFER, 200, 100, 0, "IN1", "OUT_MISSING"],
    ])
    out = apply_transfer_carried_cost(df)
    assert _price(out, "IN1") == pytest.approx(100.0)
