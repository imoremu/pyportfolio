# pyportfolio/calculators/transfer_cost_carry.py
"""
Fund-to-fund traspaso carried-cost pre-pass.

Spanish tax rule: a traspaso between funds defers taxation. The cost basis (and
acquisition date) is carried over from the origin fund to the destination fund;
what changes is the number of participaciones (each fund has a different NAV).

The user, however, records the transfer-in row with the DESTINATION NAV as
`Precio Participación`. That silently drops the deferred gain. This pre-pass
replaces the transfer-in cost with the **carried cost basis** consumed by the
paired transfer-out (FIFO), so partial transfers and chains (A -> B -> C) are
handled correctly.

Pairing is done through the reciprocal ids already present in the workbook:
`salida[ID Transacción Espejo] == entrada[ID Transacción]` (and vice versa).

It must run on the FULL dataset (all funds) BEFORE the per-fund grouping, because
the cost travels between two different funds and can chain across several.
"""

import logging

import pandas as pd

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
    TRANSACTION_ID,
    MIRROR_TRANSACTION_ID,
    COUNTERPARTY_PRODUCT,
    TEST,
)

logger = logging.getLogger(__name__)

_EPS = 1e-9


def apply_transfer_carried_cost(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `dataset` where every fund-to-fund transfer-in row has its
    `SHARE_PRICE` replaced by the cost basis carried over from the origin fund
    (carried_total_cost / participaciones_recibidas).

    Rows that cannot be paired are left untouched (a warning is logged), so the
    behaviour degrades gracefully to using the row's own price.
    """
    if dataset is None or dataset.empty or TRANSACTION_TYPE not in dataset.columns:
        return dataset

    type_lower_all = dataset[TRANSACTION_TYPE].astype(str).str.lower()
    if not (type_lower_all == TYPE_TRANSFER).any():
        return dataset  # no traspasos -> nothing to do

    work = dataset.copy()

    pairing = _pair_transfers(work)  # transfer-out index -> transfer-in index
    carried_cost_for_in = {}  # transfer-in index -> carried total cost
    stacks = {}  # (test, ticker) -> list of [available_shares, cost_per_share]

    for idx in _chronological_order(work, pairing):
        row = work.loc[idx]
        ttype = str(row.get(TRANSACTION_TYPE, "")).lower()
        shares = pd.to_numeric(pd.Series([row.get(SHARES)]), errors="coerce").iloc[0]
        if pd.isna(shares):
            continue

        key = (row.get(TEST, None), row.get(TICKER))
        stack = stacks.setdefault(key, [])

        if ttype == TYPE_BUY and shares > _EPS:
            stack.append([float(shares), _cost_per_share(shares, row.get(SHARE_PRICE), row.get(COMISION))])

        elif ttype == TYPE_TRANSFER and shares > _EPS:
            # transfer-in: cost basis = carried cost from its paired transfer-out
            carried = carried_cost_for_in.get(idx)
            if carried is not None:
                cost_ps = carried / float(shares)
            else:
                cost_ps = _cost_per_share(shares, row.get(SHARE_PRICE), row.get(COMISION))
                logger.warning(
                    f"Transfer-in row {idx} ({key[0]}/{key[1]}) has no paired transfer-out; "
                    f"falling back to its own price as cost basis."
                )
            stack.append([float(shares), cost_ps])

        elif ttype == TYPE_SELL and shares < -_EPS:
            _consume(stack, abs(float(shares)))  # gain not needed here

        elif ttype == TYPE_TRANSFER and shares < -_EPS:
            consumed_cost = _consume(stack, abs(float(shares)))
            in_idx = pairing.get(idx)
            if in_idx is not None:
                carried_cost_for_in[in_idx] = consumed_cost
            else:
                logger.warning(
                    f"Transfer-out row {idx} ({key[0]}/{key[1]}) has no paired transfer-in; "
                    f"carried cost {consumed_cost:.4f} could not be applied."
                )

    # Write carried cost per share back onto the transfer-in rows.
    for in_idx, carried in carried_cost_for_in.items():
        shares = work.at[in_idx, SHARES]
        if pd.notna(shares) and abs(shares) > _EPS:
            work.at[in_idx, SHARE_PRICE] = carried / float(shares)

    return work


def _cost_per_share(shares, price, commission) -> float:
    shares = float(shares)
    if shares <= _EPS:
        return 0.0
    price = 0.0 if pd.isna(price) else float(price)
    commission = 0.0 if pd.isna(commission) else float(commission)
    return (shares * price + commission) / shares


def _consume(stack, qty) -> float:
    """Consume `qty` shares FIFO from `stack` (list of [available, cost_per_share]).

    Returns the total cost basis consumed. Raises ValueError on overselling.
    """
    total_available = sum(lot[0] for lot in stack)
    if qty > total_available + _EPS:
        raise ValueError(
            f"Carried-cost pre-pass: consuming {qty:.4f} shares exceeds available "
            f"{total_available:.4f}."
        )
    remaining = qty
    total_cost = 0.0
    while remaining > _EPS and stack:
        lot = stack[0]
        take = min(lot[0], remaining)
        total_cost += take * lot[1]
        lot[0] -= take
        remaining -= take
        if lot[0] <= _EPS:
            stack.pop(0)
    return total_cost


def _chronological_order(work: pd.DataFrame, pairing: dict):
    """Row indices ordered so the carried cost is always available when a transfer-in is
    processed.

    Ordering is by (Test, date, priority), BUT a paired transfer-in does not use its own
    date: it inherits its paired transfer-out's (Test, date) and is placed immediately
    after it (priority 1.5). This keeps "salida before entrada" even when the transfer-in
    is dated earlier than (or the same as) its transfer-out. Pairing itself is by
    `ID Transacción Espejo`, so it is unaffected by any date mismatch.
    """
    dt = work[DATETIME]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, format="mixed", dayfirst=True, errors="coerce")

    type_lower = work[TRANSACTION_TYPE].astype(str).str.lower()
    shares = pd.to_numeric(work[SHARES], errors="coerce")
    test_series = work[TEST] if TEST in work.columns else pd.Series("", index=work.index)

    # priority: buy(0) < transfer-out(1) < sell(2) < transfer-in(3) < other(4)
    prio = pd.Series(4.0, index=work.index)
    prio[type_lower == TYPE_BUY] = 0.0
    prio[(type_lower == TYPE_TRANSFER) & (shares < -_EPS)] = 1.0
    prio[type_lower == TYPE_SELL] = 2.0
    prio[(type_lower == TYPE_TRANSFER) & (shares > _EPS)] = 3.0

    # Effective sort keys for paired transfer-ins.
    #   - Normal case (money leaves origin, arrives destination a few days later, so the
    #     transfer-out is dated on/before the transfer-in): keep the transfer-in at its OWN
    #     real date, so it interleaves correctly with any other destination movements.
    #   - Inverted case (transfer-in dated BEFORE its transfer-out): force it to the
    #     transfer-out's date, right after it, so its carried cost is ready.
    # Either way the transfer-in is never processed before its transfer-out.
    eff_test = test_series.copy()
    eff_dt = dt.copy()
    eff_prio = prio.copy()
    for out_idx, in_idx in pairing.items():
        eff_test.at[in_idx] = test_series.at[out_idx]
        out_dt, in_dt = dt.at[out_idx], dt.at[in_idx]
        if pd.isna(in_dt) or pd.isna(out_dt) or in_dt <= out_dt:
            eff_dt.at[in_idx] = out_dt
            eff_prio.at[in_idx] = 1.5  # right after the transfer-out (1.0), before same-date sells (2.0)
        else:
            eff_dt.at[in_idx] = in_dt   # its own (later) real date; natural transfer-in position
            eff_prio.at[in_idx] = 3.0

    order_df = pd.DataFrame({
        "_test": eff_test,
        "_dt": eff_dt,
        "_prio": eff_prio,
        "_pos": range(len(work)),
    }, index=work.index)
    return list(order_df.sort_values(by=["_test", "_dt", "_prio", "_pos"], kind="stable").index)


def _pair_transfers(work: pd.DataFrame) -> dict:
    """Pair each transfer-out with its transfer-in.

    Primary key: the reciprocal ids in the workbook, i.e.
    ``transfer_out[ID Transacción Espejo] == transfer_in[ID Transacción]``.
    Fallback (only when ids are absent): counterparty product + Test
    (destination.Producto == origin.Producto Contraparte and vice versa), nearest date.
    """
    type_lower = work[TRANSACTION_TYPE].astype(str).str.lower()
    shares = pd.to_numeric(work[SHARES], errors="coerce")
    outs = work.index[(type_lower == TYPE_TRANSFER) & (shares < -_EPS)].tolist()
    ins = work.index[(type_lower == TYPE_TRANSFER) & (shares > _EPS)].tolist()

    pairing = {}
    if not outs or not ins:
        return pairing

    has_ids = MIRROR_TRANSACTION_ID in work.columns and TRANSACTION_ID in work.columns
    if has_ids:
        # (test, own transaction id) -> transfer-in index
        in_by_id = {}
        for j in ins:
            tid = work.at[j, TRANSACTION_ID]
            if pd.notna(tid) and str(tid).strip() != "":
                in_by_id[(_test_of(work, j), str(tid).strip())] = j
        for o in outs:
            mirror = work.at[o, MIRROR_TRANSACTION_ID]
            if pd.notna(mirror) and str(mirror).strip() != "":
                key = (_test_of(work, o), str(mirror).strip())
                if key in in_by_id:
                    pairing[o] = in_by_id[key]
        if pairing:
            return pairing

    # Fallback: counterparty-based pairing (nearest date).
    if COUNTERPARTY_PRODUCT not in work.columns:
        return pairing

    dt = work[DATETIME]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, format="mixed", dayfirst=True, errors="coerce")

    used_ins = set()
    for o in outs:
        o_prod, o_cp, o_test, o_dt = work.at[o, TICKER], work.at[o, COUNTERPARTY_PRODUCT], _test_of(work, o), dt[o]
        candidates = [
            j for j in ins
            if j not in used_ins
            and _test_of(work, j) == o_test
            and work.at[j, TICKER] == o_cp
            and work.at[j, COUNTERPARTY_PRODUCT] == o_prod
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda j: _date_distance(dt[j], o_dt))
        pairing[o] = best
        used_ins.add(best)
    return pairing


def _test_of(work, idx):
    return work.at[idx, TEST] if TEST in work.columns else None


def _date_distance(candidate_dt, origin_dt):
    if pd.isna(candidate_dt) or pd.isna(origin_dt):
        return (2, 0)
    delta = (candidate_dt - origin_dt).days
    return (0, delta) if delta >= 0 else (1, -delta)
