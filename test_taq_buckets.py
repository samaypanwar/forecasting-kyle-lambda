"""Check bucketquery in taq_buckets.q against an independent pandas implementation.

Builds one synthetic TAQ day (trades and NBBO for two symbols) inside a local q
session, runs the q query on it, recomputes every column with pandas, and
asserts that the two agree. Run from this directory:

    poetry run python test_taq_buckets.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pykx as kx

HERE = Path(__file__).parent
DATE = pd.Timestamp("2020-02-03")
SYMS = ["AAA", "BBB"]
CONDS = " O"
OPEN, CLOSE = pd.Timedelta("09:30:00"), pd.Timedelta("16:00:00")
WIDTH = pd.Timedelta("1min")


def make_quotes(rng: np.random.Generator, sym: str, n: int) -> pd.DataFrame:
    """Simulate NBBO updates for one symbol, including some bad quotes.

    Args:
        rng (np.random.Generator): Random source.
        sym (str): Ticker symbol.
        n (int): Number of quote updates.

    Returns:
        pd.DataFrame: Columns time, sym, bid, ask, bsize, asize, sorted by time.
    """
    time = np.sort(rng.uniform(pd.Timedelta("09:25:00").value, CLOSE.value + 1e9, n))
    bid = 100 + 0.01 * np.cumsum(rng.integers(-1, 2, n))
    ask = bid + 0.01 * rng.integers(1, 4, n)

    # A few crossed and zero quotes that the query must drop.
    bad = rng.choice(n, 20, replace=False)
    ask[bad[:10]] = bid[bad[:10]] - 0.01
    bid[bad[10:]] = 0.0

    return pd.DataFrame({
        "time": pd.to_timedelta(time.astype("int64")),
        "sym": sym,
        "bid": bid.round(2),
        "ask": ask.round(2),
        "bsize": rng.integers(1, 50, n).astype("int32") * 100,
        "asize": rng.integers(1, 50, n).astype("int32") * 100,
    })


def make_trades(rng: np.random.Generator, quotes: pd.DataFrame, n: int) -> pd.DataFrame:
    """Simulate trades at the bid, the ask, the mid, or off-quote prices.

    Args:
        rng (np.random.Generator): Random source.
        quotes (pd.DataFrame): Quotes for one symbol, from make_quotes.
        n (int): Number of trades.

    Returns:
        pd.DataFrame: Columns time, sym, price, size, cond, sorted by time.
    """
    time = np.sort(rng.uniform(OPEN.value - 60e9, CLOSE.value + 60e9, n))
    trades = pd.DataFrame({"time": pd.to_timedelta(time.astype("int64"))})
    good = quotes[(quotes.bid > 0) & (quotes.ask > quotes.bid)]
    trades = pd.merge_asof(trades, good[["time", "bid", "ask"]], on="time")

    pick = rng.integers(0, 4, n)
    mid = (trades.bid + trades.ask) / 2
    price = np.select([pick == 0, pick == 1, pick == 2], [trades.bid, trades.ask, mid], mid + 0.005)
    trades["price"] = np.where(np.isnan(price), 100.0, price).round(3)
    trades["size"] = rng.integers(1, 20, n).astype("int32") * 100
    trades["cond"] = rng.choice(list(" OT"), n, p=[0.8, 0.1, 0.1])
    trades["sym"] = quotes.sym.iloc[0]
    return trades[["time", "sym", "price", "size", "cond"]]


def load_into_q(trades: pd.DataFrame, quotes: pd.DataFrame) -> None:
    """Create global trade and nbbo tables in the local q session.

    Args:
        trades (pd.DataFrame): All synthetic trades.
        quotes (pd.DataFrame): All synthetic quotes.
    """
    t = trades.assign(cond=trades["cond"].map(ord).astype("int32"))
    kx.q["trade"] = kx.toq(t)
    kx.q("trade:update date:{0}, cond:\"c\"$cond from trade".format(DATE.strftime("%Y.%m.%d")))
    kx.q["nbbo"] = kx.toq(quotes)
    kx.q("nbbo:update date:{0} from nbbo".format(DATE.strftime("%Y.%m.%d")))
    kx.q("nbbo:`sym`time xasc nbbo")


def reference(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Recompute the bucket table in pandas for one symbol.

    Args:
        trades (pd.DataFrame): Trades for one symbol.
        quotes (pd.DataFrame): Quotes for one symbol.

    Returns:
        pd.DataFrame: One row per 1-minute bucket, same columns as bucketquery.
    """
    q = quotes[(quotes.bid > 0) & (quotes.ask > quotes.bid) & (quotes.time <= CLOSE)]
    q = q.sort_values("time", kind="stable").reset_index(drop=True)

    # Lee-Ready with tick-rule fallback.
    t = trades[trades.cond.isin(list(CONDS)) & (trades.time >= OPEN) & (trades.time < CLOSE)].copy()
    t = t.sort_values("time", kind="stable").reset_index(drop=True)
    step = np.sign(t.price.diff().fillna(0.0))
    t["tick"] = step.replace(0.0, np.nan).ffill().fillna(0.0)
    t = pd.merge_asof(t, q[["time", "bid", "ask"]], on="time")
    m = ((t.bid + t.ask) / 2).fillna(t.price)

    # q compares floats with a relative tolerance of about 1e-14, so a trade
    # printed at the mid counts as "at the mid" even if the last bit differs.
    at_mid = np.isclose(t.price, m, rtol=1e-13, atol=0.0)
    above = ~at_mid & (t.price > m)
    below = ~at_mid & (t.price < m)
    t["side"] = np.where(above, 1, np.where(below, -1, np.where(t.tick > 0, 1, -1)))
    t["bucket"] = t.time.dt.floor(WIDTH)
    tb = t.groupby("bucket").agg(
        sflow=("side", lambda s: (s * t.loc[s.index, "size"]).sum()),
        vol=("size", "sum"),
        ntrd=("size", "size"),
    )

    # 1-second grid of prevailing quotes.
    grid = pd.DataFrame({"time": pd.timedelta_range(OPEN, CLOSE, freq="1s").astype("timedelta64[ns]")})
    g = pd.merge_asof(grid, q, on="time")
    g["mid"] = (g.bid + g.ask) / 2
    g["bucket"] = g.time.dt.floor(WIDTH)
    inside = g[g.time < CLOSE]
    gb = inside.groupby("bucket").agg(
        spr=("mid", lambda s: ((inside.loc[s.index, "ask"] - inside.loc[s.index, "bid"]) / s).mean()),
        dep=("bsize", lambda s: (s + inside.loc[s.index, "asize"]).mean()),
    )
    bd = g[g.time == g.bucket].set_index("bucket")["mid"].rename("mb").to_frame()
    bd["dm"] = 1e4 * (bd.mb.shift(-1) - bd.mb) / bd.mb

    r10 = g.groupby(g.time.dt.floor("10s"))["mid"].last().to_frame()
    r10 = r10[r10.index < CLOSE]
    r10["r"] = np.log(r10.mid / r10.mid.shift())
    r10["bucket"] = r10.index.floor(WIDTH)
    rv1 = (r10.r ** 2).groupby(r10.bucket).sum(min_count=1).rename("rv1")

    # Order-flow imbalance per update.
    pb, pbs, pa, pas = q.bid.shift(), q.bsize.shift(), q.ask.shift(), q.asize.shift()
    e = ((q.bid >= pb) * q.bsize - (q.bid <= pb) * pbs
         - (q.ask <= pa) * q.asize + (q.ask >= pa) * pas)
    q = q.assign(e=np.where(pb.isna(), 0.0, e), bucket=q.time.dt.floor(WIDTH))
    qb = q[(q.time >= OPEN) & (q.time < CLOSE)].groupby("bucket").agg(nq=("e", "size"), ofi=("e", "sum"))

    out = pd.DataFrame(index=pd.timedelta_range(OPEN, CLOSE - WIDTH, freq=WIDTH, name="bucket").astype("timedelta64[ns]"))
    out = out.join([bd, gb, rv1, tb, qb])
    out[["sflow", "vol", "ntrd", "nq", "ofi", "rv1"]] = out[["sflow", "vol", "ntrd", "nq", "ofi", "rv1"]].fillna(0)
    out["rv5"] = out.rv1.rolling(5, min_periods=1).sum()
    return out


def main() -> None:
    """Run bucketquery on synthetic data and compare with the pandas reference."""
    rng = np.random.default_rng(7)
    quotes = {s: make_quotes(rng, s, 30_000) for s in SYMS}
    trades = {s: make_trades(rng, quotes[s], 6_000) for s in SYMS}
    load_into_q(pd.concat(trades.values()), pd.concat(quotes.values()))

    # \l cannot take a path with spaces, so load relative to this directory.
    os.chdir(HERE)
    kx.q("\\l taq_buckets.q")
    args = (kx.SymbolVector(SYMS), kx.CharVector(CONDS), kx.DateAtom(DATE.date()), kx.TimespanAtom(WIDTH.to_pytimedelta()))
    got = kx.q("bucketquery", *args).pd()

    assert len(got) == 390 * len(SYMS), len(got)
    cols = ["mb", "dm", "spr", "dep", "rv1", "rv5", "sflow", "vol", "ntrd", "nq", "ofi"]
    for sym in SYMS:
        mine = got[got.sym == sym].set_index("bucket")[cols].astype(float)
        ref = reference(trades[sym], quotes[sym])[cols].astype(float)
        mine.index = pd.to_timedelta(mine.index)
        for c in cols:
            np.testing.assert_allclose(mine[c].to_numpy(), ref[c].to_numpy(), rtol=1e-9, atol=1e-9,
                                       err_msg=f"{sym} {c}")
    print(f"bucketquery matches pandas reference: {len(got)} rows, {len(cols)} columns, {len(SYMS)} syms")
    print(got.head(3).to_string())


if __name__ == "__main__":
    main()
