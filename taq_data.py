"""Fetch per-minute TAQ bucket tables from the course kdb+ server, with a parquet cache.

The heavy lifting runs on the server: `bucketquery` from taq_buckets.q is sent
over IPC and only the per-minute table comes back. Each trading day is cached as
one parquet file, so a rerun with a complete cache needs no server connection.

q initialises TLS when pykx is imported and reads its CA bundle from
SSL_CA_CERT_FILE, so that variable must be set before the first pykx import,
e.g. to ssl.get_default_verify_paths().cafile.
"""

import datetime as dt
from pathlib import Path

import keyring
import pandas as pd
import pykx as kx

SERVER = "tpr-mscf-kx.tepper.cmu.edu"
PORT = 5000
SYMS_PER_CALL = 5


def connect() -> kx.IntAtom:
    """Open a TLS handle to the TAQ server using credentials stored in keyring.

    Credentials follow the course setup: keyring service SERVER holds the
    andrewid under '__andrewid__' and the password under the andrewid.

    Returns:
        kx.IntAtom: q connection handle.
    """
    user = keyring.get_password(SERVER, "__andrewid__")
    password = keyring.get_password(SERVER, user)
    return kx.q.hopen(kx.SymbolAtom(f":tcps://{SERVER}:{PORT}:{user}:{password}"))


def server_dates(handle: kx.IntAtom) -> list[dt.date]:
    """List the trading dates held in the server's partitioned database.

    Args:
        handle (kx.IntAtom): Open connection from `connect`.

    Returns:
        list[dt.date]: Available dates, ascending.
    """
    return [d.py() for d in kx.q('{x"date"}', handle)]


def fetch_day(handle: kx.IntAtom, syms: list[str], conds: str, date: dt.date,
              width: dt.timedelta) -> pd.DataFrame:
    """Run bucketquery on the server for one date, a few symbols per call.

    The server aborts long queries with 'stop, and busy days (e.g. the
    late-February 2020 sell-off) exceed its limit for 30 stocks at once, so the
    symbols are sent in chunks of SYMS_PER_CALL.

    Args:
        handle (kx.IntAtom): Open connection from `connect`.
        syms (list[str]): Tickers to include.
        conds (str): Trade condition characters to keep, e.g. " @EFI".
        date (dt.date): Trading date.
        width (dt.timedelta): Bucket width, e.g. one minute.

    Returns:
        pd.DataFrame: One row per (sym, bucket); columns documented in taq_buckets.q.
    """
    query = kx.q("bucketquery")
    chunks = [syms[i:i + SYMS_PER_CALL] for i in range(0, len(syms), SYMS_PER_CALL)]
    frames = [
        kx.q("{[h;f;s;c;d;w] h(f;s;c;d;w)}", handle, query, kx.SymbolVector(chunk),
             kx.CharVector(conds), kx.DateAtom(date), kx.TimespanAtom(width)).pd()
        for chunk in chunks
    ]
    return pd.concat(frames, ignore_index=True)


def load_buckets(dates: list[dt.date], syms: list[str], conds: str, width: dt.timedelta,
                 cache_dir: Path, requery: bool = False) -> pd.DataFrame:
    """Load bucket tables for many dates, querying the server only for missing days.

    Args:
        dates (list[dt.date]): Trading dates to load.
        syms (list[str]): Tickers to include.
        conds (str): Trade condition characters to keep.
        width (dt.timedelta): Bucket width.
        cache_dir (Path): Folder holding one parquet file per date.
        requery (bool): If True, ignore the cache and refetch every date.

    Returns:
        pd.DataFrame: All dates stacked, sorted by date, sym and bucket.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {d: cache_dir / f"{d.isoformat()}.parquet" for d in dates}
    missing = [d for d, f in files.items() if requery or not f.exists()]

    if missing:
        handle = connect()
        for d in missing:
            fetch_day(handle, syms, conds, d, width).to_parquet(files[d], index=False)
        kx.q.hclose(handle)

    frames = [pd.read_parquet(files[d]) for d in dates]
    return pd.concat(frames, ignore_index=True).sort_values(["date", "sym", "bucket"], ignore_index=True)
