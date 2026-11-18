"""Feature construction for the state-dependent Kyle's lambda study.

Every normaliser, winsorisation bound and time-series coefficient is estimated on
the training days only and then applied unchanged to the test days, so no test
information leaks into any feature. State variables are lagged one bucket within
each (sym, date), so z_{t-1} never crosses the overnight gap.
"""

import numpy as np
import pandas as pd

# Half-hour windows used for the intraday profile (H2): 13 windows from 09:30.
N_WINDOWS = 13

# State variables z_{t-1} (all lagged one bucket) and their meaning.
STATE_COLS = ["spr", "rv", "dep", "act", "qupd", "ofi"]
AR_LAGS = 5
Z_CAP = 5.0


def fit_normalisers(train: pd.DataFrame) -> dict:
    """Estimate per-stock scales and winsorisation bounds on training data.

    Args:
        train (pd.DataFrame): Bucket panel restricted to training dates.

    Returns:
        dict: 'vbar' and 'depbar' (per-stock mean 1-minute volume in shares and
        mean quoted depth in lots) and 'bounds' (0.1% and 99.9% quantiles of
        the mid return and normalised flow).
    """
    vbar = train.groupby("sym")["vol"].mean()
    depbar = train.groupby("sym")["dep"].mean()
    x = train["sflow"] / train["sym"].map(vbar)
    bounds = {
        "dm": train["dm"].quantile([0.001, 0.999]).to_numpy(),
        "x": x.quantile([0.001, 0.999]).to_numpy(),
    }
    return {"vbar": vbar, "depbar": depbar, "bounds": bounds}


def raw_features(panel: pd.DataFrame, norm: dict) -> pd.DataFrame:
    """Build the dependent variable, order flow and lagged state variables.

    Args:
        panel (pd.DataFrame): Bucket panel from taq_data.load_buckets, sorted by
            date, sym and bucket, with 'minute' and 'period' columns.
        norm (dict): Output of fit_normalisers.

    Returns:
        pd.DataFrame: One row per (date, sym, minute >= 1) with columns dm, x,
        window and the lagged state variables in STATE_COLS (unstandardised).
    """
    df = panel.copy()
    lo, hi = norm["bounds"]["dm"]
    df["dm"] = df["dm"].clip(lo, hi)
    lo, hi = norm["bounds"]["x"]
    df["x"] = (df["sflow"] / df["sym"].map(norm["vbar"])).clip(lo, hi)

    # Contemporaneous state measures; they enter the models only after the lag below.
    df["spr"] = np.log(1e4 * df["spr"])
    df["rv"] = np.log(df["rv5"] + 1e-12)
    df["dep"] = np.log(df["dep"])
    df["act"] = np.log1p(df["ntrd"])
    df["qupd"] = np.log1p(df["nq"])
    df["ofi"] = df["ofi"] / df["sym"].map(norm["depbar"])

    lagged = df.groupby(["date", "sym"])[STATE_COLS].shift(1)
    df[STATE_COLS] = lagged
    df["window"] = (df["minute"] // 30).clip(upper=N_WINDOWS - 1)

    # The first bucket of each day has no lagged state.
    return df[df["minute"] >= 1].reset_index(drop=True)


def fit_standardiser(train: pd.DataFrame) -> pd.DataFrame:
    """Training means and standard deviations of the state variables.

    Args:
        train (pd.DataFrame): Training rows from raw_features.

    Returns:
        pd.DataFrame: Rows 'mean' and 'std', one column per state variable.
    """
    return train[STATE_COLS].agg(["mean", "std"])


def standardise(df: pd.DataFrame, scale: pd.DataFrame) -> pd.DataFrame:
    """Express state variables in training standard deviations, capped at +/-5.

    The cap limits the leverage of rare extreme states (e.g. a locked quote or
    a single huge OFI print) on the interaction regressions.

    Args:
        df (pd.DataFrame): Rows from raw_features.
        scale (pd.DataFrame): Output of fit_standardiser.

    Returns:
        pd.DataFrame: Copy of df with standardised state columns.
    """
    out = df.copy()
    out[STATE_COLS] = ((out[STATE_COLS] - scale.loc["mean"]) / scale.loc["std"]).clip(-Z_CAP, Z_CAP)
    return out


def ar_design(df: pd.DataFrame) -> pd.DataFrame:
    """Lags x_{t-1}, ..., x_{t-p} of normalised flow within each (sym, date).

    Args:
        df (pd.DataFrame): Rows from raw_features, sorted by date, sym, minute.

    Returns:
        pd.DataFrame: Columns xlag1..xlagp, NaN where a lag crosses the open.
    """
    grouped = df.groupby(["date", "sym"])["x"]
    return pd.DataFrame({f"xlag{j}": grouped.shift(j) for j in range(1, AR_LAGS + 1)})


def fit_ar(train: pd.DataFrame) -> np.ndarray:
    """Pooled AR(p) for normalised order flow, fitted by OLS on training rows.

    x_t = a + sum_j phi_j x_{t-j} + u_t

    Args:
        train (pd.DataFrame): Training rows from raw_features.

    Returns:
        np.ndarray: Coefficients [a, phi_1, ..., phi_p].
    """
    lags = ar_design(train)
    ok = lags.notna().all(axis=1)
    design = np.column_stack([np.ones(ok.sum()), lags[ok].to_numpy()])
    coef, *_ = np.linalg.lstsq(design, train.loc[ok, "x"].to_numpy(), rcond=None)
    return coef


def add_surprise(df: pd.DataFrame, coef: np.ndarray) -> pd.DataFrame:
    """Split flow into its AR forecast x_hat and surprise u = x - x_hat.

    Rows whose lags reach before the first modelled bucket of the day use the
    available lags only (missing lags count as zero flow).

    Args:
        df (pd.DataFrame): Rows from raw_features.
        coef (np.ndarray): Output of fit_ar.

    Returns:
        pd.DataFrame: Copy of df with x_hat and u columns.
    """
    lags = ar_design(df).fillna(0.0).to_numpy()
    out = df.copy()
    out["x_hat"] = coef[0] + lags @ coef[1:]
    out["u"] = out["x"] - out["x_hat"]
    return out
