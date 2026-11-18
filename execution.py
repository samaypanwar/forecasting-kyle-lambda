"""Execution backtest: schedules driven by lambda forecasts, scored by an ex-post evaluator.

Cost model. Buying w_t (in units of the stock's mean one-minute volume) in bucket t
moves the mid by lambda_t w_t bps; with fills spread evenly through the bucket the
average fill pays half of that move. Implementation shortfall per share, in bps, is

    IS = (1/W) * sum_t w_t * (0.5 * lambda_t * w_t + carry * sum_{s<t} lambda_s w_s)

with W = sum_t w_t. carry = 0 treats the one-minute impact as transient (the
schedule-shaping case); carry = 1 lets every earlier trade's impact persist.

Schedules. Minimising sum_t lambda_t w_t^2 subject to sum_t w_t = W gives
w_t = W * lambda_t^{-1} / sum_s lambda_s^{-1}. The adaptive schedules re-solve this
each minute for the remaining quantity, using the model's state-based forecast for
the next bucket and its training time-of-day profile for later buckets.

Evaluator. Each strategy is scored with lambda estimated ex post from the next
EVAL_WINDOW buckets of the same stock-day (OLS of dm on x through the origin). It
uses only data from bucket t onward, so it is independent of the information the
schedules used, and no forecasting model grades itself.
"""

import numpy as np
import pandas as pd

HORIZON = 60
START_MINUTES = [30, 90, 150, 210, 270, 330]
EVAL_WINDOW = 21
LAMBDA_FLOOR_SHARE = 0.1


def to_grid(df: pd.DataFrame, col: str) -> tuple[np.ndarray, pd.MultiIndex]:
    """Reshape one column into a (stock-day, minute) array.

    Args:
        df (pd.DataFrame): Feature rows with date, sym and minute.
        col (str): Column to reshape.

    Returns:
        tuple[np.ndarray, pd.MultiIndex]: Array of shape (n_stockdays, 390) with
        NaN where a minute is missing, and the (date, sym) index of its rows.
    """
    wide = df.pivot_table(index=["date", "sym"], columns="minute", values=col, aggfunc="first")
    wide = wide.reindex(columns=range(390))
    return wide.to_numpy(), wide.index


def evaluator_lambda(dm: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Forward-looking ex-post lambda: sum(x dm) / sum(x^2) over buckets t .. t+EVAL_WINDOW-1.

    Args:
        dm (np.ndarray): Mid returns, shape (n_stockdays, 390).
        x (np.ndarray): Normalised signed flow, same shape.

    Returns:
        np.ndarray: Evaluator lambda, same shape (NaN where undefined).
    """
    # Prefix sums with a leading zero: c[:, k] = sum of the first k buckets.
    xy, xx = np.nan_to_num(x * dm), np.nan_to_num(x * x)
    cxy = np.pad(np.cumsum(xy, axis=1), ((0, 0), (1, 0)))
    cxx = np.pad(np.cumsum(xx, axis=1), ((0, 0), (1, 0)))
    start = np.arange(xy.shape[1])
    end = np.minimum(start + EVAL_WINDOW, xy.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cxy[:, end] - cxy[:, start]) / (cxx[:, end] - cxx[:, start])
    return out


def static_schedule(lam: np.ndarray, total: float) -> np.ndarray:
    """Closed-form risk-neutral schedule w_t proportional to 1/lambda_t.

    Args:
        lam (np.ndarray): Positive lambda path(s), shape (..., N).
        total (float): Quantity W to execute.

    Returns:
        np.ndarray: Schedule with the same shape, summing to W along the last axis.
    """
    inv = 1.0 / lam
    return total * inv / inv.sum(axis=-1, keepdims=True)


def adaptive_schedule(lam_next: np.ndarray, profile: np.ndarray, total: float,
                      acf: np.ndarray | None = None, floor: float = 0.0) -> np.ndarray:
    """Re-solve the closed form each minute for the remaining quantity.

    At bucket t the lambda path is [lam_next[:, t], later buckets]. Without acf the
    later buckets revert straight to the profile (naive); with acf the current
    deviation decays at the training autocorrelation,
    profile[t+k] + acf[k] * (lam_next[:, t] - profile[t]) (persistence-aware).

    Args:
        lam_next (np.ndarray): State-based forecast for each bucket, shape (n, N).
        profile (np.ndarray): Time-of-day forecast for each bucket, shape (N,).
        total (float): Quantity W to execute.
        acf (np.ndarray | None): Deviation autocorrelation by lag (length >= N), or None.
        floor (float): Lower bound applied to the later-bucket forecasts.

    Returns:
        np.ndarray: Executed schedule, shape (n, N).
    """
    n, horizon = lam_next.shape
    remaining = np.full(n, total)
    w = np.zeros((n, horizon))
    for t in range(horizon):
        later = np.broadcast_to(profile[t + 1:], (n, horizon - t - 1))
        if acf is not None:
            dev = lam_next[:, t] - profile[t]
            later = later + acf[1:horizon - t][None, :] * dev[:, None]
        inv_later = (1.0 / np.maximum(later, floor)).sum(axis=1)
        share = (1.0 / lam_next[:, t]) / (1.0 / lam_next[:, t] + inv_later)
        w[:, t] = remaining * share
        remaining -= w[:, t]
    return w


def shortfall(w: np.ndarray, lam_eval: np.ndarray, carry: float) -> np.ndarray:
    """Implementation shortfall per share in bps under the evaluator lambda.

    Args:
        w (np.ndarray): Schedules, shape (n, N).
        lam_eval (np.ndarray): Evaluator lambda, shape (n, N).
        carry (float): 0 for transient impact, 1 for fully permanent impact.

    Returns:
        np.ndarray: Shortfall per order, shape (n,).
    """
    impact = lam_eval * w
    prior = np.cumsum(impact, axis=1) - impact
    return (w * (0.5 * impact + carry * prior)).sum(axis=1) / w.sum(axis=1)


def floor_lambda(lam: np.ndarray, reference: float) -> np.ndarray:
    """Floor forecasts at LAMBDA_FLOOR_SHARE of a reference level (keeps 1/lambda finite)."""
    return np.maximum(lam, LAMBDA_FLOOR_SHARE * reference)


def profile_by_minute(df: pd.DataFrame, col: str) -> np.ndarray:
    """Mean of a lambda column by minute of day (training rows), length 390.

    Args:
        df (pd.DataFrame): Feature rows with 'minute'.
        col (str): Lambda column.

    Returns:
        np.ndarray: Profile; minutes without rows take the nearest available value.
    """
    return df.groupby("minute")[col].mean().reindex(range(390)).ffill().bfill().to_numpy()


def deviation_acf(df: pd.DataFrame, col: str, profile: np.ndarray, max_lag: int = HORIZON) -> np.ndarray:
    """Autocorrelation of lambda deviations from the time-of-day profile within stock-days.

    Args:
        df (pd.DataFrame): Feature rows with the lambda column.
        col (str): Lambda column.
        profile (np.ndarray): Profile from profile_by_minute.
        max_lag (int): Number of lags (lag 0 included).

    Returns:
        np.ndarray: Correlations at lags 0 .. max_lag-1.
    """
    dev = to_grid(df.assign(dev=df[col] - profile[df["minute"]]), "dev")[0]
    acf = [1.0]
    for k in range(1, max_lag):
        a, b = dev[:, :-k].ravel(), dev[:, k:].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        acf.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return np.array(acf)


def pretrade_forecasts(grids: dict[str, np.ndarray], profiles: dict[str, np.ndarray],
                       acfs: dict[str, np.ndarray], total: float) -> pd.DataFrame:
    """Forecast and realised cost of a TWAP order for every stock-day and start hour.

    The forecast lambda path for a model is profile_t + acf(t - s) * (lambda_hat_s -
    profile_s), so the deviation observed at the order start s decays at the rate
    measured on training data. 'Const' and 'ToD' ignore state.

    Args:
        grids (dict[str, np.ndarray]): 'eval' and one state-based grid per model.
        profiles (dict[str, np.ndarray]): 'ToD' and one profile per model.
        acfs (dict[str, np.ndarray]): Deviation autocorrelations per model.
        total (float): Order size W.

    Returns:
        pd.DataFrame: One row per order with 'realized' cost (bps) and the
        uncalibrated mean forecast lambda for Const, ToD and each model.
    """
    w = np.full(HORIZON, total / HORIZON)
    rows = []
    for start in START_MINUTES:
        sl = slice(start, start + HORIZON)
        lam_eval = grids["eval"][:, sl]
        ok = np.isfinite(lam_eval).all(axis=1)
        n = int(ok.sum())
        out = {"stockday": np.flatnonzero(ok), "start": start,
               "realized": shortfall(np.tile(w, (n, 1)), lam_eval[ok], 0.0),
               "Const": np.ones(n), "ToD": np.full(n, profiles["ToD"][sl].mean())}
        for m in acfs:
            dev0 = np.nan_to_num(grids[m][ok, start] - profiles[m][start])
            out[m] = (profiles[m][sl][None, :] + acfs[m][None, :] * dev0[:, None]).mean(axis=1)
        rows.append(pd.DataFrame(out))
    return pd.concat(rows, ignore_index=True)


def run_backtest(grids: dict[str, np.ndarray], profiles: dict[str, np.ndarray], total: float,
                 acfs: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Score every strategy on every order (stock-day x start hour).

    Args:
        grids (dict[str, np.ndarray]): 'eval' plus one state-based forecast grid per
            adaptive model (e.g. 'M1', 'M2'), each of shape (n_stockdays, 390).
        profiles (dict[str, np.ndarray]): Training time-of-day lambda by minute
            (length 390) for 'ToD' and for each adaptive model.
        total (float): Order size W in units of mean one-minute volume.
        acfs (dict[str, np.ndarray] | None): If given, also run a persistence-aware
            schedule per model, named '<model> persistent'.

    Returns:
        pd.DataFrame: One row per order and strategy with columns stockday,
        start, strategy, is_transient and is_permanent (bps).
    """
    rows = []
    models = [k for k in grids if k != "eval"]
    for start in START_MINUTES:
        sl = slice(start, start + HORIZON)
        lam_eval = grids["eval"][:, sl]
        ok = np.isfinite(lam_eval).all(axis=1)
        n = int(ok.sum())
        schedules = {"TWAP": np.full((n, HORIZON), total / HORIZON),
                     "ToD": np.tile(static_schedule(profiles["ToD"][sl], total), (n, 1))}
        for m in models:
            ref = np.median(profiles[m])
            lam_next = floor_lambda(np.nan_to_num(grids[m][ok, sl], nan=ref), ref)
            profile = floor_lambda(profiles[m][sl], ref)
            schedules[m] = adaptive_schedule(lam_next, profile, total)
            if acfs is not None:
                schedules[f"{m} persistent"] = adaptive_schedule(
                    lam_next, profile, total, acf=acfs[m], floor=LAMBDA_FLOOR_SHARE * ref)
        for name, w in schedules.items():
            rows.append(pd.DataFrame({
                "stockday": np.flatnonzero(ok), "start": start, "strategy": name,
                "is_transient": shortfall(w, lam_eval[ok], 0.0),
                "is_permanent": shortfall(w, lam_eval[ok], 1.0),
            }))
    return pd.concat(rows, ignore_index=True)
