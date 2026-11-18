"""Impact models (M0, M1, M1c, M2), the lambda readout and the forecast comparison test.

M0   dm = mu + lambda x                                   (constant Kyle lambda)
M1   dm = mu + g'z + (lambda_0 + sum_k lambda_k z_k + lambda_w[window]) x
M1c  M1 + c * sign(x) sqrt|x|                             (concavity diagnostic)
M2   LightGBM f(x, z, minute), constrained increasing in x

All coefficients and hyperparameters are estimated on training days only.
"""

import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

from features import N_WINDOWS, STATE_COLS

M2_FEATURES = ["x", *STATE_COLS, "minute"]
M2_GRID = {"num_leaves": [15, 31], "min_child_samples": [200, 1000], "n_estimators": [200, 500]}
M2_FIXED = {"learning_rate": 0.03, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
            "verbose": -1, "random_state": 0}


def clustered_ols(y: pd.Series, X: pd.DataFrame, data: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS with standard errors clustered two ways, by stock and by date.

    Args:
        y (pd.Series): Dependent variable.
        X (pd.DataFrame): Regressors; a constant is added.
        data (pd.DataFrame): Rows aligned with y and X, with 'sym' and 'date'.

    Returns:
        RegressionResultsWrapper: Fitted statsmodels result.
    """
    groups = np.column_stack([pd.factorize(data["sym"])[0], pd.factorize(data["date"])[0]])
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
        cov_type="cluster", cov_kwds={"groups": groups})


def window_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Indicators for half-hour windows 1..12 (window 0, 09:30-10:00, is the base).

    Args:
        df (pd.DataFrame): Rows with a 'window' column.

    Returns:
        pd.DataFrame: Float columns w1..w12.
    """
    return pd.DataFrame({f"w{k}": (df["window"] == k).astype(float) for k in range(1, N_WINDOWS)},
                        index=df.index)


def m1_design(df: pd.DataFrame, concave: bool = False) -> pd.DataFrame:
    """Regressors for M1: state main effects plus state and window interactions with x.

    Args:
        df (pd.DataFrame): Feature rows with x, standardised state columns and window.
        concave (bool): If True, add sign(x) sqrt|x| (model M1c).

    Returns:
        pd.DataFrame: Design matrix without a constant.
    """
    w = window_dummies(df)
    X = pd.concat([df[["x"]], df[STATE_COLS], w], axis=1)
    for c in STATE_COLS:
        X[f"x*{c}"] = df["x"] * df[c]
    for c in w.columns:
        X[f"x*{c}"] = df["x"] * w[c]
    if concave:
        X["sqrt_x"] = np.sign(df["x"]) * np.sqrt(df["x"].abs())
    return X


def lambda_m1(params: pd.Series, df: pd.DataFrame) -> np.ndarray:
    """State-dependent slope of M1 with respect to x: lambda_0 + sum lambda_k z_k + lambda_w.

    Args:
        params (pd.Series): Fitted M1 coefficients.
        df (pd.DataFrame): Feature rows.

    Returns:
        np.ndarray: Lambda per row, in bps per unit of normalised flow.
    """
    lam = params["x"] + sum(params[f"x*{c}"] * df[c] for c in STATE_COLS)
    w = window_dummies(df)
    lam = lam + sum(params[f"x*{c}"] * w[c] for c in w.columns)
    return np.asarray(lam)


def day_folds(dates: pd.Series, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplit over whole trading days, with a one-day gap before each validation day block.

    Args:
        dates (pd.Series): Date of each training row.
        n_splits (int): Number of expanding-window folds.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: (train_idx, valid_idx) row positions per fold.
    """
    days = np.sort(dates.unique())
    folds = []
    for tr_days, va_days in TimeSeriesSplit(n_splits=n_splits, gap=1).split(days):
        tr = np.flatnonzero(dates.isin(days[tr_days]).to_numpy())
        va = np.flatnonzero(dates.isin(days[va_days]).to_numpy())
        folds.append((tr, va))
    return folds


def m2_params(grid_point: dict) -> dict:
    """LightGBM parameters for one grid point, with the monotone constraint on x."""
    monotone = [1 if f == "x" else 0 for f in M2_FEATURES]
    return {**M2_FIXED, **grid_point, "monotone_constraints": monotone}


def tune_m2(train: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Grid-search M2 hyperparameters by day-level time-series cross-validation.

    Args:
        train (pd.DataFrame): Training feature rows.

    Returns:
        tuple[dict, pd.DataFrame]: Best grid point and the CV table (mean
        validation MSE per grid point).
    """
    X, y = train[M2_FEATURES], train["dm"]
    folds = day_folds(train["date"])
    rows = []
    for values in itertools.product(*M2_GRID.values()):
        point = dict(zip(M2_GRID, values))
        mse = []
        for tr, va in folds:
            model = lgb.LGBMRegressor(**m2_params(point)).fit(X.iloc[tr], y.iloc[tr])
            mse.append(np.mean((y.iloc[va] - model.predict(X.iloc[va])) ** 2))
        rows.append({**point, "cv_mse": np.mean(mse)})
    table = pd.DataFrame(rows).sort_values("cv_mse", ignore_index=True)
    best = table.iloc[0][list(M2_GRID)].to_dict()
    return {k: int(v) for k, v in best.items()}, table


def fit_m2(train: pd.DataFrame, point: dict) -> lgb.LGBMRegressor:
    """Fit M2 on all training rows with the chosen hyperparameters.

    Args:
        train (pd.DataFrame): Training feature rows.
        point (dict): Grid point from tune_m2.

    Returns:
        lgb.LGBMRegressor: Fitted model.
    """
    return lgb.LGBMRegressor(**m2_params(point)).fit(train[M2_FEATURES], train["dm"])


def lambda_m2(model: lgb.LGBMRegressor, df: pd.DataFrame, x_ref: float) -> np.ndarray:
    """Local slope of M2 in x around zero flow: [f(x_ref, z) - f(-x_ref, z)] / (2 x_ref).

    Args:
        model (lgb.LGBMRegressor): Fitted M2.
        df (pd.DataFrame): Feature rows (their own x is replaced).
        x_ref (float): Half-width of the difference, in normalised flow units.

    Returns:
        np.ndarray: Lambda per row; non-negative by the monotone constraint.
    """
    up, down = df[M2_FEATURES].copy(), df[M2_FEATURES].copy()
    up["x"], down["x"] = x_ref, -x_ref
    return (model.predict(up) - model.predict(down)) / (2 * x_ref)


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, time: pd.Series, lags: int = 10) -> tuple[float, float]:
    """Diebold-Mariano test of equal mean loss, on the cross-sectional mean loss per bucket.

    The loss differential d_t = mean_i(loss_a - loss_b) is averaged over stocks in
    each (date, minute) and its mean is tested with a Newey-West variance.

    Args:
        loss_a (pd.Series): Per-row loss of model A (e.g. squared error).
        loss_b (pd.Series): Per-row loss of model B.
        time (pd.Series): Time key per row (e.g. date and minute combined).
        lags (int): Newey-West truncation lag.

    Returns:
        tuple[float, float]: DM statistic (negative favours A) and two-sided p-value.
    """
    d = (loss_a - loss_b).groupby(time.to_numpy()).mean().to_numpy()
    n, dbar = len(d), d.mean()
    e = d - dbar
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    stat = dbar / np.sqrt(var / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))
