# Forecasting Kyle's lambda: state-dependent price impact and execution

MSCF 46982 Market Microstructure and Algorithmic Trading, final paper.

## Files

| File | Content |
|---|---|
| `report.pdf` | Write-up (5 pages) |
| `final_paper.ipynb` / `final_paper.html` | Executed notebook and its HTML export |
| `taq_buckets.q` | Server-side kdb+/q query: one-minute buckets with Lee–Ready signed flow, spread, depth, OFI and realized variance |
| `taq_data.py` | Server connection (TLS, keyring credentials) and the per-day parquet cache |
| `features.py` | Lagged state variables, normalisers and the AR(5) expected/surprise flow split (training-only estimation) |
| `models.py` | M0, M1, M1c, M2 (LightGBM), lambda readout, time-series CV and the Diebold–Mariano test |
| `execution.py` | Execution backtest, ex-post evaluator and pre-trade cost forecasts |
| `test_taq_buckets.py` | Checks every column of `bucketquery` against an independent pandas implementation on synthetic data |
| `data/buckets/*.parquet` | Cached query output, one file per trading day (Feb 3–28, 2020). Included in the course submission zip only; it is derived from licensed TAQ data and is not in the public repository |
| `report/` | LaTeX source of the report (`report.tex`, `pipeline.tex`); figures are in `figures/` |

## Running

Install the packages in `requirements.txt` beyond the course environment: `pyarrow`, `statsmodels`, `scikit-learn`, `lightgbm`, `shap` and `scipy` (all available from conda-forge). Then open `final_paper.ipynb` from this folder and choose *Restart and run all*.

With the cache (in the submission zip) the notebook runs offline in under a minute. A fresh clone of the public repository has no cache, so the first run must query the server. To rebuild the cache from the course kdb+ server, set `REQUERY = True` in the setup cell; this takes about 3 minutes and needs the course keyring credentials. q reads its TLS CA bundle from `SSL_CA_CERT_FILE` when pykx is imported, so the setup cell sets that variable before the import.

The q test runs with `python test_taq_buckets.py` from this folder.
