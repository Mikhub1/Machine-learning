# Time Series Forecasting with Lag Features

## Problem
Forecast the next `H` values of a univariate time series (e.g., demand) using simple supervised learning on lag features.

## Dataset
Synthetic seasonal series generated in code (trend + seasonality + noise). No downloads.

## Approach
- Construct lag features (t-1 ... t-p) and seasonal lags (t-7, t-14).
- Train `Ridge` regression with `GridSearchCV`.
- Use expanding-window backtest (time series split) for evaluation.
- Produce a forecast plot saved under `artifacts/`.

## How to Run
```bash
python main.py
```

## Files
```
.
├── main.py
├── requirements.txt
└── README.md
```

## Extend
- Swap in `RandomForestRegressor` or `XGBRegressor`.
- Add calendar/holiday features.
- Replace generator with your CSV and date parsing.
