import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_STATE = 42

def generate_series(n=500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.02 * t
    season = 2.0 * np.sin(2 * np.pi * t / 24) + 1.5 * np.cos(2 * np.pi * t / 7)
    noise = rng.normal(0, 0.8, n)
    y = 10 + trend + season + noise
    return pd.Series(y, name="y")

def make_supervised(y: pd.Series, p=24, seasonal_lags=(7,14)):
    df = pd.DataFrame({"y": y})
    for i in range(1, p+1):
        df[f"lag_{i}"] = df["y"].shift(i)
    for s in seasonal_lags:
        df[f"lag_{s}"] = df["y"].shift(s)
    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=["y"])
    y = df["y"]
    return X, y

def backtest(X, y, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)
    model = Ridge()
    gs = GridSearchCV(model, {"alpha": [0.1, 1.0, 3.0, 10.0]}, cv=tscv, scoring="neg_mean_squared_error")
    gs.fit(X, y)
    return gs

def forecast_last(gs, y, horizon=24, p=24, seasonal_lags=(7,14)):
    hist = y.copy().tolist()
    preds = []
    for _ in range(horizon):
        feats = []
        for i in range(1, p+1):
            feats.append(hist[-i])
        for s in seasonal_lags:
            feats.append(hist[-s])
        pred = gs.best_estimator_.predict([feats])[0]
        preds.append(pred)
        hist.append(pred)
    return np.array(preds)

def main():
    os.makedirs("artifacts", exist_ok=True)
    series = generate_series(n=600)
    X, y = make_supervised(series, p=24, seasonal_lags=(7,14))

    gs = backtest(X, y, splits=5)
    print("Best params:", gs.best_params_)
    print("Best CV MSE:", -gs.best_score_)

    H = 48
    preds = forecast_last(gs, series.values, horizon=H, p=24, seasonal_lags=(7,14))

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(series.index, series.values, label="history")
    future_idx = np.arange(len(series), len(series)+H)
    plt.plot(future_idx, preds, label="forecast")
    plt.legend()
    plt.title("Time Series Forecast")
    plt.savefig("artifacts/forecast.png", dpi=150, bbox_inches="tight")

    # Simple holdout error on last 100 points (as a sanity check)
    hold = 100
    X_hold, y_hold = X.iloc[-hold:], y.iloc[-hold:]
    y_hat = gs.predict(X_hold)
    mae = mean_absolute_error(y_hold, y_hat)
    rmse = mean_squared_error(y_hold, y_hat, squared=False)
    print(f"Holdout MAE={mae:.3f} RMSE={rmse:.3f}")

if __name__ == "__main__":
    main()
