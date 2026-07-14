"""SPY forecasting pipeline reconstructed from an academic Colab project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "SPY", "IEI", "XLE"]
FEATURES = [
    "return_current", "lag_1m", "lag_2m", "lag_3m", "lag_6m",
    "volatility_12m", "dist_from_sma", "month_idx",
]


def download_prices() -> pd.DataFrame:
    """Download adjusted closing prices from Yahoo Finance."""
    raw = yf.download(
        TICKERS, start="2008-01-01", end="2026-01-01",
        auto_adjust=True, progress=False,
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return close.sort_index().ffill().dropna(how="all")


def save_eda(prices: pd.DataFrame, output_dir: Path) -> None:
    """Save correlation data and portfolio-level exploratory charts."""
    correlation = prices.resample("W").last().pct_change().dropna().corr()
    correlation.to_csv(output_dir / "weekly_correlation.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Weekly Return Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "weekly_correlation.png", dpi=160)
    plt.close()

    normalized = prices.div(prices.iloc[0]).mul(100)
    normalized.plot(figsize=(14, 7), linewidth=1.5)
    plt.title("Normalized Asset Growth (Base = 100)")
    plt.xlabel("Date")
    plt.ylabel("Normalized value")
    plt.tight_layout()
    plt.savefig(output_dir / "normalized_growth.png", dpi=160)
    plt.close()


def build_monthly_dataset(spy: pd.Series) -> pd.DataFrame:
    """Create lagged monthly features and a one-month-ahead target."""
    monthly = spy.resample("ME").last().to_frame(name="Close")
    monthly["return_current"] = monthly["Close"].pct_change()
    for months in (1, 2, 3, 6):
        monthly[f"lag_{months}m"] = monthly["return_current"].shift(months)
    monthly["volatility_12m"] = monthly["return_current"].rolling(12).std()
    monthly["sma_12"] = monthly["Close"].rolling(12).mean()
    monthly["dist_from_sma"] = monthly["Close"].div(monthly["sma_12"]).sub(1)
    monthly["month_idx"] = monthly.index.month
    monthly["target_next_return"] = monthly["return_current"].shift(-1)
    return monthly.dropna().copy()


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "directional_accuracy": float(
            np.mean(np.sign(actual) == np.sign(predicted))
        ),
    }


def run_lightgbm(dataset: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    """Train LightGBM on a chronological split and save its predictions."""
    split = int(len(dataset) * 0.8)
    train, test = dataset.iloc[:split], dataset.iloc[split:].copy()
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=5,
        random_state=42, verbosity=-1,
    )
    model.fit(train[FEATURES], train["target_next_return"])
    predicted_returns = model.predict(test[FEATURES])
    test["predicted_return"] = predicted_returns
    test["predicted_price"] = test["Close"] * (1 + predicted_returns)
    test["actual_next_price"] = test["Close"] * (
        1 + test["target_next_return"]
    )
    test.to_csv(output_dir / "lightgbm_predictions.csv")

    plt.figure(figsize=(14, 7))
    plt.plot(test.index, test["actual_next_price"], label="Actual next-month price")
    plt.plot(
        test.index, test["predicted_price"], "--", label="LightGBM prediction"
    )
    plt.title("SPY Monthly Price Forecast - LightGBM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lightgbm_forecast.png", dpi=160)
    plt.close()
    return metrics(test["target_next_return"].values, predicted_returns)


def run_prophet(spy: pd.Series) -> dict[str, float]:
    """Evaluate Prophet on the held-out final 20 percent of the series."""
    data = spy.reset_index()
    data.columns = ["ds", "y"]
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    model = Prophet(
        daily_seasonality=False, weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.fit(train)
    predicted = model.predict(test[["ds"]])["yhat"].values
    result = metrics(test["y"].values, predicted)
    result.pop("directional_accuracy")
    return result


def make_sequences(
    values: np.ndarray, look_back: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for index in range(look_back, len(values)):
        x.append(values[index - look_back:index, 0])
        y.append(values[index, 0])
    return np.asarray(x).reshape(-1, look_back, 1), np.asarray(y)


def run_deep_learning(
    spy: pd.Series, output_dir: Path
) -> dict[str, dict[str, float]]:
    """Optionally compare LSTM and GRU sequence models."""
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input
        from tensorflow.keras.models import Sequential
    except ImportError as exc:
        raise RuntimeError("Install TensorFlow to use --deep-learning") from exc

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(spy.values.reshape(-1, 1))
    x, y = make_sequences(scaled)
    split = int(len(x) * 0.8)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    results = {}

    for name, layer in (("lstm", LSTM), ("gru", GRU)):
        model = Sequential([
            Input(shape=(x_train.shape[1], 1)),
            layer(50), Dropout(0.2), Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(
            x_train, y_train, validation_split=0.2, epochs=30, batch_size=32,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )
        predicted = scaler.inverse_transform(
            model.predict(x_test, verbose=0)
        ).reshape(-1)
        actual = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).reshape(-1)
        results[name] = metrics(actual, predicted)
        results[name].pop("directional_accuracy")

        plt.figure(figsize=(12, 5))
        plt.plot(actual, label="Actual")
        plt.plot(predicted, label=name.upper())
        plt.title(f"SPY Daily Close - {name.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_forecast.png", dpi=160)
        plt.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--deep-learning", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prices = download_prices()
    if "SPY" not in prices.columns:
        raise RuntimeError("SPY data was not returned by Yahoo Finance")

    save_eda(prices, args.output_dir)
    monthly = build_monthly_dataset(prices["SPY"])
    results: dict[str, object] = {
        "lightgbm": run_lightgbm(monthly, args.output_dir),
        "prophet": run_prophet(prices["SPY"].dropna()),
    }
    if args.deep_learning:
        results.update(run_deep_learning(prices["SPY"].dropna(), args.output_dir))

    (args.output_dir / "model_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
