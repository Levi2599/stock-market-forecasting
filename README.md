# Stock Market Forecasting

An end-to-end time-series project that explores historical market behaviour and compares multiple approaches for forecasting the SPY ETF.

## Overview

The analysis uses Yahoo Finance data for **SPY, AAPL, MSFT, AMZN, NVDA, IEI, and XLE**. It covers exploratory analysis, risk metrics, feature engineering, chronological train/test splits, and forecasting.

The original academic work compared:

- Prophet for statistical time-series forecasting
- LightGBM for next-month return prediction
- LSTM and GRU neural networks for sequence modelling
- A naive previous-price baseline for context

This repository contains a cleaned, reproducible implementation reconstructed from the original Colab submission. The notebook was preserved in Google Drive only as a PDF export, so generated outputs and execution history are not included.

## Skills demonstrated

- Financial time-series collection and preprocessing
- Correlation and risk analysis
- Lag, volatility, and moving-average feature engineering
- Leakage-aware chronological evaluation
- Classical ML and deep-learning modelling
- MAE, RMSE, R², and directional-accuracy evaluation

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
pip install -r requirements.txt
python stock_forecasting.py --output-dir outputs
```

Deep-learning models are optional:

```bash
python stock_forecasting.py --output-dir outputs --deep-learning
```

## Outputs

The pipeline creates correlation data and charts, normalized-growth charts, LightGBM predictions, forecast charts, and a JSON file with model metrics.

## Methodology

Financial data is split chronologically rather than randomly. Features use historical information only, and the prediction target is shifted one month forward. This reduces look-ahead leakage and better reflects a real forecasting workflow.

## Disclaimer

For educational and portfolio purposes only. This project is not investment advice.
