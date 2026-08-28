
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Stock price prediction with LSTM (AAPL example).
# - Runs as a plain Python script (no notebook needed).
# - Saves plots to PNG files so you can view them in VS Code.
# - Minimal and close to the common Colab tutorial, with a few fixes.

import os
import math
import argparse
import numpy as np
import pandas as pd

# Ensure non-interactive backend so the script runs without a GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Use TensorFlow Keras (works on Intel, Apple Silicon, Windows)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def make_sequences(series_scaled, window=60):
    x, y = [], []
    for i in range(window, len(series_scaled)):
        x.append(series_scaled[i-window:i, 0])
        y.append(series_scaled[i, 0])
    x = np.array(x)
    y = np.array(y)
    # reshape to [samples, timesteps, features]
    x = x.reshape((x.shape[0], x.shape[1], 1))
    return x, y

def main():
    parser = argparse.ArgumentParser(description="AAPL LSTM stock predictor")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to download (default: AAPL)")
    parser.add_argument("--start", default="2012-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2019-12-17", help="End date (YYYY-MM-DD)")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (default: 1)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--window", type=int, default=60, help="Lookback window (default: 60)")
    parser.add_argument("--outdir", default="outputs", help="Directory to save plots/CSVs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Download data
    df = yf.download(args.ticker, start=args.start, end=args.end, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded. Check ticker/date range or network connection.")
    close = df["Close"].copy()

    # 2) Train/val split
    dataset = close.values
    training_data_len = math.ceil(len(dataset) * 0.8)

    # Fit scaler on training portion ONLY (prevents leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_vals = dataset[:training_data_len].reshape(-1, 1)
    scaler.fit(train_vals)

    scaled_all = scaler.transform(dataset.reshape(-1, 1))
    train_scaled = scaled_all[:training_data_len]
    val_scaled = scaled_all[training_data_len-args.window:]  # include window overlap

    # 3) Build sequences
    x_train, y_train = make_sequences(train_scaled, window=args.window)

    # 4) Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(args.window, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # 5) Train
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1)

    # 6) Validation set sequences
    x_val = []
    for i in range(args.window, len(val_scaled)):
        x_val.append(val_scaled[i-args.window:i, 0])
    x_val = np.array(x_val).reshape((-1, args.window, 1))

    # Ground-truth validation values (unscaled)
    y_val = dataset[training_data_len:]  # unscaled close prices

    # 7) Predict + invert scaling
    preds_scaled = model.predict(x_val, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).ravel()

    # 8) RMSE
    rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))

    # 9) Build DataFrame for plotting/saving
    train_series = pd.Series(dataset[:training_data_len], index=close.index[:training_data_len], name="Close")
    valid_df = pd.DataFrame({"Close": dataset[training_data_len:]}, index=close.index[training_data_len:])
    valid_df["Predictions"] = preds

    # 10) Plots
    # Close price history
    plt.figure(figsize=(16, 8))
    plt.title("Close Price History")
    plt.plot(close.index, close.values)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Close Price USD ($)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "01_close_price_history.png"), dpi=180)
    plt.close()

    # Train/Val/Predictions
    plt.figure(figsize=(16, 8))
    plt.title("Train / Validation / Predictions")
    plt.plot(train_series.index, train_series.values, label="Train")
    plt.plot(valid_df.index, valid_df["Close"].values, label="Validation")
    plt.plot(valid_df.index, valid_df["Predictions"].values, label="Predictions")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Close Price USD ($)", fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "02_train_valid_predictions.png"), dpi=180)
    plt.close()

    # 11) Single-step next prediction using last 60 closes in the downloaded range
    last_60 = dataset[-args.window:]
    last_60_scaled = scaler.transform(last_60.reshape(-1, 1))
    x_next = last_60_scaled.reshape((1, args.window, 1))
    next_price_scaled = model.predict(x_next, verbose=0)
    next_price = float(scaler.inverse_transform(next_price_scaled)[0, 0])

    # Save CSV for inspection
    valid_df.to_csv(os.path.join(args.outdir, "valid_with_predictions.csv"))

    # Console summary
    print(f"Ticker: {args.ticker}")
    print(f"Samples: {len(dataset)} | Train split: {training_data_len} | Val split: {len(dataset)-training_data_len}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Next-day style prediction from last {args.window} closes: ${next_price:.2f}")
    print(f"Saved plots to: {os.path.abspath(args.outdir)}")
    print("Files:")
    print(" - 01_close_price_history.png")
    print(" - 02_train_valid_predictions.png")
    print(" - valid_with_predictions.csv")

if __name__ == "__main__":
    main()
