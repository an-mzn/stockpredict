# Project Story: Stock Price Prediction with an LSTM

## Why I built it

I built this project as a hands-on way to explore time-series prediction with a recurrent neural network. I wanted to see how an LSTM could use a recent window of stock closing prices to estimate later prices, and to make the process concrete by comparing its predictions with data it had not trained on.

## What I wanted to find out

- How to turn a single series of daily closing prices into input windows and next-step targets.
- How an LSTM's predictions compare with actual prices over a later chronological validation period.
- How to make the experiment repeatable for different tickers and date ranges, and inspect its results visually.

This is an exploration of a modeling workflow, not a claim that historical prices alone can reliably predict future markets.

## What I built

The primary implementation is a Python command-line program. It downloads price history with `yfinance`, uses the closing-price series, and splits observations chronologically into an 80% training portion and a 20% validation portion. It fits a min/max scaler on the training data, creates rolling windows (60 observations by default), and trains a two-layer Keras LSTM followed by dense layers.

The program evaluates predictions over the validation period using RMSE in price units. It also makes one additional next-step estimate from the final lookback window in the selected historical range. Ticker, dates, lookback size, training settings, and output directory can be changed with command-line options.

## Outcome

The project delivers a runnable end-to-end experiment. A run reports its validation RMSE and saves:

- A chart of the downloaded closing-price history.
- A chart comparing the training period, validation prices, and model predictions.
- A CSV containing validation prices and corresponding predictions.

The repository does not include a representative run's RMSE or generated charts, so it does not make a quantified claim about predictive performance. Results vary with the data and training run. The default example uses AAPL history from 2012 through 2019; its final estimate is based on that historical range, not a live forecast.

## What the experiment demonstrates

- Building rolling windows from a univariate time series for supervised learning.
- Keeping validation data out of scaler fitting and preserving chronological evaluation.
- Providing a configurable command-line workflow and saving outputs for review.
- Communicating model results with plots, a CSV, and an error metric.

## Limitations and next steps

This is a small learning project, not a trading system. It has no baseline comparison, so the RMSE alone does not show whether the LSTM performs better than a simple forecasting rule. The model defaults to one training epoch, and there is no saved model, automated test suite, or recorded benchmark run. The two scripts in `examples/colab/` are earlier reference variants; the command-line program is the recommended entry point.

A useful extension would be to run the model alongside a naive baseline, record the metrics and settings, and include the resulting comparison plot. That would make the project's outcome easier to assess and reproduce.

## Run the project

See [`README.md`](README.md) for environment setup and usage. The main entry point is `stockpredict_cli.py`; `stockpredict_demo.ipynb` provides an inline-chart workflow for Jupyter or VS Code.
