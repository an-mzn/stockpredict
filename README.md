# Stock Price Prediction with an LSTM

**A Python command-line experiment that trains an LSTM on historical stock closing prices and saves validation plots and predictions.**

**Originally developed:** January–March 2024

## Overview

The primary script downloads historical closing prices with `yfinance`, creates rolling lookback windows, trains a Keras LSTM, and compares its predictions with a chronological validation period. It reports validation RMSE and writes plots and a CSV for inspection. The default example uses Apple (AAPL) data from 2012 to 2019.

The command-line version is the primary implementation. For Colab-style inline charts in VS Code, open [`stockpredict_demo.ipynb`](stockpredict_demo.ipynb), select the Python 3.11 environment where you installed the dependencies, and run its cell. Two earlier Colab-derived script variants are preserved under [`examples/colab/`](examples/colab/) as references.

## What it does

- Downloads price history for a configurable ticker and date range.
- Uses the preceding 60 closing prices by default to predict the next close.
- Splits the time series chronologically into training and validation sets.
- Fits the min/max scaler on the training portion, then trains a two-layer LSTM model.
- Saves a price-history chart, a train/validation/prediction chart, and validation predictions as CSV.

## Technologies and structure

- Python, NumPy, pandas, scikit-learn, Matplotlib, yfinance, TensorFlow/Keras
- [`stockpredict_cli.py`](stockpredict_cli.py): primary command-line implementation.
- [`stockpredict_demo.ipynb`](stockpredict_demo.ipynb): notebook entry point for inline visuals in VS Code/Jupyter.
- [`examples/colab/`](examples/colab/): two preserved Colab-derived script variants.

## Run it

Use Python 3.11 for a broadly supported TensorFlow environment. The script downloads market data at runtime, so it needs an internet connection. To use the notebook in VS Code, install the Python and Jupyter extensions and select this virtual environment as the notebook kernel.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python stockpredict_cli.py
```

To display charts in a graphical environment while still saving them, add `--show`:

```bash
python stockpredict_cli.py --show
```

When run from the included Jupyter notebook in VS Code, the charts appear inline in the notebook output. Training progress is also shown in that cell.

On Windows PowerShell, create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`; then run the `python -m pip` and `python stockpredict_cli.py` commands above.

To set the ticker, date range, training duration, lookback window, or output directory:

```bash
python stockpredict_cli.py --ticker MSFT --start 2018-01-01 --end 2020-01-01 --epochs 1 --window 60 --outdir outputs
```

The CLI defaults are `AAPL`, `2012-01-01` through `2019-12-17`, one epoch, batch size 1, a 60-day window, and an `outputs/` directory. The end date follows `yfinance`'s date-range behavior. Each run writes:

- `01_close_price_history.png`
- `02_train_valid_predictions.png`
- `valid_with_predictions.csv`

## Model and limitations

The primary script uses an 80/20 chronological split, a scaler fit on the training segment, 60-observation input windows, two LSTM layers with 50 units each, dense layers of 25 and 1 units, the Adam optimizer, and mean squared error as the training loss. It reports root mean squared error (RMSE) on the validation segment.

This is a small historical experiment, not a validated trading system. The default data period ends in 2019; the reported single-step prediction is based on the last window in whichever period was downloaded, not a live market forecast. Results depend on the downloaded data and model training run. The scripts do not include automated tests or a saved trained model.

## Provenance and release notes

The source identifies the earlier scripts as Colab-generated and describes the model as following a common LSTM tutorial pattern, but it does not identify the original tutorial or its reuse terms. Those terms should be checked before public release. The Colab Drive URL was removed from the current script files, but it remains in the existing Git commit. Confirm the notebook's sharing status and decide whether history should be rewritten before making this repository public. The repository does not include the notebook itself.

No licence is included. The repository history contains one preservation commit dated August 28, 2026; it does not represent the 2024 development history. The author email in that commit will be visible to people viewing public Git history.
