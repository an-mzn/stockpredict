# Project Context

## Identity and history

This repository contains Python scripts for an experiment that uses an LSTM neural network to estimate stock closing prices from recent historical closes. The supplied material establishes a Google Colab development workflow and the user reports the original development period as January–March 2024. It does not establish whether the work was for a university, personal, or business project, nor does it provide an assignment brief or a precise motivation beyond trying the stock-price prediction workflow.

The current Git history contains a single preservation commit, `chore: preserve stock prediction scripts`, dated August 28, 2026. This is not the original development history. The Colab Drive URL has been removed from the current exports during release preparation, but the pre-existing commit still contains it. Confirm the notebook's sharing status and decide whether to rewrite history before making this repository public; rewriting changes the commit hash. The current repository's GitHub remote returned 404 during inspection, so its hosting state was not established. Add a GitHub publication date to the README only after it is actually public.

## Project behavior

The primary script downloads daily price data for a ticker, extracts closing prices, divides the series chronologically into training and validation portions, creates rolling windows, trains a sequential LSTM model, evaluates validation predictions, and saves plots and a CSV. It also predicts one additional close from the final lookback window in the selected data range.

### Primary command-line flow

`stockpredict_cli.py` performs these steps:

1. Parse ticker, start/end dates, epochs, batch size, lookback window, and output directory. Defaults are AAPL, 2012-01-01, 2019-12-17, 1 epoch, batch size 1, 60 observations, and `outputs/`.
2. Download data with `yfinance`, explicitly requesting unadjusted OHLC values and flat columns for the single-ticker result; stop with an error if the returned frame is empty. These arguments keep the code's input shape and close-price meaning stable across recent `yfinance` default changes.
3. Read the close series and calculate an 80% training split using `ceil`.
4. Fit `MinMaxScaler` only on the training portion, then transform the full series.
5. Generate overlapping training input windows and next-step labels. Inputs have shape `[samples, window, 1]`.
6. Train a Keras `Sequential` model with two LSTM layers (50 units each), followed by dense layers of 25 and 1 units. The model uses Adam and mean squared error, and trains with the requested batch size and epoch count.
7. Create validation windows with the preceding lookback overlap, predict validation closes, invert scaling, and calculate RMSE in price units.
8. Save a close-price history chart, a train/validation/prediction chart, and `valid_with_predictions.csv`.
9. Use the last `window` observations from the selected historical range for one more model prediction, then print that value and the run summary.

The script saves figures under the supplied output directory, which is created if necessary. Matplotlib selects an available backend; `--show` displays figures in a graphical or notebook environment as well as saving them. The default CLI run does not call `show`, so it remains suitable for headless use.

### Preserved Colab-derived variants

- `examples/colab/stockpredict_dataframe_variant.py` keeps the DataFrame-style close-price representation. Its RMSE expression computes the absolute mean signed error rather than root mean squared error, despite the nearby comment. It also assigns predictions to a sliced DataFrame without an explicit copy.
- `examples/colab/stockpredict_series_variant.py` uses a Series for close prices and corrects the RMSE expression and the prediction plotting columns. Both Colab-derived versions execute notebook-style cells at import/run time, display plots interactively, and download the selected historical data again for the final prediction.

These files are retained as historical variants, not the recommended entry point. They scale the complete dataset before the split, unlike the CLI script, which fits the scaler on training data only.

## Repository structure

- `stockpredict_cli.py`: main runnable script with CLI arguments and saved artifacts.
- `stockpredict_demo.ipynb`: small notebook wrapper that runs the CLI with plot display enabled for inline notebook output.
- `examples/colab/stockpredict_dataframe_variant.py`: DataFrame-based Colab export.
- `examples/colab/stockpredict_series_variant.py`: Series-based Colab export.
- `requirements.txt`: dependencies for the primary script and historical exports.
- `.gitignore`: excludes virtual environments, caches, generated outputs, and local IDE settings.
- `README.md`: public-facing project and setup summary.

## Dependencies and integrations

The primary script uses Python's standard library (`argparse`, `math`, `os`), NumPy, pandas, Matplotlib, scikit-learn's `MinMaxScaler`, yfinance, and TensorFlow's Keras API. The notebook workflow also uses Jupyter/IPython kernel support. The historical exports import standalone Keras and `pandas-datareader`; `pandas-datareader` is unused in their code but remains in the manifest because the preserved files import it.

`yfinance` retrieves price data at run time. No downloaded datasets, API credentials, keys, private endpoints, or static third-party assets are included in the repository. The source does not document the original tutorial attribution or licensing terms; those remain to be established before public release. No open-source licence has been added.

## Technical choices visible in the code

- The prediction target is a single series: closing price, rather than multiple market features.
- The time split is chronological, with the first 80% used for training and the remaining 20% for validation.
- The CLI's scaler is fitted on training observations only. The Colab variants fit it before splitting, allowing validation-period extrema to affect scaling.
- A rolling window converts the univariate series into supervised next-step examples.
- The CLI's non-interactive plotting and file output allow runs in terminal or headless environments.
- Ticker, dates, training settings, lookback length, and output location are CLI-configurable.

The code does not explain why the architecture size, optimizer, loss, split ratio, or default training duration were selected. The context records those implementation facts without assigning undocumented rationale.

## Dependencies and running

Create a virtual environment with a Python release supported by TensorFlow (Python 3.11 is the documented recommendation), then install `requirements.txt`. Run `python stockpredict_cli.py --help` for argument options and `python stockpredict_cli.py` for the default historical example. A network connection is needed to retrieve price data. TensorFlow installation availability can vary by Python version and operating system.

The Colab-derived variants use the older interactive workflow and may be affected by version changes in Keras, yfinance, pandas, or data providers. The project has no dependency lockfile, automated tests, or CI configuration.

## Limitations

- The default date range ends in December 2019 and should not be described as a present-day prediction.
- The final single-step estimate is the point after the last observation in the chosen range, not an independently observed future price.
- Training defaults to one epoch and batch size one; there is no benchmark comparison or broader evaluation protocol in the supplied code.
- The code has minimal validation for ranges that are too short for the configured lookback or for unusable data returned by a provider.
- Output values depend on external historical data and model initialization/training behavior; no seed is set.
- The exported legacy DataFrame variant's RMSE calculation is incorrect as noted above.
- There is no deployment, web interface, database, saved model, or automated test suite in the supplied project.

## Release-preparation changes

- Renamed the CLI implementation to `stockpredict_cli.py` to mark it as the recommended entry point.
- Moved the two Colab-derived variants into `examples/colab/` and renamed them according to their close-price representation.
- Removed the embedded Google Drive notebook ID from both Colab exports while retaining a note that they originated in Colab.
- Made the CLI's `yfinance.download` call explicitly request unadjusted prices and a flat single-ticker DataFrame, avoiding recent default changes that can alter the close field or return shape.
- Added an optional `--show` mode and a Jupyter notebook wrapper so VS Code can display both plots inline while retaining the CLI's saved outputs.
- Added public setup and project documentation, a dependency manifest, and ignores for generated/local files.

No original model logic was intentionally changed during this preparation.
