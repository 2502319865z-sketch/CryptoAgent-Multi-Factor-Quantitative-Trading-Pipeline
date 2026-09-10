# CryptoAgent: Multi-Factor Quantitative Trading Pipeline

An end-to-end research pipeline for building and evaluating cryptocurrency factor strategies. The project combines market data, cross-sectional factor construction, portfolio sorting, news sentiment analysis, sentiment/return diagnostics, and dynamic factor allocation.

> **Research use only.** This repository is an experimental analytics project, not financial advice or a production trading system. Backtested performance does not guarantee future results.

## Live demo

**[Open the CryptoAgent web application](https://ao1rmhklbtp7.trickle.host)**

[View the standalone live-demo link page](CryptoAgent_Live_Demo_Link.pdf)

The web interface was designed and built by Kengtao Wu. The deployment was verified as reachable on September 10, 2026.

## What the project does

- Downloads daily OHLCV data for top cryptocurrencies from the CoinDesk Data API.
- Engineers volume-shock, momentum, volatility, short-term reversal, and weekly return features.
- Builds equal-weighted and value-weighted long, short, and long-short factor portfolios with PyBondLab.
- Collects CoinDesk news and scores it with VADER sentiment analysis.
- Extends VADER with a crypto-specific custom lexicon.
- Creates overall and coin-level sentiment time series, BTC sentiment strategies, and diagnostic plots.
- Tests lead/lag relationships between sentiment changes and BTC returns with cross-correlation and Granger tests.
- Dynamically combines factor portfolios using rolling risk-adjusted scores, softmax weights, concentration limits, turnover controls, and transaction costs.

## Pipeline overview

```text
CoinDesk market data
        |
        v
Stage 1: daily OHLCV
        |
        v
Stage 2: weekly factor features
        |
        v
Stage 3: long / short / long-short portfolios
        |
        v
Dynamic factor allocation and performance reports

CoinDesk news
        |
        v
Text cleaning + crypto tagging + custom VADER
        |
        v
Sentiment indices + BTC sentiment strategies
        |
        v
Cross-correlation and Granger diagnostics
```

The market-factor and news branches are separate workflows. The extended sentiment workflow downloads its own BTC history and aligns those returns with weekly sentiment series; it does not reuse the Stage 1–2 market dataset automatically.

## Repository structure

| File | Purpose |
| --- | --- |
| `crypto_pipeline.py` | Market-data Stage 1 (OHLCV download) and Stage 2 (factor engineering). |
| `run_crypto_pipeline.py` | Editable driver for market-data Stages 1–2. |
| `crypto_pipeline_stage3.py` | Cross-sectional portfolio sorting, factor return panels, performance charts, and transaction-cost analysis. |
| `run_crypto_stage3.py` | Driver for Stage 3 using a configured Stage 2 CSV path. |
| `dynamic_model.py` | Dynamic factor weighting, turnover/cost adjustment, benchmarks, plots, and performance summary. |
| `run_dynamic_model.py` | Editable driver for the dynamic model. |
| `crypto_news_pipeline.py` | Standalone four-stage news download, text cleaning, basic VADER scoring, and plots. |
| `run_crypto_news_pipeline.py` | Driver for the standalone news pipeline. |
| `sentiment_pipeline.py` | Extended crypto sentiment pipeline with tagging, custom VADER, filters, time series, and BTC strategies. |
| `run_sentiment_pipeline.py` | Driver for the extended sentiment workflow. |
| `vader_custom_lexicon.py` | Crypto-specific words and sentiment weights added to VADER. |
| `get_daily_ohlcv2.py` | Retrying OHLCV downloader used by the sentiment/BTC workflow. |
| `news_pipeline.py` | Sentiment/return merge, differencing, cross-correlation, lag correlation, and Granger tests. |
| `run_news_pipeline.py` | Driver for sentiment-versus-return diagnostics. |

## Research reports

- [Part A: Data Design and Analysis](CryptoAgent_Part_A_Data_Design_and_Analysis.pdf) - Kengtao Wu, July 24, 2025
- [Part B: Model Design and Implementation](CryptoAgent_Part_B_Model_Design_and_Implementation.pdf) - Kengtao Wu, August 6, 2025

These reports document the design rationale behind CryptoAgent. They are not required to run the Python pipeline.

## Requirements

- Python 3.10 or newer
- A CoinDesk Data API key for the market-data pipeline
- Internet access for data downloads

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy pandas requests tqdm matplotlib seaborn wordcloud nltk scikit-learn statsmodels scipy pybondlab
```

Download the NLTK resources before importing the news modules:

```bash
python -m nltk.downloader punkt punkt_tab stopwords wordnet vader_lexicon
```

There is currently no pinned dependency file, so exact reproducibility across package versions is not guaranteed.

## Quick start: market-factor workflow

### 1. Configure the data download

Open `run_crypto_pipeline.py` and set:

```python
API_KEY = "YOUR_COINDESK_API_KEY"
PAGES = [1]
TOP_LIMIT = 50
HISTORY_LIMIT = 2000
CURRENCY = "USD"
WEEK_FOLDER = "week5_crypto"
```

Do not commit a real API key. The current runner uses a Python constant; for shared or production use, replace it with an environment variable or secret manager.

### 2. Download prices and build factors

```bash
python run_crypto_pipeline.py
```

Default outputs:

```text
week5_crypto/results/clean_data/
├── stage_1_crypto_data.csv
└── stage_2_crypto_data.csv
```

Stage 2 converts daily observations to Wednesday-ending weekly observations (`W-WED`) and creates:

- log returns and weekly returns;
- volume shocks over 7, 14, 21, 28, and 42 days;
- momentum over 14, 21, 28, 42, and 90 days;
- annualized volatility over the same momentum windows;
- daily and weekly short-term reversal proxies.

Stablecoins, wrapped tokens, symbols containing `USD`, and observations resembling delisting events are filtered from the weekly panel.

### 3. Form factor portfolios

Confirm that `CSV_PATH` in `run_crypto_stage3.py` points to the Stage 2 file, then run:

```bash
python run_crypto_stage3.py
```

For each available predictor, Stage 3 uses PyBondLab to form five portfolios by default and exports equal-weighted (`EW_`) and value-weighted (`VW_`) long, short, and long-short returns.

Default outputs:

```text
week5_crypto/stage3_output/
├── clean_data/
│   ├── ls_returns.csv
│   ├── long_returns.csv
│   ├── short_returns.csv
│   └── ls_returns_net.csv
└── figures/
    ├── cumret_top5.png
    ├── mean_bars.png
    ├── vol_bars.png
    ├── sharpe_bars.png
    ├── cumret_cost_impact.png
    ├── sharpe_cost_impact.png
    └── additional factor dashboards
```

The built-in transaction-cost analysis assumes weekly rebalancing and subtracts a fixed two-sided cost from long-short returns.

### 4. Run dynamic factor allocation

Review `MODEL_CONFIG`, `BASE_RESULTS_DIR`, `BTC_RETURNS_PATH`, and `OUTPUT_DIR` in `run_dynamic_model.py`, then run:

```bash
python run_dynamic_model.py
```

The model:

1. selects either `EW_` or `VW_` factor return series;
2. computes a rolling mean-to-volatility score;
3. maps scores to non-negative weights with a row-wise softmax;
4. applies a maximum weight and a maximum period-to-period weight change;
5. shifts weights by one period to reduce look-ahead in realized returns;
6. subtracts turnover-based transaction costs; and
7. compares the result with an equal-weight baseline, an in-sample best factor, and BTC.

Outputs are written under `week5_crypto/stage3_output/dynamic_model_output/`:

- `dynamic_strategy_returns.csv`
- `table_performance_summary.csv`
- `plot_cumulative_returns.png`
- `plot_weight_heatmap.png`
- `plot_turnover_costs.png`

## Quick start: news-sentiment workflow

There are two related news workflows.

### Basic four-stage workflow

Edit the dates in `run_crypto_news_pipeline.py`, then run:

```bash
python run_crypto_news_pipeline.py
```

This workflow downloads CoinDesk articles, cleans and lemmatizes article bodies, counts common words, applies standard VADER, and produces word clouds, a confusion matrix, a classification report, and a compound-score histogram.

Default output directories:

```text
news_results/
├── news_clean_data/
└── news_figures/
```

### Extended crypto-specific sentiment workflow

The extended workflow expects `stage_1_news_raw.csv` either in the repository root or in `results/news_clean_data/`.

Review the settings at the top of `run_sentiment_pipeline.py`, then run:

```bash
python run_sentiment_pipeline.py
```

The workflow combines article titles and bodies, removes a conservative set of neutral words, counts words, tags articles by dominant crypto keyword, and applies VADER with `vader_custom_lexicon.py`. It then:

- removes weak sentiment observations using `|compound| >= THR`;
- removes articles below a daily word-count percentile;
- rescales compound sentiment from `[-1, 1]` to `[0, 100]`;
- computes length-weighted daily and weekly sentiment indices;
- creates overall and per-asset sentiment plots;
- compares predicted sentiment with the API-provided positive label; and
- constructs BTC sentiment-conditioned strategies.

The default crypto tags are BTC, ETH, XRP, USDT, BNB, SOL, USDC, ADA, DOGE, TON, Meme, and Mixed.

Main outputs under `results/` include:

```text
results/
├── news_clean_data/
│   ├── clean_news_timeseries.csv
│   ├── compound_timeseries_daily.csv
│   ├── compound_timeseries_W-WED.csv
│   └── btc_sentiment_strategies_W-WED.csv
└── news_figures/
    ├── confusion_matrix.png
    ├── classification_report.txt
    ├── cumulative-return and Sharpe plots
    └── sentiment histograms, time series, heatmaps, and gauges
```

### Sentiment/return diagnostics

After the extended workflow has produced the two weekly input files, run:

```bash
python run_news_pipeline.py
```

This creates cross-correlation, lag-correlation, dual-axis, and bidirectional Granger-test charts under `causality/`.

Granger-test significance means that lagged values improve prediction within the fitted model. It does **not** establish economic or causal truth.

## Command-line entry points

Some core modules can also be called directly:

```bash
python crypto_pipeline.py --api_key YOUR_KEY --pages 1 --top_limit 50 --history_limit 2000 --week week5_crypto

python crypto_news_pipeline.py \
  --start_dt 2025-01-01 \
  --end_dt 2025-01-10 \
  --base_dir .

python crypto_pipeline_stage3.py \
  week5_crypto/results/clean_data/stage_2_crypto_data.csv \
  --week week5_crypto \
  --hp 1 \
  --nportf 5
```

For Stage 3, the wrapper `run_crypto_stage3.py` is currently safer than direct execution; see the known limitations below.

## Important known limitations

The repository is a research prototype and has several reproducibility and correctness risks that should be resolved before relying on its results:

1. **No dependency lockfile or automated tests.** Library/API changes may break the pipeline or alter results.
2. **Hard-coded paths and configuration.** The runner scripts assume folders such as `week5_crypto` and `results/news_clean_data`.
3. **Stage 3 reversal-name mismatch.** Its default sort list requests `strev`, while Stage 2 creates `strev_daily` and `strev_weekly`; the missing factor is therefore skipped unless `sort_vars` is supplied explicitly or the code is corrected.
4. **Direct Stage 3 execution risk.** `stage3_add_transaction_cost_analysis` is defined after the module's `__main__` block. Directly running `crypto_pipeline_stage3.py` can call it before it is defined; importing it through `run_crypto_stage3.py` avoids this order problem.
5. **Invalid example rolling window.** `run_dynamic_model.py` sets `rolling_window` to `0.25`, but `dynamic_model.py` converts it to an integer. This becomes zero and does not represent a valid rolling research horizon. Use a positive integer such as `12`.
6. **Synthetic BTC fallback.** If the configured BTC return file is missing, `run_dynamic_model.py` creates random dummy returns. These must never be treated as a real benchmark or reported as empirical performance.
7. **In-sample selection bias.** Top factors and the “best single factor” are selected using full-sample mean returns. This can materially overstate out-of-sample performance.
8. **Simplified costs and risk metrics.** Transaction costs are approximate, market impact and liquidity constraints are not modeled, and Sharpe ratios assume a zero risk-free rate.
9. **API and data-quality dependence.** Coverage, symbol mapping, rate limits, field definitions, and historical revisions depend on the upstream CoinDesk service.
10. **Unused news API-key argument.** The basic news downloader accepts `api_key`, but its current HTTP request does not send that key. Authenticated access and rate-limit behavior should therefore be verified before large downloads.
11. **No execution or risk-management layer.** The project generates research signals and backtests; it does not place orders, enforce live limits, or monitor production positions.

## Recommended next improvements

- Add `requirements.txt` or `pyproject.toml` with pinned versions.
- Move secrets and runtime settings to environment variables or a validated config file.
- Fix the Stage 3 function-order and reversal-column mismatches.
- Remove the synthetic benchmark fallback and fail loudly when real BTC data is unavailable.
- Add data-schema validation and unit tests for factor calculations, temporal alignment, and look-ahead prevention.
- Replace full-sample factor selection with walk-forward or nested out-of-sample evaluation.
- Add deterministic caching and metadata for API responses so results can be reproduced.

## License

No license file is currently included. Unless a license is added, reuse and redistribution rights are not explicitly granted.
