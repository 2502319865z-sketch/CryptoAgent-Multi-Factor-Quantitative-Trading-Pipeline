"""
Crypto data pipeline – Stage 1 and Stage 2.

Stage 1 :  Download daily OHLCV for the top coins.
Stage 2 :  Derive volume shocks, momentum, volatility, weekly returns,
           and save a cleaned weekly data set.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Logging --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Directory helpers ----------------------------------------------------------


def _ensure_dir(root: Path, sub: str | Path) -> Path:
    """Return *root/sub* as Path, creating parents as needed."""
    path = root / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_week_dirs(
    week_folder: str | Path = "week_crypto",
    results_folder: str = "results",
    data_sub: str = "clean_data",
) -> Path:
    """Return *data_dir* inside <week_folder>/<results_folder>/…"""
    project_root = Path.cwd().resolve()
    week_root = project_root if project_root.name == str(week_folder) else _ensure_dir(
        project_root, week_folder
    )
    results_root = _ensure_dir(week_root, results_folder)
    data_dir = _ensure_dir(results_root, data_sub)
    return data_dir


# ---------------------------------------------------------------------------
# API helpers ----------------------------------------------------------------

BASE_URL = "https://data-api.coindesk.com"


def _headers(api_key: str) -> dict[str, str]:
    return {"authorization": f"Apikey {api_key}"}


def get_top_coins(
    api_key: str,
    pages: List[int],
    limit: int = 100,
    sort_by: str = "CIRCULATING_MKT_CAP_USD",
) -> List[str]:
    """Return a list of coin symbols across *pages* sorted by *sort_by*."""
    coins: List[str] = []

    for page in pages:
        url = (
            f"{BASE_URL}/asset/v1/top/list?"
            f"page={page}&page_size={limit}"
            f"&sort_by={sort_by}&sort_direction=DESC"
            "&groups=ID,BASIC,MKT_CAP"
        )
        resp = requests.get(url, headers=_headers(api_key), timeout=30)
        data = resp.json()

        if "Data" not in data or "LIST" not in data["Data"]:
            logging.warning("Page %d returned no data: %s", page, data.get("Message"))
            continue

        for coin in data["Data"]["LIST"]:
            coins.append(coin["SYMBOL"])

        logging.info("Collected %d symbols from page %d", len(data["Data"]["LIST"]), page)

    if not coins:
        raise RuntimeError("No symbols retrieved. Check API key or parameters.")

    return coins


def get_daily_ohlcv(
    symbol: str,
    api_key: str,
    limit: int = 2000,
    currency: str = "USD",
) -> pd.DataFrame | None:
    """Return daily OHLCV for *symbol* or None on error."""
    url = (
        f"{BASE_URL}/index/cc/v1/historical/days"
        f"?market=cadli&instrument={symbol}-{currency}"
        f"&limit={limit}&aggregate=1&fill=true&apply_mapping=true"
    )
    resp = requests.get(url, headers=_headers(api_key), timeout=30)
    data = resp.json()

    if data.get("Response") == "Error" or "Data" not in data:
        logging.warning("No data for %s: %s", symbol, data.get("Message"))
        return None

    df = pd.DataFrame(data["Data"])
    df["date"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
    df = df.rename(
        columns={
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "btc_volume",
            "QUOTE_VOLUME": "usd_volume",
        }
    )

    df = df[["date", "open", "high", "low", "close", "usd_volume", "btc_volume"]].copy()
    df["usd_volume_mil"] = df["usd_volume"] / 1e6
    df["symbol"] = symbol
    df.set_index(["symbol", "date"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Stage 1 – ETL --------------------------------------------------------------


def stage1_etl(
    api_key: str,
    pages: List[int],
    top_limit: int = 100,
    history_limit: int = 2000,
    currency: str = "USD",
    sleep_sec: float = 0.5,
    data_dir: Path | None = None,
    filename: str = "stage_1_crypto_data.csv",
) -> pd.DataFrame:
    """
    Download OHLCV history for the top coins and return a tidy DataFrame.
    """
    logging.info("Fetching list of top coins …")
    symbols = get_top_coins(api_key, pages, top_limit)
    logging.info("Total symbols collected: %d", len(symbols))

    all_frames: List[pd.DataFrame] = []
    for sym in symbols:
        logging.info("Downloading history for %s", sym)
        df = get_daily_ohlcv(sym, api_key, history_limit, currency)
        if df is not None:
            all_frames.append(df)
        time.sleep(sleep_sec)

    if not all_frames:
        raise RuntimeError("No historical data retrieved.")

    data = pd.concat(all_frames).sort_index()

    if data_dir is not None:
        out_path = data_dir / filename
        data.to_csv(out_path)
        logging.info("Stage 1 CSV written to %s", out_path)

    return data


# ---------------------------------------------------------------------------
# Stage 2 – feature engineering ---------------------------------------------


def stage2_feature_engineering(
    tidy_prices: pd.DataFrame | None = None,
    csv_path: Path | None = None,
    data_dir: Path | None = None,
    filename: str = "stage_2_crypto_data.csv",
) -> pd.DataFrame:
    """
    Create volume shocks, momentum, volatility, weekly returns, and
    save the cleaned weekly data set.
    """
    if tidy_prices is None:
        if csv_path is None:
            raise ValueError("Provide either tidy_prices or csv_path.")
        logging.info("Reading Stage 1 CSV from %s", csv_path)
        tidy_prices = pd.read_csv(
            csv_path, index_col=["symbol", "date"], parse_dates=["date"]
        )

    # Reset index for easier operations
    df = tidy_prices.reset_index().sort_values(["symbol", "date"]).copy()

    # Volume shocks ---------------------------------------------------------
    for m in [7, 14, 21, 28, 42]:
        rolling_mean = (
            df.groupby("symbol")["usd_volume"]
            .shift(1)
            .rolling(m, min_periods=m)
            .mean()
        )
        df[f"v_{m}d"] = np.log(df["usd_volume"]) - np.log(rolling_mean)

    # Log returns -----------------------------------------------------------
    df["log_return"] = np.log1p(
        df.groupby("symbol")["close"].pct_change()
    )
    df["log_return"] = np.where(df["log_return"] > 2, 2, df["log_return"])
    df = df.replace([-np.inf, np.inf], np.nan)
    
    # Momentum and volatility ----------------------------------------------
    for m in [14, 21, 28, 42, 90]:
        shifted = df.groupby("symbol")["log_return"].shift(7)
        df[f"momentum_{m}"] = (
            np.exp(
                shifted.rolling(m, min_periods=m).sum()
            )
            - 1.0
        )
        df[f"volatility_{m}"] = (
            df.groupby("symbol")["log_return"]
            .rolling(m, min_periods=m)
            .std()
            .reset_index(level=0, drop=True)
        ) * np.sqrt(365.0)

    # Short-term reversal proxy --------------------------------------------
    df["strev_daily"] = df["log_return"]

    # ----------------------------------------------------------------------
    # Weekly resample (Wednesday) and returns
    dfw = (
        df.set_index("date")
        .groupby("symbol")
        .resample("W-WED")
        .last()
        .droplevel("symbol")
    )

    dfw["return"] = dfw.groupby("symbol")["close"].pct_change()
    dfw["return"] = np.where(dfw["return"] > 2, 2, dfw["return"])
    dfw['strev_weekly'] = dfw["return"]

    dfw = dfw.reset_index()

    # ----------------------------------------------------------------------
    # Remove stable coins and wrapped tokens
    stable_tickers = [
        "USD", "USDT", "USDC", "TUSD", "BUSD", "PAX", "USDP", "GUSD",
        "DAI", "SUSD", "USDN", "FRAX", "USDX", "USDJ", "XUSD", "USDD",
        "UST", "USTC",
        "EUR", "EURT", "EURS", "EUROC", "SEUR", "SEUR", "SEUR", "SEUR",
        "AEUR", "EURC", "AGEUR", "PAR","PAXG", "PYUSD", "USD1", "USDE"
    ]
    wrapped_tickers = [
        "WBTC", "WETH", "WBNB", "WSTETH", "WUSDC", "WUSDT",
        "WCRO", "WFTM", "WTRX", "WCELO", "WFIL", "WGLMR",
        "WXRP", "WLTC", "WSOL", "WADA",
    ]
    
    tickers_to_drop = {t.upper() for t in stable_tickers + wrapped_tickers}
    
    # Build masks
    is_exact_drop   = dfw["symbol"].str.upper().isin(tickers_to_drop)
    has_usd_substr  = dfw["symbol"].str.upper().str.contains("USD", na=False)
    
    # Keep rows that are **not** flagged by either rule
    dfw = dfw[~(is_exact_drop | has_usd_substr)].copy()

    # Basic cleaning --------------------------------------------------------
    dfw = dfw[dfw["return"] > -1.0]           # exclude delist events
    dfw = dfw.replace([-np.inf, np.inf], np.nan)

    # Column order ----------------------------------------------------------
    col_order = [
        "date", "symbol", "return", "open", "high", "low", "close","usd_volume",
        "btc_volume", "v_7d",
        "v_14d", "v_21d", "v_28d", "v_42d", "momentum_14",
        "volatility_14", "momentum_21", "volatility_21", "momentum_28",
        "volatility_28", "momentum_42", "volatility_42", "momentum_90",
        "volatility_90", "strev_daily","strev_weekly"
    ]
    dfw = dfw[[c for c in col_order if c in dfw.columns]].copy()
    
    

    # Save ------------------------------------------------------------------
    if data_dir is not None:
        out_path = data_dir / filename
        dfw.to_csv(out_path, index=False)
        logging.info("Stage 2 CSV written to %s", out_path)

    return dfw


# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(
        prog="crypto_pipeline",
        description="Crypto ETL (Stage 1) and feature engineering (Stage 2)",
    )
    p.add_argument("--api_key", required=True, help="CryptoCompare API key")
    p.add_argument("--pages", type=int, nargs="+", default=[1], help="Pages for top coins")
    p.add_argument("--top_limit", type=int, default=100, help="Coins per page")
    p.add_argument("--history_limit", type=int, default=2000, help="Days of history")
    p.add_argument("--currency", default="USD", help="Quote currency")
    p.add_argument("--week", default="week_crypto", help="Output folder")
    return p.parse_args()


def _main() -> None:  # pragma: no cover
    args = _parse_args()
    data_dir = build_week_dirs(args.week)

    # Stage 1
    df_prices = stage1_etl(
        api_key=args.api_key,
        pages=args.pages,
        top_limit=args.top_limit,
        history_limit=args.history_limit,
        currency=args.currency,
        data_dir=data_dir,
    )

    # Stage 2
    stage2_feature_engineering(
        tidy_prices=df_prices,
        data_dir=data_dir,
    )

    print("Done!")
    print("Data ->", data_dir.resolve())


if __name__ == "__main__":  # pragma: no cover
    _main()
