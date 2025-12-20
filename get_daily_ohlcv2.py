import requests
from typing import Optional
import pandas as pd
import logging
import time

def _headers(api_key: str) -> dict[str, str]:
    return {"authorization": f"Apikey {api_key}"}

def get_daily_ohlcv2(
    symbol: str,
    api_key: str,
    limit: int = 2000,
    currency: str = "USD",
    max_retries: int = 3,
    wait: float = 1.0,
    verbose: bool = True,          
) -> Optional[pd.DataFrame]:
    """
    Download daily OHLCV for *symbol* from the Coindesk Crypto-Compare API.

    Parameters
    ----------
    symbol        : str   – e.g. 'BTC', 'ETH'
    api_key       : str   – your Coindesk/CC API key
    limit         : int   – how many days to return (<= 2000)
    currency      : str   – quote currency (default 'USD')
    max_retries   : int   – times to retry if TIMESTAMP missing or error
    wait          : float – seconds to wait between retries

    Returns
    -------
    pd.DataFrame indexed by ['symbol', 'date'] or None if all retries fail.
    """
    BASE_URL = "https://data-api.coindesk.com"
    url = (
        f"{BASE_URL}/index/cc/v1/historical/days"
        f"?market=cadli&instrument={symbol}-{currency}"
        f"&limit={limit}&aggregate=1&fill=true&apply_mapping=true"
    )

    for attempt in range(1, max_retries + 1):
        try:
            safe_headers = {k: ('***' if k.lower() == 'authorization' else v)
                            for k, v in _headers(api_key).items()}
            logging.info("REQUEST -> GET %s | hdrs=%s", url, safe_headers)
            
            resp = requests.get(url, headers=_headers(api_key), timeout=30)

            # ----------------------- VERBOSE DIAGNOSTICS --------------------
            if verbose:
                safe_headers = {k: ("***" if k.lower() == "authorization" else v)
                                for k, v in resp.request.headers.items()}
                logging.info(
                    "[%s] HTTP %s  |  req-hdrs=%s  |  rate-remaining=%s",
                    symbol,
                    resp.status_code,
                    safe_headers,
                    resp.headers.get("x-ratelimit-remaining"),
                )
                logging.debug("[%s] raw-json=%s", symbol, resp.text[:500])
            # ----------------------------------------------------------------

            data = resp.json()

            # API-level error or missing payload
            if data.get("Response") == "Error" or "Data" not in data:
                logging.warning(
                    "No data for %s (attempt %d/%d): %s",
                    symbol,
                    attempt,
                    max_retries,
                    data.get("Message"),
                )
                raise ValueError("API response error")

            # Make sure TIMESTAMP exists; otherwise force retry
            if not data["Data"] or "TIMESTAMP" not in data["Data"][0]:
                logging.warning(
                    "TIMESTAMP missing for %s (attempt %d/%d) – retrying …",
                    symbol,
                    attempt,
                    max_retries,
                )
                raise KeyError("TIMESTAMP")

            # -----------------------------------------------------------------
            # Normal parsing path
            # -----------------------------------------------------------------
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
            df = df[
                ["date", "open", "high", "low", "close", "usd_volume", "btc_volume"]
            ].copy()

            # Convenience: express USD volume in millions
            df["usd_volume_mil"] = df["usd_volume"] / 1e6
            df["symbol"] = symbol
            df.set_index(["symbol", "date"], inplace=True)

            return df

        except (requests.RequestException, ValueError, KeyError) as exc:
            # Connection problem OR explicit retry trigger
            if attempt < max_retries:
                time.sleep(wait)
                continue
            logging.error("Failed to fetch OHLCV for %s: %s", symbol, exc)

    # All retries exhausted
    return None