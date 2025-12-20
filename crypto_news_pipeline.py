"""
CoinDesk news sentiment pipeline – Stages 1-4.

Stage 1 :  Pull news between start_dt and end_dt via CoinDesk API.
Stage 2 :  Clean text (stop-words, lemmatise) and count common words.
Stage 3 :  Run VADER sentiment; add score & label columns.
Stage 4 :  Word-clouds, confusion matrix, compound-score histogram.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List
from collections import Counter
import re

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
tqdm.pandas()

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def _ensure_dir(root: Path, sub: str | Path) -> Path:
    path = root / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_week_dirs(
    base_dir: str | Path | None = None,
    results_folder: str = "news_results",
    data_sub: str = "news_clean_data",
    fig_sub: str = "news_figures",
) -> dict[str, Path]:
    """
    Create (if needed) and return:
        <base_dir or cwd>/
            news_results/
                news_clean_data/
                news_figures/
    """
    root = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd().resolve()
    results_root = _ensure_dir(root, results_folder)
    return {
        "data_dir": _ensure_dir(results_root, data_sub),
        "fig_dir": _ensure_dir(results_root, fig_sub),
    }


# ---------------------------------------------------------------------------
# Stage 1 – fetch news
# ---------------------------------------------------------------------------
def fetch_news_range(
    api_key: str | None,
    start_dt: datetime,
    end_dt: datetime,
    lang: str = "EN",
) -> pd.DataFrame:
    """
    Pull CoinDesk news between *start_dt* and *end_dt* (inclusive).
    Logs the query date at each step so you can track progress.
    """
    url = "https://data-api.coindesk.com/news/v1/article/list"
    out: list[pd.DataFrame] = []

    while end_dt > start_dt:
        query_ts  = int(end_dt.timestamp())
        query_day = end_dt.strftime("%Y-%m-%d")
        logging.info("Requesting articles up to %s (UTC)", query_day)

        resp = requests.get(f"{url}?lang={lang}&to_ts={query_ts}")
        if not resp.ok:
            logging.error("Request failed with status %s", resp.status_code)
            break

        d = pd.DataFrame(resp.json()["Data"])
        if d.empty:
            logging.info("No data returned for %s – stopping loop.", query_day)
            break

        d["date"] = pd.to_datetime(d["PUBLISHED_ON"], unit="s")
        out.append(d[d["date"] >= start_dt])

        # step backward to the day before the earliest article we just received
        end_dt = datetime.utcfromtimestamp(d["PUBLISHED_ON"].min() - 1)

    news = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    logging.info("Fetched %d articles in total", len(news))
    return news



def stage1_load_news(
    api_key: str | None,
    start_dt: datetime,
    end_dt: datetime,
    data_dir: Path,
    filename: str = "stage_1_news_raw.csv",
) -> pd.DataFrame:
    tic = time.time()
    logging.info("Stage 1 – downloading news …")

    df = fetch_news_range(api_key, start_dt, end_dt)

    # keep columns / rename like original script
    drop_cols = [
        "GUID",
        "PUBLISHED_ON_NS",
        "IMAGE_URL",
        "SUBTITLE",
        "AUTHORS",
        "URL",
        "UPVOTES",
        "DOWNVOTES",
        "SCORE",
        "CREATED_ON",
        "UPDATED_ON",
        "SOURCE_DATA",
        "CATEGORY_DATA",
        "STATUS",
        "SOURCE_ID",
        "TYPE",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df.columns = df.columns.str.lower()
    other = [c for c in df.columns if c not in ["date", "id"]]
    df = df[["date", "id"] + other]

    # POSITIVE field → numeric
    if "sentiment" in df.columns:
        df["positive"] = np.where(df["sentiment"].str.upper() == "POSITIVE", 1, 0)
        df = df.drop(columns="sentiment")
    else:
        df["positive"] = np.nan  # if sentiment missing

    out = data_dir / filename
    df.to_csv(out, index=False)
    logging.info("Saved raw news -> %s (%.2f s)", out.name, time.time() - tic)
    return df


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_STOP = set(stopwords.words("english"))
_LEM = WordNetLemmatizer()


def _preprocess_text(txt: str) -> str:
    tokens = word_tokenize(str(txt))
    keep = [t for t in tokens if t.isalpha() and t not in _STOP]
    lemmas = [_LEM.lemmatize(t) for t in keep]
    return " ".join(lemmas)


def _basic_tokens(txt: str) -> list[str]:
    tok = txt.split()
    return [w for w in tok if w.isalpha()]


# ---------------------------------------------------------------------------
# Stage 2 – clean text & common words
# ---------------------------------------------------------------------------
def stage2_clean_text(
    df_raw: pd.DataFrame,
    data_dir: Path,
    max_common: int = 500,
    filename: str = "stage_2_news_clean.csv",
) -> pd.DataFrame:
    tic = time.time()
    logging.info("Stage 2 – cleaning text …")

    df = df_raw.copy()
    # choose body for analysis; rename to reviewText like original pattern
    df["reviewText"] = df["body"].progress_apply(_preprocess_text)

    # common words
    tokens = " ".join(df["reviewText"])
    counts = Counter(_basic_tokens(tokens)).most_common(max_common)
    pd.DataFrame(counts, columns=["word", "count"]).to_csv(
        data_dir / "stage_2_common_words.csv", index=False
    )

    out = data_dir / filename
    df.to_csv(out, index=False)
    logging.info("Saved cleaned data -> %s (%.2f s)", out.name, time.time() - tic)
    return df


# ---------------------------------------------------------------------------
# Stage 3 – VADER sentiment
# ---------------------------------------------------------------------------
_VADER = SentimentIntensityAnalyzer()


def _vader_scores(txt: str) -> pd.Series:
    return pd.Series(_VADER.polarity_scores(txt))


def stage3_sentiment(
    df_clean: pd.DataFrame,
    data_dir: Path,
    filename: str = "stage_3_news_sentiment.csv",
) -> pd.DataFrame:
    tic = time.time()
    logging.info("Stage 3 – running VADER …")

    df = df_clean.copy()
    df[["neg", "neu", "pos", "compound"]] = df["reviewText"].progress_apply(
        _vader_scores
    )
    df["sentiment"] = (df["compound"] > 0.05).astype(int)

    out = data_dir / filename
    df.to_csv(out, index=False)
    logging.info("Saved sentiment data -> %s (%.2f s)", out.name, time.time() - tic)
    return df


# ---------------------------------------------------------------------------
# Stage 4 – plots
# ---------------------------------------------------------------------------
def stage4_plots(sent_df: pd.DataFrame, fig_dir: Path) -> None:
    tic = time.time()
    logging.info("Stage 4 – creating plots …")

    # Word clouds (token-level filter)
    sid = SentimentIntensityAnalyzer()
    pos_tokens = word_tokenize(
        " ".join(sent_df[sent_df.sentiment == 1]["reviewText"])
    )
    neg_tokens = word_tokenize(
        " ".join(sent_df[sent_df.sentiment == 0]["reviewText"])
    )
    pos_words = [w for w in pos_tokens if sid.polarity_scores(w)["compound"] >= 0.1]
    neg_words = [w for w in neg_tokens if sid.polarity_scores(w)["compound"] <= -0.1]

    wc_args = dict(width=800, height=400, background_color="white")
    for tag, words in [("positive", pos_words), ("negative", neg_words)]:
        WordCloud(**wc_args).generate(" ".join(words)).to_file(
            str(fig_dir / f"wordcloud_{tag}.png")
        )

    # Confusion matrix if ground truth present
    if "positive" in sent_df.columns and sent_df["positive"].notna().any():
        cm = confusion_matrix(sent_df["positive"], sent_df["sentiment"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig(fig_dir / "confusion_matrix.png", bbox_inches="tight")
        plt.close()

        rep = classification_report(sent_df["positive"], sent_df["sentiment"])
        (fig_dir / "classification_report.txt").write_text(rep)

    # Compound-score histogram
    plt.figure(figsize=(8, 4))
    plt.hist(
        sent_df[sent_df["compound"] >= 0.05]["compound"],
        bins=50,
        alpha=0.7,
        density=True,
        color="green",
        label="compound ≥ 0.05",
    )
    plt.hist(
        sent_df[sent_df["compound"] < 0.05]["compound"],
        bins=50,
        alpha=0.7,
        density=True,
        color="red",
        label="compound < 0.05",
    )
    plt.title("Distribution of VADER Compound Scores")
    plt.xlabel("Compound score")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(fig_dir / "compound_distribution.png", bbox_inches="tight")
    plt.close()

    logging.info("Plots saved to %s (%.2f s)", fig_dir, time.time() - tic)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(
        prog="news_pipeline",
        description="CoinDesk news sentiment pipeline (Stages 1-4)",
    )
    p.add_argument("--api_key", default=None, help="CoinDesk API key (optional)")
    p.add_argument("--start_dt", default="2025-01-01", help="YYYY-MM-DD (UTC)")
    p.add_argument("--end_dt", default="2025-01-06", help="YYYY-MM-DD (UTC)")
    p.add_argument("--base_dir", default=".", help="Folder where results go")
    return p.parse_args()


def _main() -> None:  # pragma: no cover
    args = _parse_args()
    start_dt = datetime.fromisoformat(args.start_dt)
    end_dt = datetime.fromisoformat(args.end_dt)

    dirs = build_week_dirs(args.base_dir)
    df1 = stage1_load_news(args.api_key, start_dt, end_dt, dirs["data_dir"])
    df2 = stage2_clean_text(df1, dirs["data_dir"])
    df3 = stage3_sentiment(df2, dirs["data_dir"])
    stage4_plots(df3, dirs["fig_dir"])

    print("Done!  Data  ->", dirs["data_dir"].resolve())
    print("       Plots ->", dirs["fig_dir"].resolve())


if __name__ == "__main__":  # pragma: no cover
    _main()
