"""
CoinDesk sentiment pipeline – week 9
====================================
Exports ONE CSV:
    week9/news_clean_data/clean_news_timeseries.csv
Steps
-----
Stage 2 : cleans 
Stage 3 : VADER on `all_text`, filters (|compound| ≥ THR and ≥ 25-pct words),
          rescale to 0-100, produce raw/thr/final histograms, daily line,
          monthly heat-map, fear-and-greed gauge; write the CSV.
"""
from __future__ import annotations
import logging, time
from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report
import re

#### Our custom dictionary ####
from vader_custom_lexicon import custom_words    


# ----------------------------------------------------------------- logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
tqdm.pandas()

# ----------------------------------------------------------------- folders
def _ensure_dir(root: Path, sub: str | Path) -> Path:
    p = root / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_week_dirs(
    base_dir: str | Path | None = None,
    results_folder: str = "results",
    data_sub: str = "news_clean_data",
    fig_sub: str = "news_figures",
) -> Dict[str, Path]:
    root = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd().resolve()
    res_root = _ensure_dir(root, results_folder)
    return {
        "data_dir": _ensure_dir(res_root, data_sub),
        "fig_dir": _ensure_dir(res_root, fig_sub),
    }

# ----------------------------------------------------------------- Stage 2
def stage2_add_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Stage-2 preprocessing:
      • Insert 'date'
      • Build and clean 'all_text'
      • Tag article by dominant crypto keyword -> 'crypto_tag'
    """

    # ------------- sentiment-neutral words to drop (safe subset) -------------
    neutral_stopwords = {
        "the", "a", "an",
        "is", "are", "was", "were", "am", "be", "been", "being",
        "it", "its", "they", "them", "their",
        "he", "him", "his", "she", "her", "hers",
        "we", "us", "our",
        "you", "your", "yours",
        "i", "me", "my", "mine",
        "in", "on", "at", "by", "with", "of", "for", "to", "from",
        "this", "that", "these", "those",
        "and", "or",
        "if", "then", "when", "while", "where",
        "which", "who", "whom", "whose",
        "what", "how",
        "because", "since", "as", "so"
    }

    def _remove_neutral_words(text: str) -> str:
        tokens = text.split()
        cleaned = []
        for tok in tokens:
            core = re.sub(r"^[^A-Za-z']+|[^A-Za-z']+$", "", tok)
            if core.lower() not in neutral_stopwords:
                cleaned.append(tok)
        return " ".join(cleaned)

    # ------------- crypto keyword mapping & tagging helpers ------------------
    crypto_map = {
        "BTC":  ["BTC", "BITCOIN", "Bitcoin"],
        "ETH":  ["ETH", "ETHEREUM", "Ethereum"],
        "XRP":  ["XRP", "RIPPLE", "Ripple"],
        "USDT": ["USDT", "TETHER", "Tether"],
        "BNB":  ["BNB", "BINANCE COIN", "Binance Coin"],
        "SOL":  ["SOL", "SOLANA", "Solana"],
        "USDC": ["USDC", "USD COIN", "USD Coin"],
        "ADA":  ["ADA", "CARDANO", "Cardano"],
        "DOGE": ["DOGE", "DOGECOIN", "Dogecoin"],
        "TON":  ["TON", "TONCOIN", "Toncoin"]
    }


    meme_terms = {
        "MEME", "SHIBA", "SHIB", "PEPE", "FLOKI",
        "MOG", "BONK", "DOGE", "DOGECOIN","Meme"
    }

    def _tag_keywords(kw: str) -> str | None:
        if pd.isna(kw):
            return None
        kw_up = kw.upper()
        hits = set()

        # top-10 tickers
        for tag, terms in crypto_map.items():
            if any(t in kw_up for t in terms):
                hits.add(tag)

        # meme category
        if any(m in kw_up for m in meme_terms):
            hits.add("Meme")

        if len(hits) == 0:
            return None
        if len(hits) == 1:
            return hits.pop()
        return "Mixed"

    # ------------------------------ pipeline ---------------------------------
    df = df_raw.copy()

    # rename and date split
    df = df.rename(columns={"date": "time"})
    df["time"] = pd.to_datetime(df["time"])
    df.insert(1, "date", df["time"].dt.date)

    # build + clean all_text
    df["all_text"] = (
        df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
    ).str.strip()
    df["all_text"] = df["all_text"].apply(_remove_neutral_words)
    
    # word count #
    df["n_words"] = df["all_text"].astype(str).str.split().str.len()
    
    # crypto tagging
    df["crypto_tag"] = df["keywords"].astype(str).apply(_tag_keywords)

    # reorder: place all_text after body, crypto_tag after keywords
    cols = list(df.columns)
    cols.insert(cols.index("body") + 1, cols.pop(cols.index("all_text")))
    cols.insert(cols.index("keywords") + 1, cols.pop(cols.index("crypto_tag")))
    return df[cols]

# ----------------------------------------------------------------- helpers
_VADER = SentimentIntensityAnalyzer()

# Update VADER here #
_VADER.lexicon.update(custom_words)   


def _vader_scores(txt: str) -> pd.Series:
    return pd.Series(_VADER.polarity_scores(txt))


def _add_word_filters(df: pd.DataFrame, pctl: float = 25.0) -> pd.DataFrame:
    """
    Adds two columns:
      * n_words   – word count of `all_text`
      * below_pXX – True if article length is below the chosen percentile
    """
    out = df.copy()
    out["n_words"] = out["all_text"].astype(str).str.split().str.len()
    thresh = out.groupby("date")["n_words"].transform(
        lambda x: np.percentile(x, pctl)
    )
    out["below_pctl"] = out["n_words"] < thresh
    return out


def _rescale(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["compound_pct"] = (out["compound"] + 1.0) * 50.0
    return out

# ----------------------------------------------------------------- weighted helpers
def _wmean(g: pd.DataFrame, weight_col: str | None) -> float:
    if weight_col and weight_col in g.columns:
        w = g[weight_col].to_numpy()
        return np.average(g["compound_pct"].to_numpy(), weights=w) if w.sum() else np.nan
    return g["compound_pct"].mean()

def _export_timeseries(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    weight_col: str | None = None,
    rule: str | None = None,
) -> None:
    """
    Build a wide time-series table: date × (Overall + each crypto_tag),
    then write daily CSV and, if `rule` is given, a resampled CSV.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ---------- per-tag daily weighted means
    tag_daily = (
        df.groupby(["date", "crypto_tag"])
          .apply(lambda g: _wmean(g, weight_col))
          .reset_index(name="compound_pct")
          .pivot(index="date", columns="crypto_tag", values="compound_pct")
    )

    # ---------- overall daily weighted mean
    overall = (
        df.groupby("date")
          .apply(lambda g: _wmean(g, weight_col))
          .rename("Overall")
          .to_frame()
    )

    daily_tbl = pd.concat([overall, tag_daily], axis=1).sort_index()
    daily_tbl.to_csv(out_dir / "compound_timeseries_daily.csv", float_format="%.4f")

    # ---------- optional resample (e.g. 'W-WED')
    if rule:
        resampled = daily_tbl.resample(rule).mean()
        resampled.to_csv(out_dir / f"compound_timeseries_{rule}.csv", float_format="%.4f")

# ----------------------------------------------------------------- Stage 3
def stage3_sentiment_and_plots(
    df_clean: pd.DataFrame,
    dirs: Dict[str, Path],
    *,
    thr: float = 0.05,
    word_pctl: float = 25.0,
    weight_col: str | None = "n_words",   # default weighting
    resample_rule: str | None = "W-WED",  # e.g. weekly on Wed; set None to skip
) -> pd.DataFrame:
    """
    Run VADER on `all_text`, apply filters, build weighted time-series,
    create plots, and save outputs.

    Parameters
    ----------
    df_clean      : DataFrame after Stage-2.
    dirs          : dict with 'data_dir' and 'fig_dir'.
    thr           : compound threshold (|compound| ≥ thr).
    word_pctl     : drop articles below this daily word-count percentile.
    weight_col    : column used as weights when averaging (None → unweighted).
    resample_rule : optional pandas offset alias for an extra resampled series
                    (e.g. 'W-WED', 'M', 'Q'); None → no resample CSV.
    Returns
    -------
    df_final : cleaned article-level DataFrame that feeds Stage-4.
    """

    tic = time.time()
    logging.info("Stage 3 – VADER, filters, plots")

    # ------------------------------------------------ sentiment scores
    df = df_clean.copy()
    df[["neg", "neu", "pos", "compound"]] = df["all_text"].progress_apply(_vader_scores)

    # filter 1 – |compound| ≥ thr
    df_thr = df.loc[df["compound"].abs() >= thr].copy()

    # filter 2 – below daily word-count percentile
    df_final = (
        _add_word_filters(df_thr, word_pctl)
        .loc[lambda d: ~d["below_pctl"]]
        .copy()
    )

    # labels & rescale
    df_final["sentiment"] = np.where(df_final["compound"] >= thr, "positive", "negative")
    df_final = _rescale(df_final)

    # ------------------------------------------------ plots – overall (weighted)
    _hist_grid(df,       dirs["fig_dir"] / "hist_raw.png",        "Sentiment – raw sample")
    _hist_grid(df_thr,   dirs["fig_dir"] / "hist_thr.png",        f"Sentiment – |compound| ≥ {thr}")
    _hist_grid(df_final, dirs["fig_dir"] / "hist_final.png",      "Sentiment – final sample")

    _daily_line(df_final,
                dirs["fig_dir"] / "daily_avg_sentiment.png",
                weight_col)

    _heatmap(df_final,           dirs["fig_dir"] / "monthly_avg_heatmap.png")
    _fear_greed_gauge(df_final,  dirs["fig_dir"] / "fear_greed_gauge.png")  # unweighted

    # ------------------------------------------------ plots – per crypto_tag
    def _safe(tag: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", tag)

    tags = sorted([t for t in df_final["crypto_tag"].dropna().unique()])
    for tag in tags:
        m_raw   = df["crypto_tag"]       == tag
        m_thr   = df_thr["crypto_tag"]   == tag
        m_final = df_final["crypto_tag"] == tag
        if not m_final.any():
            continue  # nothing to plot

        suff = _safe(tag)

        # _hist_grid(df.loc[m_raw],
        #            dirs["fig_dir"] / f"hist_raw_{suff}.png",
        #            f"Sentiment – raw ({tag})")
        # _hist_grid(df_thr.loc[m_thr],
        #            dirs["fig_dir"] / f"hist_thr_{suff}.png",
        #            f"Sentiment – |compound| ≥ {thr} ({tag})")
        _hist_grid(df_final.loc[m_final],
                   dirs["fig_dir"] / f"hist_final_{suff}.png",
                   f"Sentiment – final ({tag})")

        _daily_line(df_final.loc[m_final],
                    dirs["fig_dir"] / f"daily_avg_sentiment_{suff}.png",
                    weight_col)

        _heatmap(df_final.loc[m_final],
                 dirs["fig_dir"] / f"monthly_avg_heatmap_{suff}.png")

        _fear_greed_gauge(df_final.loc[m_final],
                          dirs["fig_dir"] / f"fear_greed_gauge_{suff}.png")

    # ------------------------------------------------ export time-series tables
    _export_timeseries(
        df_final,
        dirs["data_dir"],
        weight_col=weight_col,
        rule=resample_rule,
    )

    # ------------------------------------------------ save cleaned article-level CSV
    out_csv = dirs["data_dir"] / "clean_news_timeseries.csv"
    df_final.to_csv(out_csv, index=False)
    logging.info("Wrote final time-series -> %s  (%.2f s)", out_csv.name, time.time() - tic)

    return df_final

# --------------------------- Stage 3 plot helpers
def _hist_grid(df: pd.DataFrame, fname: Path, title: str) -> None:
    cols  = ["compound", "pos", "neg", "neu"]
    labs  = ["Compound", "Positive", "Negative", "Neutral"]
    clrs  = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    for i, (c, l, col) in enumerate(zip(cols, labs, clrs)):
        r, cc = divmod(i, 2)
        ax[r, cc].hist(df[c], bins=50, color=col, alpha=0.7, edgecolor="white")
        ax[r, cc].set_title(l)
        ax[r, cc].grid(alpha=0.3, linewidth=0.5)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


def _daily_line(df: pd.DataFrame, fname: Path, weight_col: str | None = None) -> None:
    daily = (
        df.groupby("date")
          .apply(lambda g: _wmean(g, weight_col))
          .reset_index(name="compound_pct")
    )
    daily["date"] = pd.to_datetime(daily["date"]).sort_values()

    plt.figure(figsize=(15, 8))
    plt.plot(daily["date"], daily["compound_pct"], linewidth=1.5, color="#2E86AB")
    plt.fill_between(daily["date"], daily["compound_pct"], color="#2E86AB", alpha=0.3)
    plt.axhline(50, color="red", linestyle="--", alpha=0.7, label="Neutral (50)")
    plt.title("Daily Weighted Average Sentiment (0–100)")
    plt.xlabel("Date"); plt.ylabel("Weighted Avg Compound (%)")
    plt.grid(alpha=0.3, linewidth=0.5); plt.legend()
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.close()



def _heatmap(df: pd.DataFrame, fname: Path) -> None:
    daily = df.groupby("date")["compound_pct"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    pivot = (
        daily.groupby(["year", "month"])["compound_pct"].mean()
        .unstack(fill_value=np.nan)
        .sort_index(ascending=False)
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlBu_r", center=50,
        cbar_kws={"label": "Avg Sentiment (%)"},
        linewidths=0.5, linecolor="white",
    )
    plt.title("Monthly Average Sentiment Heat-map")
    plt.xlabel("Month"); plt.ylabel("Year")
    plt.xticks(
        ticks=np.arange(12) + 0.5,
        labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        rotation=0,
    )
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


def _fear_greed_gauge(df: pd.DataFrame, fname: Path) -> None:
    recent = df[df["date"] >= df["date"].max() - timedelta(days=6)]
    avg = recent["compound_pct"].mean()

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection="polar"))
    colors = ["#8B0000", "#FF4500", "#FFD700", "#90EE90", "#006400"]
    bounds = [0, 20, 40, 60, 80, 100]

    for i in range(5):
        t0, t1 = np.pi * (bounds[i] / 100), np.pi * (bounds[i + 1] / 100)
        ax.fill_between(np.linspace(t0, t1, 20), 0.5, 1, color=colors[i], alpha=0.8)

    for sc in [0, 25, 50, 75, 100]:
        ang = np.pi * (sc / 100)
        ax.plot([ang, ang], [0.5, 0.55], "k-", lw=1)
        ax.text(ang, 0.6, f"{sc}", ha="center", va="center", fontsize=10)

    needle = np.pi * (avg / 100)
    ax.plot([needle, needle], [0, 0.9], "k-", lw=8); ax.plot(needle, 0, "ko", ms=15)

    cat, col = _cat(avg)
    ax.text(np.pi / 2, 0.2, f"{avg:.0f}", ha="center", va="center", fontsize=60, weight="bold")
    plt.figtext(0.5, 0.15, "Last 7-day Average", ha="center", fontsize=13)
    plt.figtext(0.5, 0.10, f"Current Status: {cat}", ha="center", fontsize=15, weight="bold", color=col)

    ax.set_ylim(0, 1.3); ax.set_xlim(0, np.pi)
    ax.set_theta_zero_location("W"); ax.set_theta_direction(1)
    ax.grid(False); ax.set_rticks([]); ax.set_thetagrids([])
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


def _cat(v: float) -> tuple[str, str]:
    if v < 20: return "Extreme Fear", "#8B0000"
    if v < 40: return "Fear", "#FF4500"
    if v < 60: return "Neutral", "#FFD700"
    if v < 80: return "Greed", "#90EE90"
    return "Extreme Greed", "#006400"

# ----------------------------------------------------------------- Stage 4
def stage4_confusion(
    df_sent: pd.DataFrame,
    dirs: Dict[str, Path],
) -> None:
    """
    If a `positive` (0/1) ground-truth column exists, produce
      • overall confusion-matrix plot  + report
      • per-crypto_tag  confusion-matrix plots + reports
    """
    if "positive" not in df_sent.columns or df_sent["positive"].isna().all():
        logging.info("Stage 4 – no ground-truth labels -> skipped")
        return

    tic = time.time()
    logging.info("Stage 4 – confusion matrix")

    df = df_sent.copy()
    df["predicted_positive"] = (df["sentiment"] == "positive").astype(int)

    # --------------- helper for safe filenames
    _safe = lambda s: re.sub(r"[^A-Za-z0-9]+", "_", str(s))

    # --------------- overall matrix
    cm = confusion_matrix(df["positive"], df["predicted_positive"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
    )
    plt.title("Confusion Matrix – all tags")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(dirs["fig_dir"] / "confusion_matrix.png", dpi=150)
    plt.close()
    rep = classification_report(df["positive"], df["predicted_positive"])
    (dirs["fig_dir"] / "classification_report.txt").write_text(rep)

    # --------------- per-crypto_tag matrices
    tags = sorted([t for t in df["crypto_tag"].dropna().unique()])
    for tag in tags:
        sub = df[df["crypto_tag"] == tag]
        if sub["positive"].nunique() < 2:      # need both classes to draw a matrix
            logging.info("Stage 4 – %s skipped (needs both classes)", tag)
            continue
        cm_tag = confusion_matrix(sub["positive"], sub["predicted_positive"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_tag, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
        )
        plt.title(f"Confusion Matrix – {tag}")
        plt.ylabel("Actual"); plt.xlabel("Predicted")
        plt.tight_layout()
        fname = dirs["fig_dir"] / f"confusion_matrix_{_safe(tag)}.png"
        plt.savefig(fname, dpi=150)
        plt.close()

        rep_tag = classification_report(sub["positive"], sub["predicted_positive"])
        (dirs["fig_dir"] / f"classification_report_{_safe(tag)}.txt").write_text(rep_tag)

    logging.info("Stage 4 finished (%.2f s)", time.time() - tic)

# ----------------------------------------------------------------- Stage 5 – BTC sentiment-differential strategies
def build_btc_sentiment_strategies(
    price_df: pd.DataFrame,
    dirs: Dict[str, Path],
    *,
    rule: str = "W-WED",
) -> pd.DataFrame:
    """
    Create two BTC long/short strategies driven by changes in sentiment
    indices and output:
      • wide weekly time-series CSV
      • cumulative-return plot
      • Sharpe-ratio bar plot

    Parameters
    ----------
    price_df : DataFrame
        BTC OHLCV history (index must contain a 'date' level/column);
        only the 'close' column is used.
    dirs     : dict of Path – output folders from build_week_dirs().
    rule     : pandas resample rule (e.g., 'W-WED') – must match Stage 3 export.

    Returns
    -------
    df : DataFrame with columns
         ['close', 'ret_btc',
          'delta_all',  'pos_all',  'ret_strat_all',
          'delta_btc',  'pos_btc',  'ret_strat_btc']
         and DateTimeIndex at the chosen rule.
    """
    # ------------------------- price prep -------------------------
    px = (
        price_df
        .copy()
        .reset_index()
        .rename(columns={"date": "Date"})      # ensure a clean name
        .set_index("Date")
        .sort_index()
        [["close"]]
    )
    px_w = px.resample(rule).last().dropna()
    px_w["ret_btc"] = px_w["close"].pct_change()

    # ------------------------- sentiment --------------------------
    sent_path = dirs["data_dir"] / f"compound_timeseries_{rule}.csv"
    if not sent_path.exists():
        raise FileNotFoundError(f"Sentiment file {sent_path} not found. "
                                "Make sure Stage 3 finished successfully.")
    sent = (
        pd.read_csv(sent_path, parse_dates=["date"])
        .rename(columns={"date": "Date"})
        .set_index("Date")
        .sort_index()
    )
    
    sent = sent.shift(1)
    
    for col in ("Overall", "BTC"):
        if col not in sent.columns:
            raise KeyError(f"Column '{col}' missing from {sent_path}")

    # ------------------------- merge ------------------------------
    df = px_w.join(sent[["Overall", "BTC"]], how="inner")

    # ------------------------- signals & positions ----------------
    df["delta_all"] = df["Overall"].diff()   # t – t-1
    df["delta_btc"] = df["BTC"].diff()

    df["pos_all"] = np.where(df["delta_all"] > 0, -1, 1)
    df["pos_btc"] = np.where(df["delta_btc"] > 0, -1, 1)

    df["ret_strat_all"] = df["pos_all"].shift() * df["ret_btc"]
    df["ret_strat_btc"] = df["pos_btc"].shift() * df["ret_btc"]

    df = df.dropna(subset=["ret_strat_all", "ret_strat_btc"])

    # ------------------------- save CSV ---------------------------
    out_csv = dirs["data_dir"] / f"btc_sentiment_strategies_{rule}.csv"
    df.to_csv(out_csv, float_format="%.6f")
    logging.info("BTC strategies table -> %s", out_csv.name)

    # ------------------------- cumulative-return plot -------------
    cum = (1 + df[["ret_strat_all", "ret_strat_btc", "ret_btc"]]).cumprod() - 1
    plt.figure(figsize=(10, 6))
    plt.plot(cum.index, cum["ret_strat_all"], label="STRAT_ALL", lw=2)
    plt.plot(cum.index, cum["ret_strat_btc"], label="STRAT_BTC", lw=2)
    plt.plot(cum.index, cum["ret_btc"],        label="BTC Buy-&-Hold", lw=2, ls="--")
    plt.title("Cumulative Returns")
    plt.ylabel("Return"); plt.xlabel("Date"); plt.legend(); plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(dirs["fig_dir"] / f"cumret_{rule}.png", dpi=150)
    plt.close()

    # ------------------------- Sharpe ratio plot ------------------
    ann = np.sqrt(52)  # weekly → annual
    sharpe = {
        "STRAT_ALL": df["ret_strat_all"].mean() / df["ret_strat_all"].std() * ann,
        "STRAT_BTC": df["ret_strat_btc"].mean() / df["ret_strat_btc"].std() * ann,
        "BTC":       df["ret_btc"].mean()       / df["ret_btc"].std()       * ann,
    }
    plt.figure(figsize=(6, 4))
    plt.bar(sharpe.keys(), sharpe.values(), color=["#2E86AB", "#A23B72", "#777777"])
    plt.ylabel("Annualised Sharpe Ratio")
    plt.title("Sharpe Ratios")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(dirs["fig_dir"] / f"sharpe_{rule}.png", dpi=150)
    plt.close()

    logging.info("BTC strategy plots written")
    return df[
        ["close", "ret_btc",
         "delta_all", "pos_all", "ret_strat_all",
         "delta_btc", "pos_btc", "ret_strat_btc"]
    ]
