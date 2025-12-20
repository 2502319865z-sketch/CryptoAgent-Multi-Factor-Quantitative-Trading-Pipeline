# news_pipeline.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------- logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --------------------- data loading and prep
def load_and_merge_sentiment_returns(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir).resolve()

    sent_path = data_dir / "compound_timeseries_W-WED.csv"
    strat_path = data_dir / "btc_sentiment_strategies_W-WED.csv"

    sent = pd.read_csv(sent_path, parse_dates=["date"])[["date", "Overall", "BTC"]]
    strat = pd.read_csv(strat_path, parse_dates=["Date"])[["Date", "ret_btc"]]
    strat = strat.rename(columns={"Date": "date"})

    df = pd.merge(sent, strat, on="date", how="inner").sort_values("date")
    logging.info("Merged dataset shape: %s", df.shape)
    return df

def compute_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add first differences of sentiment indices.
    d_Overall : Overall sentiment_t − Overall sentiment_{t-1}
    d_BTC     : BTC-specific sentiment_t − BTC-specific sentiment_{t-1}
    """
    out = df.copy()
    out["d_Overall"] = out["Overall"].diff()
    out["d_BTC"]     = out["BTC"].diff()
    return out.dropna()


# --------------------- Granger causality logic
def run_granger_tests(df: pd.DataFrame, max_lag: int = 4) -> dict:
    """
    Granger-causality tests in both directions:

        d_Overall  ->  ret_btc
        d_BTC      ->  ret_btc
        ret_btc    ->  d_Overall
        ret_btc    ->  d_BTC

    A dict with results tables and text summaries is returned.
    Plots are saved to ./causality/.
    """
    pairs = [
        ("d_Overall", "ret_btc"),
        ("d_BTC",     "ret_btc"),
        ("ret_btc",   "d_Overall"),
        ("ret_btc",   "d_BTC"),
    ]

    # ensure ./causality exists
    plots_dir = Path.cwd() / "causality"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for cause, effect in pairs:
        logging.info("Running Granger test: %s -> %s", cause, effect)
        data = df[[effect, cause]].dropna()      # effect first, cause second
        gtest = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        rows = [
            (lag, res[0]["ssr_ftest"][0], res[0]["ssr_ftest"][1])
            for lag, res in gtest.items()
        ]
        stats_df = pd.DataFrame(rows, columns=["lag", "F_stat", "p_value"])

        best = stats_df.loc[stats_df["p_value"].idxmin()]
        if best["p_value"] < 0.05:
            summary = (
                f"{cause} Granger-causes {effect} "
                f"(p={best['p_value']:.4f}, lag={int(best['lag'])})"
            )
        else:
            summary = (
                f"No strong evidence that {cause} Granger-causes {effect} "
                f"(min p={best['p_value']:.4f})"
            )

        # console output
        print("\nGranger results: {0} -> {1}".format(cause, effect))
        print(stats_df.to_string(index=False, formatters={
            "F_stat": "{:.4f}".format,
            "p_value": "{:.4f}".format,
        }))
        print("Interpretation: " + summary + "\n")

        results[f"{cause}->{effect}"] = {"table": stats_df, "summary": summary}

        _plot_granger_stats(
            stats_df,
            cause=cause,
            effect=effect,
            out_dir=plots_dir
        )

    return results


# --------------------- plot
def _plot_granger_stats(
    stats_df: pd.DataFrame,
    *,
    cause: str,
    effect: str,
    out_dir: Path
) -> None:
    """Line plot of F-statistics across lags."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        stats_df["lag"], stats_df["F_stat"],
        marker="o", linewidth=2, color="#2E86AB"
    )
    plt.axhline(
        4.0, color="red", linestyle="--", alpha=0.7,
        label="approx 5% critical value"
    )
    plt.title(f"Granger causality: {cause} -> {effect}")
    plt.xlabel("Lag")
    plt.ylabel("F-statistic")
    plt.grid(alpha=0.3)
    plt.legend()

    fname = f"granger_{cause}_to_{effect}.png".replace("/", "_")
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150)
    plt.close()

# --------------------------------------------------------------------------
# EXTRA DIAGNOSTIC PLOTS
# --------------------------------------------------------------------------
import scipy.stats as ss

def _ensure_diag_dir() -> Path:
    d = Path.cwd() / "causality" / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ------------------------------------------------------------------
# Cross-correlation plot (version-agnostic)
# ------------------------------------------------------------------
def plot_ccf_delta_sent_ret(df: pd.DataFrame, max_lag: int = 8) -> None:
    """
    Stem plot of the cross-correlation function between ΔOverall
    and ret_btc, for lags –max_lag … +max_lag.
    """
    diag_dir = _ensure_diag_dir()
    x = df["d_Overall"] - df["d_Overall"].mean()
    y = df["ret_btc"]   - df["ret_btc"].mean()

    lags = list(range(-max_lag, max_lag + 1))
    ccf  = [x.corr(y.shift(-k)) for k in lags]

    plt.figure(figsize=(10, 5))
    # stem() call without unsupported kwargs
    markerline, stemlines, baseline = plt.stem(lags, ccf, basefmt=" ")
    plt.setp(markerline, marker="o", markersize=6, color="#2E86AB")
    plt.setp(stemlines, linewidth=1.2, color="#2E86AB")

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Cross-Correlation: ΔOverall vs ret_btc")
    plt.xlabel("Lag (weeks; positive = ΔOverall leads)")
    plt.ylabel("Correlation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(diag_dir / "ccf_deltaOverall_ret_btc.png", dpi=150)
    plt.close()



def plot_corr_matrix_lags(df: pd.DataFrame, max_lag: int = 4) -> None:
    """Heat-map: correlation of ret_btc with ΔOverall and ΔBTCsent at lags 1…max_lag."""
    diag_dir = _ensure_diag_dir()
    rows = ["d_Overall", "d_BTC"]
    cols = [f"Lag_{k}" for k in range(1, max_lag + 1)]
    mat  = np.zeros((len(rows), max_lag))

    for r_idx, r in enumerate(rows):
        for k in range(1, max_lag + 1):
            mat[r_idx, k - 1] = df[r].shift(k).corr(df["ret_btc"])

    plt.figure(figsize=(8, 3))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r",
                yticklabels=rows, xticklabels=cols,
                center=0.0, cbar_kws={"label": "Corr"})
    plt.title("ret_btc vs sentiment lags (positive lag = sentiment leads)")
    plt.tight_layout()
    plt.savefig(diag_dir / "corr_matrix_ret_btc_vs_sent_lags.png", dpi=150)
    plt.close()


def plot_dual_axis(df: pd.DataFrame) -> None:
    """Dual-axis plot: z-scored ΔOverall (left) vs ret_btc (right)."""
    diag_dir = _ensure_diag_dir()
    z_dOverall = ss.zscore(df["d_Overall"].values, nan_policy="omit")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["date"], z_dOverall, color="#2E86AB", label="z-score ΔOverall")
    ax1.set_ylabel("z-score ΔOverall", color="#2E86AB")
    ax1.tick_params(axis="y", labelcolor="#2E86AB")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["ret_btc"], color="#A23B72", label="ret_btc")
    ax2.set_ylabel("ret_btc", color="#A23B72")
    ax2.tick_params(axis="y", labelcolor="#A23B72")

    fig.suptitle("ΔOverall (standardised) vs BTC Weekly Return")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(diag_dir / "dual_axis_zDeltaOverall_ret_btc.png", dpi=150)
    plt.close()
