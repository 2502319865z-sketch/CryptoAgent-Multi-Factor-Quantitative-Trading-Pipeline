# dynamic_model.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# seaborn 可选：若环境里无 seaborn，可删去两行
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except Exception:
    pass

# --- Global Settings ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # 修正：%Y-%m-%d
)

def _softmax_rows(df: pd.DataFrame, temperature: float = 1.0) -> pd.DataFrame:
    """数值稳定的逐行 softmax。"""
    Z = df.divide(max(temperature, 1e-8))
    Z = Z.sub(Z.max(axis=1), axis=0)               # 数值稳定
    E = np.exp(Z.clip(-50, 50))
    W = E.div(E.sum(axis=1), axis=0)
    return W

def _cap_and_normalize(row: pd.Series, w_max: float) -> pd.Series:
    r = row.clip(lower=0.0, upper=w_max)
    s = float(r.sum())
    if not np.isfinite(s) or s <= 0:
        n = int((~row.isna()).sum())
        return pd.Series(0 if n == 0 else 1.0 / max(n, 1), index=row.index)
    return r / s

def _apply_dw_limit(prev_w: pd.Series, target_w: pd.Series, dw_max: float) -> pd.Series:
    delta = (target_w - prev_w).clip(lower=-dw_max, upper=dw_max)
    w_new = prev_w + delta
    s = float(w_new.sum())
    if not np.isfinite(s) or s <= 0:
        n = len(w_new)
        w_new[:] = 1.0 / max(n, 1)
        return w_new
    return w_new / s

def run_dynamic_factor_weighting(
    base_results_dir: str | Path,
    output_dir: str | Path,
    btc_returns_path: str | Path,
    config: dict
) -> None:
    """
    Implements the dynamic factor-weighting model.

    Args:
        base_results_dir (str | Path): Directory containing ls_returns.csv.
        output_dir (str | Path): Directory to save results and plots.
        btc_returns_path (str | Path): CSV with BTC weekly returns (columns: date, ret).
        config (dict): Model parameters.
    """
    base_results_dir = Path(base_results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Starting Dynamic Factor-Weighting Model...")

    # --- 1) Load factor returns (long-short legs) ---
    factor_returns_path = base_results_dir / "ls_returns.csv"
    if not factor_returns_path.exists():
        raise FileNotFoundError(f"Factor returns not found at: {factor_returns_path}")

    factor_returns = pd.read_csv(factor_returns_path, index_col="date", parse_dates=True)

    weighting_scheme = str(config.get("weighting_scheme", "EW")).upper()  # "EW" or "VW"
    selected = [c for c in factor_returns.columns if c.startswith(weighting_scheme + "_")]
    if not selected:
        raise ValueError(f"No columns found starting with '{weighting_scheme}_' in {factor_returns_path.name}")

    base_portfolios = factor_returns[selected].copy()
    base_portfolios.columns = [c.replace(f"{weighting_scheme}_", "") for c in selected]
    logging.info(f"Loaded {len(base_portfolios.columns)} base portfolios ({weighting_scheme}).")

    # --- 2) Rolling score (12-week Sharpe-like), with safeguards ---
    window = int(config.get("rolling_window", 12))
    mean_w = base_portfolios.rolling(window=window, min_periods=window).mean()
    std_w  = base_portfolios.rolling(window=window, min_periods=window).std(ddof=0).replace(0, np.nan)
    score  = mean_w / std_w

    # --- 3) Raw weights via softmax ---
    temperature = float(config.get("temperature", 1.0))
    raw_weights = _softmax_rows(score, temperature=temperature)

    # --- 4) Apply path-dependent constraints ---
    cap_w_max = float(config.get("cap_w_max", 0.6))
    dw_max    = float(config.get("change_limit_dw_max", 0.2))
    constrained = raw_weights.copy()

    # 逐期处理：首行用等权或可选 raw_weights 首行
    if constrained.iloc[0].isna().all():
        constrained.iloc[0] = 1.0 / len(constrained.columns)
    else:
        constrained.iloc[0] = _cap_and_normalize(constrained.iloc[0].fillna(0.0), w_max=cap_w_max)

    for t in range(1, len(constrained)):
        row = constrained.iloc[t].fillna(0.0)
        row = _cap_and_normalize(row, w_max=cap_w_max)
        prev = constrained.iloc[t-1]
        constrained.iloc[t] = _apply_dw_limit(prev_w=prev, target_w=row, dw_max=dw_max)

    # 防前视：t 权重用于 t→t+1 的收益
    final_weights = constrained.shift(1).dropna()

    # --- 5) Combined returns, turnover & costs ---
    aligned_returns = base_portfolios.loc[final_weights.index]
    combo_gross = (final_weights * aligned_returns).sum(axis=1)

    turnover = (final_weights - final_weights.shift(1)).abs().sum(axis=1)
    turnover = turnover.fillna(turnover.iloc[1] if len(turnover) > 1 else 0.0)

    bps = float(config.get("costs_bps", 20.0))
    tx_cost = turnover * (bps / 1e4)
    combo_net = combo_gross - tx_cost

    # --- 6) Baselines ---
    ew_baseline = base_portfolios.loc[final_weights.index].mean(axis=1)

    best_name = base_portfolios.mean().idxmax()
    best_single = base_portfolios.loc[final_weights.index, best_name]
    logging.info(f"Best single factor (in-sample): {best_name}")

    btc_df = pd.read_csv(btc_returns_path, index_col="date", parse_dates=True)
    if "ret" not in btc_df.columns:
        raise ValueError(f"{btc_returns_path} must contain a 'ret' column.")
    btc_return = btc_df.loc[final_weights.index, "ret"].dropna()

    results_df = pd.DataFrame({
        "Dynamic_Net": combo_net,
        "Dynamic_Gross": combo_gross,
        "EqualWeight_Baseline": ew_baseline,
        "BestSingleFactor_InSample": best_single,
        "BTC_Benchmark": btc_return,
    }).dropna()

    results_df.to_csv(output_dir / "dynamic_strategy_returns.csv")
    logging.info(f"Saved returns to {output_dir/'dynamic_strategy_returns.csv'}")

    # --- 7) Plots ---
    # A) Cumulative NAVs (linear y; 如需对数轴，确保最小值>0再设 log)
    plt.figure(figsize=(14, 8))
    cum_nav = (1 + results_df).cumprod()
    cum_nav["Dynamic_Net"].plot(label="Dynamic Strategy (Net)", linewidth=2.5)
    cum_nav["EqualWeight_Baseline"].plot(label="Equal-Weight Baseline", linestyle="--")
    cum_nav["BestSingleFactor_InSample"].plot(label=f"Best Single Factor ({best_name})", linestyle=":")
    cum_nav["BTC_Benchmark"].plot(label="BTC Benchmark", linestyle="-.", color="grey")
    plt.title("Dynamic Strategy vs. Baselines - Cumulative NAV")
    plt.ylabel("NAV")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_cumulative_returns.png", dpi=160)
    plt.close()
    logging.info("Generated cumulative returns plot.")

    # B) Weight heatmap（matplotlib.imshow）
    try:
        import seaborn as sns
        plt.figure(figsize=(14, 8))
        sns.heatmap(final_weights.T, cmap="viridis", cbar_kws={"label": "Weight"})
        plt.title("Dynamic Factor Weights Over Time")
        plt.xlabel("Date")
        plt.ylabel("Factor")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_weight_heatmap.png", dpi=160)
        plt.close()
        logging.info("Generated weight heatmap plot.")
    except Exception:
        pass

    # C) Turnover & costs
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(turnover.index, turnover, label="Turnover")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Weekly Turnover")
    ax2 = ax1.twinx()
    ax2.plot(tx_cost.index, tx_cost * 10000, color="tab:red", label="Cost (bps)")
    ax2.set_ylabel("Transaction Cost (bps)", color="tab:red")
    ax1.grid(True, alpha=0.5); fig.tight_layout()
    plt.savefig(output_dir / "plot_turnover_costs.png", dpi=160)
    plt.close()
    logging.info("Generated turnover & costs plot.")

    # --- 8) Performance table（周频年化×52） ---
    WEEKS_PER_YEAR = 52

    def _metrics(ser: pd.Series) -> dict:
        ann_ret = ser.mean() * WEEKS_PER_YEAR
        ann_vol = ser.std(ddof=0) * np.sqrt(WEEKS_PER_YEAR)
        sharpe = (ann_ret / ann_vol) if ann_vol > 0 else np.nan
        nav = (1 + ser).cumprod()
        drawdown = nav / nav.cummax() - 1
        max_dd = drawdown.min()
        win_rate = (ser > 0).mean()
        return {
            "Annualized Return (%)": ann_ret * 100,
            "Annualized Volatility (%)": ann_vol * 100,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd * 100,
            "Win Rate (%)": win_rate * 100,
        }

    perf = {col: _metrics(results_df[col]) for col in results_df.columns}
    perf_tbl = pd.DataFrame(perf).T
    perf_tbl["Avg Weekly Turnover (%)"] = np.nan
    perf_tbl.loc["Dynamic_Net", "Avg Weekly Turnover (%)"] = turnover.mean() * 100

    logging.info("\n--- Performance Summary Table ---\n" + perf_tbl.round(2).to_string())
    perf_tbl.to_csv(output_dir / "table_performance_summary.csv")
    logging.info(f"Saved performance table to {output_dir/'table_performance_summary.csv'}")
