"""
crypto_pipeline_stage3.py
-------------------------------------------------------------------------------
Stage 3 : Form long, short, and long-short portfolios by single-sorting on a
          list of predictors, using PyBondLab (pbl).

Inputs  : 1) A Stage 2 CSV (default: stage_2_crypto_data.csv) containing
             - date      (parseable to datetime)
             - symbol
             - ret       (weekly return decimal, not %)
             - usd_volume, btc_volume, plus predictor columns
          2) PyBondLab installed (importable as `pbl`).

Outputs : <week_folder>/stage3_output/clean_data/
              ls_returns.csv
              long_returns.csv
              short_returns.csv
          <week_folder>/stage3_output/figures/
              cumret_EW.png
              cumret_VW.png
-------------------------------------------------------------------------------
"""

from __future__ import annotations

# ---------------------------------------------------------------------------#
# Imports                                                                    #
# ---------------------------------------------------------------------------#
import argparse
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyBondLab (user must have it)
import PyBondLab as pbl  # pip install pybondlab

from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------#
# Global settings                                                            #
# ---------------------------------------------------------------------------#

# crypto_pipeline_stage3.py - 在文件顶部添加

# ... (其他全局设置) ...
TRADING_DAYS = 365
TRANSACTION_COST_BPS = 10  # 交易成本，单位是基点 (bps)。10 bps = 0.1%
# ---------------------------------------------------------------------------#



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
sns.set_theme(style="darkgrid")
plt.rcParams["figure.dpi"] = 120

TRADING_DAYS = 365       # used for annualising where needed

# ---------------------------------------------------------------------------#
# Directory helpers (shared with Stage 1–2)                                  #
# ---------------------------------------------------------------------------#
def _ensure_dir(root: Path, sub: str | Path) -> Path:
    p = root / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_stage3_dirs(
    week_folder: str | Path,
    results_folder: str = "stage3_output",
    fig_sub: str = "figures",
    data_sub: str = "clean_data",
) -> tuple[Path, Path]:
    cwd = Path.cwd().resolve()
    week_root = cwd if cwd.name == str(week_folder) else _ensure_dir(cwd, week_folder)
    res_root = _ensure_dir(week_root, results_folder)
    return _ensure_dir(res_root, fig_sub), _ensure_dir(res_root, data_sub)


# ---------------------------------------------------------------------------#
# Stage 3 function                                                           #
# ---------------------------------------------------------------------------#
def stage3_sorting_strategies(
    stage2_csv: Path | str,
    *,
    week_folder: str | Path = "week_crypto",
    sort_vars: List[str] | None = None,
    holding_period: int = 1,
    n_portf: int = 5,
) -> None:
    """
    Form long, short, and long-short portfolios for each predictor in *sort_vars*.
    Saves cumulative-return plots plus three tidy CSVs of weekly returns.
    """
    default_vars = [
        "v_28d", "v_42d", "momentum_14", "volatility_14", "momentum_21",
        "volatility_21", "momentum_28", "volatility_28", "momentum_42",
        "volatility_42", "momentum_90", "volatility_90", "strev",
    ]
    sort_vars = sort_vars or default_vars

    stage2_csv = Path(stage2_csv).expanduser().resolve()
    if not stage2_csv.exists():
        raise FileNotFoundError(stage2_csv)

    fig_dir, data_dir = build_stage3_dirs(week_folder)

    # ------------------------------------------------------------------ #
    # 1. Read and format data                                            #
    # ------------------------------------------------------------------ #
    logging.info("Reading Stage 2 file %s", stage2_csv)
    df = pd.read_csv(stage2_csv, parse_dates=["date"])

    df = (
        df.sort_values(["symbol", "date"])
          .rename(columns={"return": "ret"})
          .assign(VW=lambda d: d["usd_volume"], ID=lambda d: d["symbol"],
                  RATING_NUM=1)
    )

    strategies_ls = pd.DataFrame()
    strategies_l  = pd.DataFrame()
    strategies_s  = pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 2. Loop through sort variables                                     #
    # ------------------------------------------------------------------ #
    for var in sort_vars:
        if var not in df.columns:
            logging.warning("Skipping %s (column missing)", var)
            continue

        logging.info("Sorting on %s", var)

        single_sort = pbl.SingleSort(holding_period, var, n_portf)
        params = {"strategy": single_sort, "rating": None}

        t0 = time.time()
        results = pbl.StrategyFormation(df.copy(), **params).fit()
        logging.info("Finished %s (%.1f s)", var, time.time() - t0)

        # Long-short
        ls = pd.concat(results.get_long_short(), axis=1)
        ls.columns = [f"EW_{var}", f"VW_{var}"]
        strategies_ls = pd.concat([strategies_ls, ls], axis=1)

        # Long
        lon = pd.concat(results.get_long_leg(), axis=1)
        lon.columns = [f"EW_{var}_Long", f"VW_{var}_Long"]
        strategies_l = pd.concat([strategies_l, lon], axis=1)

        # Short
        sht = pd.concat(results.get_short_leg(), axis=1)
        sht.columns = [f"EW_{var}_Short", f"VW_{var}_Short"]
        strategies_s = pd.concat([strategies_s, sht], axis=1)

    # ------------------------------------------------------------------ #
    # 3. Save tidy return panels                                         #
    # ------------------------------------------------------------------ #
    ls_csv   = data_dir / "ls_returns.csv"
    long_csv = data_dir / "long_returns.csv"
    short_csv = data_dir / "short_returns.csv"

    strategies_ls.to_csv(ls_csv, index_label="date")
    strategies_l.to_csv(long_csv, index_label="date")
    strategies_s.to_csv(short_csv, index_label="date")

    logging.info("Saved LS  -> %s", ls_csv)
    logging.info("Saved L   -> %s", long_csv)
    logging.info("Saved S   -> %s", short_csv)

    # ------------------------------------------------------------------ #
    # 4. Performance plots                                               #
    # ------------------------------------------------------------------ #
    sns.set_theme(style="whitegrid")
    
    # Select EW columns, rank by mean return, keep top-5
    ew_df = strategies_ls[[c for c in strategies_ls.columns if c.startswith("EW_")]]
    top5_cols = ew_df.mean().sort_values(ascending=False).head(5).index.tolist()
    
    # BTC benchmark
    btc_ser = (
        df.loc[df["symbol"].str.upper() == "BTC"]
          .set_index("date")["ret"]
          .reindex(ew_df.index)
          .rename("BTC")
    )
    
    # =
    plot_df = pd.concat([ew_df[top5_cols], btc_ser], axis=1).dropna()
    
    # --- 调用新增的交易成本分析功能 ---
    # ... (前面的代码)
    
    
    # --- 调用新增的交易成本分析功能 --- (正确的缩进)
    stage3_add_transaction_cost_analysis(
        ls_returns_df=strategies_ls,
        btc_returns_ser=btc_ser,
        fig_dir=fig_dir,
        data_dir=data_dir,
        cost_bps=TRANSACTION_COST_BPS
    )

    # 4a. Cumulative-return plot
    # ... (后面的代码)
    
    # ------------------------------------------------------------------ #
    # 4a. Cumulative-return plot                                         #
    # ------------------------------------------------------------------ #
    cum = (1.0 + plot_df).cumprod() - 1.0
    plt.figure(figsize=(10, 6))
    for col in cum.columns:
        plt.plot(cum.index, cum[col], label=col)
    plt.title("Cumulative Returns – Top 5 Factors vs BTC")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "cumret_top5.png")
    plt.close()
    logging.info("Saved plot -> %s", fig_dir / "cumret_top5.png")
    
    # ------------------------------------------------------------------ #
    # 4b. Bar charts: Mean, Vol, Sharpe                                  #
    # ------------------------------------------------------------------ #
    WEEKS_PER_YEAR = 52
    ann_mean   = plot_df.mean() * WEEKS_PER_YEAR * 100.0        # percent p.a.
    ann_vol    = plot_df.std(ddof=0) * np.sqrt(WEEKS_PER_YEAR) * 100.0
    ann_sharpe = ann_mean / ann_vol
    
    metrics = {
        "mean_bars.png"  : ("Mean % p.a.",   ann_mean),
        "vol_bars.png"   : ("Vol % p.a.",    ann_vol),
        "sharpe_bars.png": ("Sharpe",        ann_sharpe),
    }
    
    base_colors = sns.color_palette("tab10", len(plot_df.columns))
    
    for fname, (title, series) in metrics.items():
        plt.figure(figsize=(8, 5))
        plt.bar(series.index, series.values, color=base_colors)
        plt.title(title + " – Top 5 Factors vs BTC")
        plt.ylabel(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_png = fig_dir / fname
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        logging.info("Saved plot -> %s", out_png)
        
        
    # Set up the plotting style
    plt.style.use('dark_background')
    sns.set_palette("husl")
    
    # Assuming your data setup
    ew_df = strategies_ls[[c for c in strategies_ls.columns if c.startswith("EW_")]]
    top5_cols = ew_df.mean().sort_values(ascending=False).head(5).index.tolist()
    top5_data = ew_df[top5_cols].copy()
    
    # Color palette for consistency across plots
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6BCF7F', '#A8E6CF']
    factor_colors = dict(zip(top5_cols, colors))
    
    # 1. PERFORMANCE HEATMAP
    def create_performance_heatmap():
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create monthly performance matrix
        monthly_data = top5_data.resample('M').mean()
        monthly_data.index = monthly_data.index.strftime('%Y-%m')
        
        # Create heatmap
        im = ax.imshow(monthly_data.T.values, cmap='RdYlGn', aspect='auto', alpha=0.8)
        
        # Customize
        ax.set_xticks(range(len(monthly_data.index)))
        ax.set_xticklabels(monthly_data.index, rotation=45, ha='right')
        ax.set_yticks(range(len(top5_cols)))
        ax.set_yticklabels([col.replace('EW_', '') for col in top5_cols])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Monthly Average Return', rotation=270, labelpad=20)
        
        # Style
        ax.set_title('Monthly Performance Heatmap - Top 5 EW Factors', 
                    fontsize=18, fontweight='bold', pad=20, color='white')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_png = fig_dir / 'ew_performance_heatmap.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
    
    # 2. CUMULATIVE RETURNS WITH GRADIENT FILL
    def create_cumulative_returns():
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Calculate cumulative returns
        cum_returns = (1 + top5_data.fillna(0)).cumprod()
        
        for i, col in enumerate(top5_cols):
            color = factor_colors[col]
            
            # Plot line
            line = ax.plot(cum_returns.index, cum_returns[col], 
                          linewidth=3, label=col.replace('EW_', ''), 
                          color=color, alpha=0.9)
            
            # Add gradient fill
            ax.fill_between(cum_returns.index, 1, cum_returns[col], 
                           color=color, alpha=0.1)
        
        # Style
        ax.set_title('Cumulative Returns - Top 5 EW Factors', 
                    fontsize=20, fontweight='bold', pad=20, color='white')
        ax.set_ylabel('Cumulative Return', fontsize=14, color='white')
        ax.set_xlabel('Date', fontsize=14, color='white')
        
        # Format dates
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        # Legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        out_png = fig_dir / 'ew_cumulative_returns.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
    
    # 3. ROLLING SHARPE RATIO DASHBOARD
    def create_rolling_sharpe_dashboard():
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1])
        
        # Main rolling Sharpe plot
        ax1 = fig.add_subplot(gs[0, :])
        
        # Calculate rolling Sharpe (252 trading days)
        rolling_sharpe = top5_data.rolling(window=63).mean() / top5_data.rolling(window=63).std() * np.sqrt(252)
        
        for col in top5_cols:
            color = factor_colors[col]
            ax1.plot(rolling_sharpe.index, rolling_sharpe[col], 
                    linewidth=2.5, label=col.replace('EW_', ''), color=color)
        
        ax1.set_title('Rolling Sharpe Ratio (1Y Window) - Top 5 EW Factors', 
                     fontsize=18, fontweight='bold', color='white')
        ax1.set_ylabel('Sharpe Ratio', fontsize=12, color='white')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='white', linestyle='-', alpha=0.5)
        
        # Volatility subplot
        ax2 = fig.add_subplot(gs[1, 0])
        vol_data = top5_data.rolling(window=63).std() * np.sqrt(252)
        for col in top5_cols:
            ax2.plot(vol_data.index, vol_data[col], 
                    linewidth=1.5, color=factor_colors[col], alpha=0.8)
        ax2.set_title('Rolling Volatility', fontsize=14, color='white')
        ax2.set_ylabel('Volatility', fontsize=10, color='white')
        ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        ax3 = fig.add_subplot(gs[1, 1])
        for i, col in enumerate(top5_cols):
            ax3.hist(top5_data[col].dropna(), alpha=0.6, bins=30, 
                    color=factor_colors[col], density=True)
        ax3.set_title('Return Distribution', fontsize=14, color='white')
        ax3.set_xlabel('Returns', fontsize=10, color='white')
        
        # Correlation heatmap
        ax4 = fig.add_subplot(gs[2, :])
        corr_matrix = top5_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, ax=ax4, cbar_kws={"shrink": .8})
        ax4.set_title('Correlation Matrix', fontsize=14, color='white')
        
        plt.tight_layout()
        out_png = fig_dir / 'ew_rolling_sharpe_dashboard.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
    
    # 4. PERFORMANCE METRICS RADAR CHART
    def create_performance_radar():
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Calculate metrics
        metrics = {}
        for col in top5_cols:
            data = top5_data[col].dropna()
            if len(data) == 0:
                continue
                
            # Calculate max drawdown safely
            cum_ret = (1 + data).cumprod()
            rolling_max = cum_ret.expanding().max()
            drawdown = (cum_ret - rolling_max) / rolling_max
            max_dd = abs(drawdown.min()) if not drawdown.empty else 0
            
            metrics[col] = {
                'Annual Return': data.mean() * 252,
                'Sharpe Ratio': data.mean() / data.std() * np.sqrt(252) if data.std() > 0 else 0,
                'Max Drawdown': max_dd,
                'Win Rate': (data > 0).mean(),
                'Volatility': data.std() * np.sqrt(252),
                'Avg Return': abs(data.mean()) * 252
            }
        
        if not metrics:
            print("No valid data for radar chart")
            return
        
        # Normalize metrics for radar chart
        metric_names = list(metrics[list(metrics.keys())[0]].keys())
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Calculate min/max for normalization with safety check
        normalized_metrics = {}
        for metric in metric_names:
            all_values = [metrics[col][metric] for col in metrics.keys() if not np.isnan(metrics[col][metric])]
            if len(all_values) == 0:
                continue
            min_val, max_val = min(all_values), max(all_values)
            
            for col in metrics.keys():
                if col not in normalized_metrics:
                    normalized_metrics[col] = []
                
                # Avoid division by zero
                if max_val - min_val == 0:
                    normalized_metrics[col].append(0.5)  # Use middle value
                else:
                    norm_val = (metrics[col][metric] - min_val) / (max_val - min_val)
                    normalized_metrics[col].append(norm_val)
        
        # Plot each factor
        for i, col in enumerate(metrics.keys()):
            if col in normalized_metrics:
                values = normalized_metrics[col]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=col.replace('EW_', ''), color=factor_colors[col])
                ax.fill(angles, values, alpha=0.1, color=factor_colors[col])
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Radar Chart', size=16, fontweight='bold', 
                    pad=30, color='white')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_png = fig_dir / 'ew_performance_radar.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
    
    # 5. DRAWDOWN ANALYSIS
    def create_drawdown_analysis():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Calculate drawdowns
        for col in top5_cols:
            cum_ret = (1 + top5_data[col].fillna(0)).cumprod()
            rolling_max = cum_ret.expanding().max()
            drawdown = (cum_ret - rolling_max) / rolling_max
            
            # Plot cumulative returns with drawdown shading
            ax1.plot(cum_ret.index, cum_ret, linewidth=2, 
                    label=col.replace('EW_', ''), color=factor_colors[col])
            
            # Shade drawdown periods
            ax1.fill_between(cum_ret.index, cum_ret, rolling_max, 
                            where=(drawdown < -0.05), alpha=0.3, 
                            color=factor_colors[col], interpolate=True)
            
            # Plot drawdown
            ax2.fill_between(drawdown.index, 0, drawdown, 
                            alpha=0.7, color=factor_colors[col], 
                            label=col.replace('EW_', ''))
        
        # Style first subplot
        ax1.set_title('Cumulative Returns with Major Drawdowns (>5%)', 
                     fontsize=16, fontweight='bold', color='white')
        ax1.set_ylabel('Cumulative Return', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Style second subplot
        ax2.set_title('Drawdown Analysis', fontsize=16, fontweight='bold', color='white')
        ax2.set_ylabel('Drawdown', color='white')
        ax2.set_xlabel('Date', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_png = fig_dir / 'ew_drawdown_analysis.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
    
    # 6. INTERACTIVE-STYLE CANDLESTICK-INSPIRED RETURNS
    def create_returns_candles():
        fig, ax = plt.subplots(figsize=(18, 10))
        
        # Create monthly aggregations for "candle" style
        monthly_stats = top5_data.resample('M').agg(['min', 'max', 'mean', 'std']).round(4)
        
        x_pos = 0
        width = 0.15
        
        for i, col in enumerate(top5_cols):
            x_positions = np.arange(len(monthly_stats)) + i * width
            
            # Get data
            lows = monthly_stats[(col, 'min')]
            highs = monthly_stats[(col, 'max')]
            opens = monthly_stats[(col, 'mean')] - monthly_stats[(col, 'std')]
            closes = monthly_stats[(col, 'mean')] + monthly_stats[(col, 'std')]
            
            color = factor_colors[col]
            
            # Draw "candles"
            for j, (low, high, open_val, close_val) in enumerate(zip(lows, highs, opens, closes)):
                x = x_positions[j]
                
                # High-low line
                ax.plot([x, x], [low, high], color=color, linewidth=1, alpha=0.7)
                
                # Body
                body_color = color if close_val >= open_val else 'red'
                body_height = abs(close_val - open_val)
                body_bottom = min(open_val, close_val)
                
                rect = patches.Rectangle((x - width/3, body_bottom), width*2/3, body_height,
                                       linewidth=1, edgecolor=color, facecolor=body_color, alpha=0.8)
                ax.add_patch(rect)
        
        # Customize
        ax.set_title('Monthly Return "Candles" - Top 5 EW Factors', 
                    fontsize=18, fontweight='bold', color='white', pad=20)
        ax.set_ylabel('Returns', fontsize=14, color='white')
        ax.set_xlabel('Month', fontsize=14, color='white')
        
        # Set x-axis
        ax.set_xticks(np.arange(len(monthly_stats)) + width * 2)
        ax.set_xticklabels([idx.strftime('%Y-%m') for idx in monthly_stats.index], rotation=45)
        
        # Legend
        legend_elements = [patches.Patch(color=factor_colors[col], label=col.replace('EW_', '')) 
                          for col in top5_cols]
        ax.legend(handles=legend_elements, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.8)
        
        plt.tight_layout()
        out_png = fig_dir / 'ew_returns_candles.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='black')
        plt.show()
    
    # Execute all plots
    print("Creating visualization suite for Top 5 EW Factors...")
    print(f"Top 5 factors: {top5_cols}")
    
    create_performance_heatmap()
    create_cumulative_returns()
    create_rolling_sharpe_dashboard()
    create_performance_radar()
    create_drawdown_analysis()
    create_returns_candles()
    
    print("\nAll plots saved successfully!")
        
       
# ---------------------------------------------------------------------------#
# CLI entry-point                                                            #
# ---------------------------------------------------------------------------#
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 crypto sorting strategies")
    p.add_argument("stage2_csv", help="Path to stage_2_crypto_data.csv")
    p.add_argument("--week", default="week_crypto", help="Week folder")
    p.add_argument("--hp", type=int, default=1, help="Holding period")
    p.add_argument("--nportf", type=int, default=5, help="Number of portfolios")
    p.add_argument(
        "--vars",
        nargs="+",
        default=None,
        help="Predictor columns to sort on (default built-in list)",
    )
    return p.parse_args()


def _main_from_cli() -> None:
    a = _parse_args()
    stage3_sorting_strategies(
        a.stage2_csv,
        week_folder=a.week,
        sort_vars=a.vars,
        holding_period=a.hp,
        n_portf=a.nportf,
    )


if __name__ == "__main__":
    _main_from_cli()


# crypto_pipeline_stage3.py - 粘贴到文件最底部

# ===========================================================================#
# 新增功能：交易成本分析 (NEW FEATURE: TRANSACTION COST ANALYSIS)         #
# ===========================================================================#
def stage3_add_transaction_cost_analysis(
    ls_returns_df: pd.DataFrame,
    btc_returns_ser: pd.Series,
    fig_dir: Path,
    data_dir: Path,
    cost_bps: int = 10
) -> None:
    """
    对多空策略的回报应用交易成本，并生成对比图表。
    """
    logging.info("Running transaction cost analysis with %d bps cost...", cost_bps)
    
    # 选取Top 5 EW因子进行分析
    ew_df = ls_returns_df[[c for c in ls_returns_df.columns if c.startswith("EW_")]].copy()
    top5_cols = ew_df.mean().sort_values(ascending=False).head(5).index.tolist()
    
    gross_returns = pd.concat([ew_df[top5_cols], btc_returns_ser], axis=1).dropna()
    
    # --- 计算净回报 ---
    # 假设多空策略每周都交易，成本是双向的（一买一卖）
    # 比特币买入持有策略只在期初买入，所以我们假设无持续交易成本
    cost_decimal = (cost_bps / 10000.0) * 2  # 乘以2代表双边成本
    
    net_returns = gross_returns.copy()
    for col in top5_cols:
        net_returns[col] = gross_returns[col] - cost_decimal
        
    net_returns.columns = [f"{col}_NET" for col in gross_returns.columns]
    
    # 保存净回报CSV
    net_csv_path = data_dir / "ls_returns_net.csv"
    net_returns.to_csv(net_csv_path)
    logging.info("Saved NET returns -> %s", net_csv_path.name)

    # --- 绘制对比图表 ---
    
    # 1. 对比累计回报图
    cum_gross = (1 + gross_returns).cumprod()
    cum_net = (1 + net_returns).cumprod()

    plt.style.use('default') # 使用默认的白色背景以便图表清晰
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top5_cols) + 1))
    
    for i, col in enumerate(gross_returns.columns):
        ax.plot(cum_gross.index, cum_gross[col], label=f"{col} (Gross)", color=colors[i], linewidth=2)
        if col != 'BTC':
            ax.plot(cum_net.index, cum_net[f"{col}_NET"], label=f"{col} (Net)", color=colors[i], linestyle='--', linewidth=2)
    
    ax.set_title(f"Impact of Transaction Costs ({cost_bps*2} bps) on Cumulative Returns")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "cumret_cost_impact.png", dpi=150)
    plt.close()
    logging.info("Saved plot -> cumret_cost_impact.png")
    
    # 2. 对比夏普比率柱状图
    WEEKS_PER_YEAR = 52
    
    # 计算年化夏普比率
    def annualized_sharpe(series):
        return (series.mean() * WEEKS_PER_YEAR) / (series.std() * np.sqrt(WEEKS_PER_YEAR))

    sharpe_gross = gross_returns.apply(annualized_sharpe)
    sharpe_net = net_returns.apply(annualized_sharpe)
    
    sharpe_df = pd.DataFrame({'Gross Sharpe': sharpe_gross, 'Net Sharpe': sharpe_net.values})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sharpe_df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title(f"Impact of Transaction Costs on Annualized Sharpe Ratios")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_xlabel("Strategy")
    ax.tick_params(axis='x', rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "sharpe_cost_impact.png", dpi=150)
    plt.close()
    logging.info("Saved plot -> sharpe_cost_impact.png")
    
    