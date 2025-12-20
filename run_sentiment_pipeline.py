"""
Run stages 2-4 of the week 9 CoinDesk pipeline.
Only one CSV is produced: week9/news_clean_data/clean_news_timeseries.csv
"""
from pathlib import Path
import matplotlib as mpl
import pandas as pd
from get_daily_ohlcv2 import get_daily_ohlcv2

from sentiment_pipeline import (
    build_week_dirs,
    stage2_add_columns,
    stage3_sentiment_and_plots,
    stage4_confusion,
    build_btc_sentiment_strategies,   # NEW
)

# --------------------------- user settings
BASE_DIR         = "."
WHITE_BACKGROUND = True     # False → dark background style
THR              = 0.05     # |compound| threshold
PC_TILE          = 25.0
RAW_FILE         = "stage_1_news_raw.csv" 
weight_col       = "n_words"
resample_rule    = "W-WED"
# ----------------------------------------

# style selector
if WHITE_BACKGROUND:
    mpl.rcdefaults()                # restore default (white bg, black text)
else:
    mpl.pyplot.style.use("dark_background")

def main() -> None:
    dirs = build_week_dirs(BASE_DIR)

    # locate & load stage-1 CSV
    candidates = [Path(BASE_DIR)/RAW_FILE, dirs["data_dir"]/RAW_FILE]
    for p in candidates:
        if p.exists():
            raw_path = p
            break
    else:
        raise FileNotFoundError(f"{RAW_FILE} not found in {candidates}")

    raw = pd.read_csv(raw_path) # Change compression if needed #
    print(f"Loaded {len(raw):,} raw articles from {raw_path}")

    # Stage 2
    clean = stage2_add_columns(raw)

    # Stage 3
    sent = stage3_sentiment_and_plots(clean, dirs, thr=THR, word_pctl=PC_TILE,
                                      weight_col=weight_col ,resample_rule= resample_rule)

    # Stage 4
    stage4_confusion(sent, dirs)
    
    btc_history = get_daily_ohlcv2(symbol="BTC", api_key=None)  # your existing call

    strategies = build_btc_sentiment_strategies(
        price_df=btc_history,
        dirs=dirs,
        rule=resample_rule,     # must match Stage 3 export
    )


    print("Done!")
    print("Final CSV ->", (dirs["data_dir"] / "clean_news_timeseries.csv").resolve())
    print("Plots     ->", dirs["fig_dir"].resolve())


if __name__ == "__main__":
    main()
