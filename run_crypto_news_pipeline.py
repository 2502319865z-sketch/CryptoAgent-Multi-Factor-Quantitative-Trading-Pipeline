"""
Execute Stages 1-4 of the CoinDesk news sentiment pipeline.
"""

from datetime import datetime
from crypto_news_pipeline import (
    stage1_load_news,
    stage2_clean_text,
    stage3_sentiment,
    stage4_plots,
    build_week_dirs,
)

# ---------------------------------------------------------------------------
# User-adjustable constants
# ---------------------------------------------------------------------------
API_KEY   = None                # optional – CoinDesk still allows public calls
START_DT  = datetime(2025, 1, 1)
END_DT    = datetime(2025, 1, 10)
BASE_DIR  = "."                 # current working directory 
# ---------------------------------------------------------------------------


def main() -> None:
    dirs = build_week_dirs(BASE_DIR)

    # Stage 1
    raw = stage1_load_news(API_KEY, START_DT, END_DT, dirs["data_dir"])

    # Stage 2
    clean = stage2_clean_text(raw, dirs["data_dir"])

    # Stage 3
    sent = stage3_sentiment(clean, dirs["data_dir"])

    # Stage 4
    stage4_plots(sent, dirs["fig_dir"])

    print("Done!")
    print("Data  ->", dirs["data_dir"].resolve())
    print("Plots ->", dirs["fig_dir"].resolve())


if __name__ == "__main__":   # required on Windows
    main()
