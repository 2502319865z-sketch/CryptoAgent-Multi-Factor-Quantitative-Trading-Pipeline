# run_dynamic_model.py

from dynamic_model import run_dynamic_factor_weighting
from pathlib import Path
import pandas as pd
import numpy as np

# --- Configuration ---
# --- Revised configuration block ---------------------------------
MODEL_CONFIG = {
    # 1) Make the score a bit more forward-looking ------------------
    #    Shorter window ⇒ faster reaction to regime changes.
    "rolling_window": 0.25,          # ↓ from 12

    # 2) Sharpen the soft-max --------------------------------------
    #    Lower τ ⇒ higher contrast between winners & laggards.
    "temperature": 15,           # ↓ from 1.0  (values 0.4-0.7 work well)

    # 3) Allow a bit more concentration, but keep diversification ---
    "cap_w_max": 0.1,            # ↑ from 0.60

    # 4) Relax the speed-limit just enough to follow trends ---------
    "change_limit_dw_max": 0.25,  # ↑ from 0.20

    # 5) Cost assumption: still conservative but not punitive -------
    "costs_bps": 5,              # ↓ from 20  (matches ~0.12 %/week on major CEX)

    # 6) Keep using the equal-weighted legs as inputs ---------------
    "weighting_scheme": "EW"      # no change
}

# --- File Paths ---
# run_dynamic_model.py
BASE_RESULTS_DIR = Path("./week5_crypto/stage3_output/clean_data")  
BTC_RETURNS_PATH = Path("./week5_crypto/stage3_output/btc_returns_full.csv")  

OUTPUT_DIR = Path("./week5_crypto/stage3_output/dynamic_model_output")

def main():
    # 
    if not BTC_RETURNS_PATH.exists():
        print(f"Warning: BTC returns file not found at {BTC_RETURNS_PATH}. Creating a temporary dummy file.")
        idx = pd.date_range(start="2023-01-01", end="2025-12-31", freq="W")
        dummy_df = pd.DataFrame(index=idx)
        dummy_df["ret"] = np.random.normal(0.005, 0.05, len(idx))
        dummy_df.index.name = "date"
        BTC_RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        dummy_df.to_csv(BTC_RETURNS_PATH)

    run_dynamic_factor_weighting(
        base_results_dir=BASE_RESULTS_DIR,
        output_dir=OUTPUT_DIR,
        btc_returns_path=BTC_RETURNS_PATH,
        config=MODEL_CONFIG,
    )
    print("\nDynamic model execution finished successfully!")
    print(f"Check results in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
