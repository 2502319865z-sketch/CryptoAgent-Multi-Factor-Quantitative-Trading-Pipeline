"""
run_crypto_stage3.py
-------------------------------------------------------------------------------
Standalone driver for Stage 3 of the crypto pipeline.

HOW TO USE
----------
1. Ensure Stage 2 has produced `stage_2_crypto_data.csv`
   inside <week_folder>/results/clean_data/.
2. Adjust `CSV_PATH` below if your folder name differs.
3. Run:
       python run_crypto_stage3.py
-------------------------------------------------------------------------------
"""

from pathlib import Path
from crypto_pipeline_stage3 import stage3_sorting_strategies

# ---------------------------------------------------------------------------
# SINGLE INPUT: path to the Stage 2 crypto CSV                               #
# ---------------------------------------------------------------------------
CSV_PATH = Path("week5_crypto/results/clean_data/stage_2_crypto_data.csv").resolve()

# infer week folder (two levels up: week_crypto/)
WEEK_FOLDER = CSV_PATH.parents[2]

def main() -> None:
    stage3_sorting_strategies(
        CSV_PATH,
        week_folder=WEEK_FOLDER,
    )
    print("Stage 3 finished.")
    print("Check outputs inside",
          (WEEK_FOLDER / "stage3_output").resolve())

if __name__ == "__main__":
    main()
