# run_news_pipeline.py
# -----------------------------------------------------------
# Executes Week-9 sentiment/return analytics and diagnostics
# -----------------------------------------------------------

from news_pipeline import (
    load_and_merge_sentiment_returns,
    compute_differences,
    run_granger_tests,
    plot_ccf_delta_sent_ret,      
    plot_corr_matrix_lags,       
    plot_dual_axis,              
)

BASE_DIR = "./results/news_clean_data"

def main():
    # ------------------ load & prepare ----------------------
    df      = load_and_merge_sentiment_returns(BASE_DIR)
    df_diff = compute_differences(df)

    # ------------------ diagnostics ------------------------
    plot_ccf_delta_sent_ret(df_diff, max_lag=8)
    plot_corr_matrix_lags(df_diff, max_lag=4)
    plot_dual_axis(df_diff)

    # ------------------ Granger tests ----------------------
    results = run_granger_tests(df_diff, max_lag=4)

    # ------------------ console summary --------------------
    print("\n--- INTERPRETATION SUMMARY ---")
    for key, res in results.items():
        print(res["summary"])

if __name__ == "__main__":
    main()
