import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    print("Starting...")
    DIR = Path("thesis_results_1")

    print("Slurping Files...")
    xgb_E13 = pd.read_csv("thesis_results_1/E13_vanilla_xgboost.csv")
    rf_E13 = pd.read_csv(DIR / "E13_vanilla_random_forest.csv")
    print("Done")

    core_idx_cols = [
        "raw_data_location",
        "spectrum_id",
        "sequence",
        "modifications",
        "is_decoy",
        "protein_id",
        "charge",
    ]

    print("Renaming...")
    xgb_E13.rename(
        columns={
            col: col.replace("peptide_forest", "xgboost") for col in xgb_E13.columns
        },
        inplace=True,
    )

    rf_E13.rename(
        columns={
            col: col.replace("peptide_forest", "random_forest") for col in rf_E13.columns
        },
        inplace=True,
    )
    print("Done")

    xgb_cols = [col for col in xgb_E13.columns if "xgboost" in col]
    xgb_E13 = xgb_E13[core_idx_cols + xgb_cols]

    print("Merging...")
    merged_df = pd.merge(
        xgb_E13,
        rf_E13,
        on=core_idx_cols,
        how="outer",
        suffixes=("_xgb", "_rf")
    )

    merged_df.to_csv(DIR / "E13_merged.csv", index=False)
