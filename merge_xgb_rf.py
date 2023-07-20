import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    print("Starting...")
    DIR = Path("thesis_results_1")

    for ds in ["E13", "E32", "E41", "E50"]:
        print("Slurping Files...")
        xgb = pd.read_csv(f"thesis_results_1/{ds}_xgb")
        rf = pd.read_csv(DIR / f"{ds}_rf")
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
        xgb.rename(
            columns={
                col: col.replace("peptide_forest", "xgboost") for col in xgb.columns
            },
            inplace=True,
        )

        rf.rename(
            columns={
                col: col.replace("peptide_forest", "random_forest") for col in rf.columns
            },
            inplace=True,
        )
        print("Done")

        xgb_cols = [col for col in xgb.columns if "xgboost" in col]
        xgb = xgb[core_idx_cols + xgb_cols]

        print("Merging...")
        merged_df = pd.merge(
            xgb,
            rf,
            on=core_idx_cols,
            how="outer",
            suffixes=("_xgb", "_rf")
        )

        merged_df.to_csv(DIR / f"{ds}_merged.csv", index=False)
