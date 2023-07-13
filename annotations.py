import pandas as pd
import numpy as np

df = pd.read_csv("old_plots/consolidated_results_500.csv", index_col=0)
df["algorithm"] = "xgboost"
# df.loc[df["dir"].str.contains("rf"), "algorithm"] = "random_forest"


# add dataset column
def extract_last_exx(file_name):
    return file_name.split("_")[-2]


mask = df["file"].str.contains("vanilla")
df.loc[~mask, "dataset"] = df.loc[~mask, "file"].apply(extract_last_exx)
df.loc[mask, "dataset"] = df.loc[mask, "file"].str.slice(0, 3)
df = df[df["file"].str.len() > 3]


def get_iteration_count(filename):
    filename = filename.split("_")
    filename = [f for f in filename if f.startswith("E")]
    return len(filename)


df["iteration"] = df["file"].apply(get_iteration_count)


def get_first_exx(file_name):
    if "_E" in file_name:
        return [i for i in file_name.split("_") if i.startswith("E")][0]
    else:
        return np.nan


mask_2 = df["iteration"] == 2
df.loc[mask_2, "dataset"] = df.loc[mask_2, "file"].apply(get_first_exx)


df.loc[df["file"].str.contains("cross"), "iteration"] = -1

df.to_csv("processed_results_500.csv")

print()
