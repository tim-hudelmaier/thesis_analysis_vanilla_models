import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_heatmap(files: dict):
    dfs = []
    for dataset, df in files.items():
        df = pd.read_csv(df, index_col=0)
        df["dataset"] = dataset
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    colors_row = {
        "0": (94 / 255, 60 / 255, 153 / 255),
        "1": (253 / 255, 204 / 255, 138 / 255),
        "2": (252 / 255, 141 / 255, 89 / 255),
        ">= 3": (215 / 255, 48 / 255, 31 / 255),
    }

    q_val_cols = [c for c in df.columns if "q-value_" in c and not "xtandem" in c]
    plt_df = df[df["dataset"] == "E13"].copy(deep=True)
    plt_df = plt_df[plt_df[q_val_cols].min(axis=1) < 0.01]

    sys.setrecursionlimit(10000000)

    plt_df["Agreement"] = np.nan
    mask = (
        ((plt_df[q_val_cols[:-1]] < 0.01).sum(axis=1) >= 3)
        & (plt_df["q-value_peptide_forest"] < 0.01)
    ) | (
        ((plt_df[q_val_cols[:-1]] > 0.01).sum(axis=1) >= 3)
        & (plt_df["q-value_peptide_forest"] > 0.01)
    )
    plt_df.loc[mask, "Agreement"] = ">= 3"
    mask = (
        ((plt_df[q_val_cols[:-1]] < 0.01).sum(axis=1) == 2)
        & (plt_df["q-value_peptide_forest"] < 0.01)
    ) | (
        ((plt_df[q_val_cols[:-1]] > 0.01).sum(axis=1) == 2)
        & (plt_df["q-value_peptide_forest"] > 0.01)
    )
    plt_df.loc[mask, "Agreement"] = "2"
    mask = (
        ((plt_df[q_val_cols[:-1]] < 0.01).sum(axis=1) == 1)
        & (plt_df["q-value_peptide_forest"] < 0.01)
    ) | (
        ((plt_df[q_val_cols[:-1]] > 0.01).sum(axis=1) == 1)
        & (plt_df["q-value_peptide_forest"] > 0.01)
    )
    plt_df.loc[mask, "Agreement"] = "1"
    mask = (
        ((plt_df[q_val_cols[:-1]] < 0.01).sum(axis=1) == 0)
        & (plt_df["q-value_peptide_forest"] < 0.01)
    ) | (
        ((plt_df[q_val_cols[:-1]] > 0.01).sum(axis=1) == 0)
        & (plt_df["q-value_peptide_forest"] > 0.01)
    )
    plt_df.loc[mask, "Agreement"] = "0"

    filtered_columns = [
        col
        for col in plt_df.columns
        if col.startswith("q-value_") or col == "Agreement"
    ]
    new_df = plt_df.drop(
        columns=[col for col in plt_df.columns if col not in filtered_columns]
    )

    import matplotlib.colors as pltcolors

    top = plt.get_cmap("Greens")(np.linspace(0.6, 1, 664))
    mid = plt.get_cmap("Blues")(np.linspace(0.6, 1, 232))
    low = plt.get_cmap("Greys")(np.linspace(0.8, 0.5, 432))
    colors = np.vstack((low, mid, top))

    threefold_map = pltcolors.LinearSegmentedColormap.from_list("threefold", colors)

    row_colors = plt_df["Agreement"].map(colors_row)
    ax = sns.clustermap(
        pd.DataFrame(
            -np.log2(
                new_df.replace(0.0, 1e-5)
                .sort_values("Agreement")
                .drop(columns="Agreement")
            ),
            index=plt_df.index,
        ),
        yticklabels=False,
        cmap=threefold_map,
        col_cluster=True,
        row_cluster=True,
        cbar_pos=(1.05, 0.2, 0.03, 0.4),
        method="ward",
        metric="euclidean",
        cbar_kws={"extend": "max", "label": "$-\log_2(q)$"},
        row_colors=row_colors,
        vmin=-np.log2(10e-1),
        vmax=-np.log2(10e-5),
    )
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c) for c in colors_row.values()]
    plt.legend(
        handles,
        colors_row,
        title="Agreement",
        bbox_to_anchor=(1.1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc="upper right",
    )

    use_ticks = [10914, 10777, 5647, 5286]
    ax.ax_heatmap.set(yticks=use_ticks, yticklabels=list("ABCD"))
    plt.savefig("heatmap.pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    DIR = Path("./thesis_results_1")
    # files = {
    #     "E13": DIR / "E13_vanilla_xgboost.csv",
    #     "E32": DIR / "E32_vanilla_xgboost.csv",
    #     "E41": DIR / "E41_vanilla_xgboost.csv",
    #     "E50": DIR / "E50_vanilla_xgboost.csv",
    # }
    files = {
        "E13": DIR / "E13_merged.csv",
    }
    plot_heatmap(files)
