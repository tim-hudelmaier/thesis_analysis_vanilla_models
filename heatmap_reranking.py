import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ENGINE_REPL_DICT = {
    "comet_2020_01_4": "Comet 2020.01.4",
    "mascot_2_6_2": "Mascot 2.6.2",
    "msamanda_2_0_0_17442": "MSAmanda 2.0.0.17442",
    "msfragger_3_0": "MSFragger 3.0",
    "msgfplus_2021_03_22": "MSGF+ 2021.03.22",
    "omssa_2_1_9": "OMSSA 2.1.9",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def plot_heatmap(files: dict):
    dfs = []
    for dataset, df in files.items():
        df = pd.read_csv(df)    # , index_col=0)
        df["dataset"] = dataset
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    colors_row = {
        "0": (94 / 255, 60 / 255, 153 / 255),
        "1": (253 / 255, 204 / 255, 138 / 255),
        "2": (252 / 255, 141 / 255, 89 / 255),
        "3": (215 / 255, 48 / 255, 31 / 255),
        ">= 4": (165 / 255, 0 / 255, 38 / 255),
    }

    q_val_cols = [c for c in df.columns if "q-value_" in c and not "xtandem" in c]
    plt_df = df[df["dataset"] == "E13"].copy(deep=True)
    plt_df = plt_df[plt_df[q_val_cols].min(axis=1) < 0.01]

    sys.setrecursionlimit(10000000)

    for alg in ["random_forest", "xgboost"]:
        plt_df[f"Agreement {ENGINE_REPL_DICT[alg]}"] = np.nan
        mask = (
            ((plt_df[q_val_cols[:-1]] < 0.01).sum(axis=1) >= 4)
            & (plt_df[f"q-value_{alg}"] < 0.01)
        ) | (
            ((plt_df[q_val_cols[:-1]] > 0.01).sum(axis=1) >= 4)
            & (plt_df[f"q-value_{alg}"] > 0.01)
        )
        plt_df.loc[mask, f"Agreement {ENGINE_REPL_DICT[alg]}"] = ">= 4"
        mask = (
            ((plt_df[q_val_cols[:-1]] < 0.01).sum(axis=1) == 3)
            & (plt_df[f"q-value_{alg}"] < 0.01)
        ) | (
            ((plt_df[q_val_cols[:-1]] > 0.01).sum(axis=1) == 3)
            & (plt_df[f"q-value_{alg}"] > 0.01)
        )
        plt_df.loc[mask, f"Agreement {ENGINE_REPL_DICT[alg]}"] = "3"
        mask = (
            ((plt_df[q_val_cols[:-1]] < 0.01).sum(axis=1) == 2)
            & (plt_df[f"q-value_{alg}"] < 0.01)
        ) | (
            ((plt_df[q_val_cols[:-1]] > 0.01).sum(axis=1) == 2)
            & (plt_df[f"q-value_{alg}"] > 0.01)
        )
        plt_df.loc[mask, f"Agreement {ENGINE_REPL_DICT[alg]}"] = "2"
        mask = (
            ((plt_df[q_val_cols[:-1]] < 0.01).sum(axis=1) == 1)
            & (plt_df[f"q-value_{alg}"] < 0.01)
        ) | (
            ((plt_df[q_val_cols[:-1]] > 0.01).sum(axis=1) == 1)
            & (plt_df[f"q-value_{alg}"] > 0.01)
        )
        plt_df.loc[mask, f"Agreement {ENGINE_REPL_DICT[alg]}"] = "1"
        mask = (
            ((plt_df[q_val_cols[:-1]] < 0.01).sum(axis=1) == 0)
            & (plt_df[f"q-value_{alg}"] < 0.01)
        ) | (
            ((plt_df[q_val_cols[:-1]] > 0.01).sum(axis=1) == 0)
            & (plt_df[f"q-value_{alg}"] > 0.01)
        )
        plt_df.loc[mask, f"Agreement {ENGINE_REPL_DICT[alg]}"] = "0"

    plt_df["Agreement XGBoost vs Random Forest"] = "0"
    mask = (plt_df["q-value_xgboost"] < 0.01) & (plt_df["q-value_random_forest"] < 0.01)
    plt_df.loc[mask, "Agreement XGBoost vs Random Forest"] = "1"

    filtered_columns = [
        col
        for col in plt_df.columns
        if col.startswith("q-value_") or col.startswith("Agreement")
    ]
    new_df = plt_df.drop(
        columns=[col for col in plt_df.columns if col not in filtered_columns]
    )
    new_df.rename(
        columns={
            i: ENGINE_REPL_DICT[i.replace("q-value_", "")]
            for i in new_df.columns
            if "q-value_" in i
        },
        inplace=True,
    )

    import matplotlib.colors as pltcolors

    top = plt.get_cmap("Greens")(np.linspace(0.6, 1, 664))
    mid = plt.get_cmap("Blues")(np.linspace(0.6, 1, 232))
    low = plt.get_cmap("Greys")(np.linspace(0.8, 0.5, 432))
    colors = np.vstack((low, mid, top))

    threefold_map = pltcolors.LinearSegmentedColormap.from_list("threefold", colors)

    row_colors_comp = plt_df["Agreement XGBoost vs Random Forest"].map(colors_row)
    row_colors_xgb = plt_df["Agreement XGBoost"].map(colors_row)
    row_colors_rf = plt_df["Agreement Random Forest"].map(colors_row)
    row_colors = pd.concat([row_colors_comp, row_colors_xgb, row_colors_rf], axis=1)
    ax = sns.clustermap(
        pd.DataFrame(
            -np.log10(
                new_df.replace(0.0, 1e-5)
                # .sort_values(["Agreement_xgboost_vs_random_forest", "Agreement xgboost", "Agreement random_forest"])
                .drop(
                    columns=[
                        "Agreement XGBoost vs Random Forest",
                        "Agreement XGBoost",
                        "Agreement Random Forest",
                    ]
                )
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
        cbar_kws={"extend": "max", "label": "$-\log_10(q)$"},
        row_colors=row_colors,
        vmin=-np.log10(10e-1),
        vmax=-np.log10(10e-5),
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

    reordered_labels = plt_df.index[ax.dendrogram_row.reordered_ind].tolist()
    original_indexes = [
        1108518,  # disagree n1
        787338,  # disagree, n2
        803133,  # highest ranked xgb
        2669,  # highest ranked rf
    ]
    pos = [reordered_labels.index(i) for i in original_indexes]

    use_ticks = pos
    ax.ax_heatmap.set(yticks=use_ticks, yticklabels=list("ABCD"))
    plt.savefig("heatmap_800dpi.png", dpi=800, bbox_inches="tight")

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
