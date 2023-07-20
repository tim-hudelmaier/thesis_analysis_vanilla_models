import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
]

ENGINE_REPL_DICT = {
    "comet_2020_01_4": "Comet 2020.01.4",
    "mascot_2_6_2": "Mascot 2.6.2",
    "msamanda_2_0_0_17442": "MSAmanda 2.0.0.17442",
    "msfragger_3_0": "MSFragger 3.0",
    "msgfplus_2021_03_22": "MSGF+ 2021.03.22",
    "omssa_2_1_9": "OMSSA 2.1.9",
    # "peptide_forest": "PeptideForest",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def plot_psms_at_qval_threshold(
    files: dict,
    palette: list = PALETTE,
    engine_repl_dict: dict = ENGINE_REPL_DICT,
    title=None,
):
    dfs = []
    for dataset, df in files.items():
        df = pd.read_csv(df, index_col=0)
        df["dataset"] = dataset
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    plt_df = pd.melt(
        df.groupby("dataset")[[c for c in df.columns if "top_target_" in c]]
        .agg("sum")
        .reset_index(),
        value_vars=[c for c in df.columns if "top_target_" in c],
        id_vars=["dataset"],
    )
    plt_df.columns = ["Dataset", "Engine", "nPSMs with q-val <= 1%"]
    plt_df["Engine"] = (
        plt_df["Engine"].str.replace("top_target_", "").replace(engine_repl_dict)
    )
    plt_df.to_csv(f"./{title}_psms_at_t.csv", index=False)
    ax = sns.barplot(
        data=plt_df,
        x="Dataset",
        y="nPSMs with q-val <= 1%",
        hue="Engine",
        hue_order=list(engine_repl_dict.values()),
        palette=palette,
    )
    ax.set_ylim(0, 30000)
    # for p in ax.patches:
    #     ax.annotate(
    #         format(p.get_height(), ".1f"),
    #         (p.get_x() + p.get_width() / 2.0, p.get_height()),
    #         ha="center",
    #         va="center",
    #         xytext=(0, 10),
    #         textcoords="offset points",
    #     )
    # plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(f"./{title}_npsms.pdf", dpi=400, bbox_inches="tight")
    plt.show()


def plot_q_value_curve(
    files: dict,
    palette: list = PALETTE,
    engine_repl_dict: dict = ENGINE_REPL_DICT,
    title=None,
):
    dfs = []
    for dataset, df in files.items():
        df = pd.read_csv(df, index_col=0)
        df["dataset"] = dataset
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    data = []
    q_val_cols = [c for c in df.columns if "q-value_" in c]
    for ds, grp in df.groupby("dataset"):
        for x in np.logspace(-4, -1, 100):
            for engine in q_val_cols:
                data.append(
                    [
                        ds,
                        x,
                        engine_repl_dict[engine.replace("q-value_", "")],
                        len(grp[grp[engine] <= x]),
                    ]
                )
    plt_df = pd.DataFrame(
        data, columns=["Dataset", "q-value threshold", "Engine", "n PSMs"]
    )
    plt_df.to_csv(f"./{title}_q_value_lines.csv", index=False)
    ax = sns.lineplot(
        data=plt_df,
        x="q-value threshold",
        y="n PSMs",
        hue="Engine",
        hue_order=list(engine_repl_dict.values()),
        palette=palette,
    )
    ax.set_ylim(0, 30000)
    # plt.title(title)
    plt.xscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(
        f"./{title}_q_value_lines.pdf",
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    from pathlib import Path

    DIR = Path("./thesis_results_1")
    files = {
        "E13": DIR / "E13_merged.csv",
        "E32": DIR / "E32_merged.csv",
        "E41": DIR / "E41_merged.csv",
        "E50": DIR / "E50_merged.csv",
    }
    plot_psms_at_qval_threshold(files, title="xgb_v_rf")
    plot_q_value_curve(files, title="xgb_v_rf")
