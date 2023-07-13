import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

palette = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
]


if __name__ == "__main__":
    rf_bar = pd.read_csv("fig1/rf_vanilla_psms_at_t.csv")
    xgb_bar = pd.read_csv("fig1/xgb_vanilla_psms_at_t.csv")
    rf_line = pd.read_csv("fig1/rf_vanilla_q_value_lines.csv")
    xgb_line = pd.read_csv("fig1/xgb_vanilla_q_value_lines.csv")

    rf_bar.loc[rf_bar["Engine"] == "PeptideForest", "Engine"] = "Random Forest"
    xgb_bar.loc[xgb_bar["Engine"] == "PeptideForest", "Engine"] = "XGBoost"
    rf_line.loc[rf_line["Engine"] == "PeptideForest", "Engine"] = "Random Forest"
    xgb_line.loc[xgb_line["Engine"] == "PeptideForest", "Engine"] = "XGBoost"

    xgb_bar = xgb_bar[xgb_bar["Engine"] == "XGBoost"]
    xgb_line = xgb_line[xgb_line["Engine"] == "XGBoost"]

    merged_bar = pd.concat([rf_bar, xgb_bar], ignore_index=True)
    merged_line = pd.concat([rf_line, xgb_line], ignore_index=True)

    ax = sns.lineplot(
        data=merged_line,
        x="q-value threshold",
        y="n PSMs",
        hue="Engine",
        palette=palette,
    )
    ax.set_ylim(0, 30000)
    plt.xscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    # save as svg
    plt.savefig(
        f"./vanilla_q_value_lines.pdf",
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()

    ax = sns.barplot(
        data=merged_bar,
        x="Dataset",
        y="nPSMs with q-val <= 1%",
        hue="Engine",
        palette=palette,
    )
    ax.set_ylim(0, 30000)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    # save as svg
    plt.savefig(
        f"./vanilla_psms_at_t.pdf",
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()
