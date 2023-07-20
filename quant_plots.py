import glob
import itertools
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import regex as re

engine_repl_dict = {
    "comet_2020_01_4": "Comet 2020.01.4",
    "mascot_2_6_2": "Mascot 2.6.2",
    "msamanda_2_0_0_17442": "MSAmanda 2.0.0.17442",
    "msfragger_3_0": "MSFragger 3.0",
    "msgfplus_2021_03_22": "MSGF+ 2021.03.22",
    "omssa_2_1_9": "OMSSA 2.1.9",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

files = {
    "E13": "thesis_results_1/E13_merged.csv",
    "E32": "thesis_results_1/E32_merged.csv",
    "E41": "thesis_results_1/E41_merged.csv",
    "E50": "thesis_results_1/E50_merged.csv",
}

dfs = []
for dataset, df in files.items():
    df = pd.read_csv(df)        # , index_col=0)
    df["dataset"] = dataset
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

quants = glob.glob("./quant/*.csv")
quants = {file_path.split('P0109699')[1].split('_')[0]: file_path for file_path in quants}

# Add species column
def determine_species(row):
    code = 0
    if "HUMAN" in row["protein_id"]:
        code += 1  # 2^^0
    if "ECOLI" in row["protein_id"]:
        code += 2  # 2^^1
    if "cont" in row["protein_id"]:
        code += 4  # 2^^2
    return code


df["species_code"] = df.apply(determine_species, axis=1)
#%%
tmts = ["126", "127L", "127H", "128L", "128H", "129L", "129H", "130L", "130H", "131L"]

expected_values = {
    "126": (0, 1),
    "127L": (1, 1),
    "127H": (1, 0),
    "128L": (0.5, 1),
    "128H": (0.5, 0),
    "129L": (0.2, 1),
    "129H": (0.2, 0),
    "130L": (0.1, 1),
    "130H": (0.1, 0),
    "131L": (1, 1),
    # label: (EColi, Human)
}
#%%
dfs = []
for ds, csv in quants.items():
    quant_df = pd.read_csv(csv, index_col=0)
    quant_df["raw_data_location"] = df[df["dataset"] == ds][
        "raw_data_location"
    ].unique()[0]
    dfs.append(quant_df)
quant_df = pd.concat(dfs, ignore_index=True)
#%%
quant_df = quant_df.pivot(
    index=["spectrum_id", "raw_data_location"], columns="label", values="quant_value"
)
#%%
for l1, l2 in itertools.combinations(tmts, 2):
    quant_df[f"log2({l1}/{l2})"] = np.log2(quant_df[l1] / quant_df[l2])
    quant_df[f"ion_intensity({l1}+{l2}) / 2"] = (quant_df[l1] + quant_df[l2]) / 2
    quant_df[f"max(ion_intensity({l1},{l2}))"] = quant_df[[l1, l2]].max(axis=1)
#%%
merged = pd.merge(
    df[df["dataset"] == "E13"],
    quant_df,
    on=["spectrum_id", "raw_data_location"],
    right_index=False,
)
#%%
l1 = "127L"
l2 = "130L"
target = "mascot_2_6_2"
expected = np.log2(expected_values[l1][0] / expected_values[l2][0])
print(f"Expected values: {expected:.2f}")
min_intensity_met = merged[merged[f"ion_intensity({l1}+{l2}) / 2"] > 1.58e6]
bins = [f"i-bin {i}" for i in range(1, 11)]
min_intensity_met["intensity_bin"] = pd.qcut(
    min_intensity_met[f"max(ion_intensity({l1},{l2}))"], len(bins), labels=bins
)
min_intensity_met = min_intensity_met[
    min_intensity_met["top_target_random_forest"]
    | min_intensity_met[f"top_target_{target}"]
]
min_intensity_met = min_intensity_met[
    (min_intensity_met["species_code"] == 2) & ~min_intensity_met["is_decoy"]
]

plt_df = pd.melt(
    min_intensity_met,
    id_vars=[f"log2({l1}/{l2})", "intensity_bin"],
    value_vars=["top_target_random_forest", f"top_target_{target}"],
    var_name="Engine",
)
plt_df = plt_df[plt_df["value"]]
plt_df["Engine"] = (
    plt_df["Engine"].str.replace("top_target_", "").replace(engine_repl_dict)
)
#%%
sns.violinplot(
    data=plt_df,
    x="intensity_bin",
    y=f"log2({l1}/{l2})",
    hue="Engine",
    split=True,
    scale="area",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.hlines(y=expected, color="r", linestyles="dashed", xmin=-1, xmax=10)
plt.xticks(rotation=45)
plt.savefig(
    f"./tmtquantviolin_{target}.png",
    dpi=400,
    bbox_inches="tight",
)
plt.show()
#%%
import scipy

for bin, grp in plt_df.groupby("intensity_bin"):
    funct = scipy.stats.ks_2samp
    statistic, pvalue = funct(
        grp[grp["Engine"] == "Random Forest"][f"log2({l1}/{l2})"].values,
        grp[grp["Engine"] == "Mascot 2.6.2"][f"log2({l1}/{l2})"].values,
    )
    print(pvalue < 0.01,
        f">> Kolmogorov-Smirnov Random Forest vs MSGF+ 2021.03.22 @ {bin}: p-value : {pvalue:.4f}"
    )
#%%
l1 = "127L"
l2 = "130L"
expected = np.log2(expected_values[l1][0] / expected_values[l2][0])
print(f"Expected values: {expected:.2f}")
min_intensity_met = merged[merged[f"ion_intensity({l1}+{l2}) / 2"] > 1.58e6]
bins = [f"i-bin {i}" for i in range(1, 11)]
min_intensity_met["intensity_bin"] = pd.qcut(
    min_intensity_met[f"max(ion_intensity({l1},{l2}))"], len(bins), labels=bins
)
min_intensity_met = min_intensity_met[
    min_intensity_met["top_target_random_forest"]
    | min_intensity_met["top_target_xgboost"]
]
min_intensity_met = min_intensity_met[
    (min_intensity_met["species_code"] == 2) & ~min_intensity_met["is_decoy"]
]

plt_df = pd.melt(
    min_intensity_met,
    id_vars=[f"log2({l1}/{l2})", "intensity_bin"],
    value_vars=["top_target_random_forest", "top_target_xgboost"],
    var_name="Engine",
)
plt_df = plt_df[plt_df["value"]]
plt_df["Engine"] = (
    plt_df["Engine"].str.replace("top_target_", "").replace(engine_repl_dict)
)

sns.violinplot(
    data=plt_df,
    x="intensity_bin",
    y=f"log2({l1}/{l2})",
    hue="Engine",
    split=True,
    scale="area",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.hlines(y=expected, color="r", linestyles="dashed", xmin=-1, xmax=10)
plt.xticks(rotation=45)
plt.savefig(
    "./tmtquantviolin_xgboost.png",
    dpi=400,
    bbox_inches="tight",
)
plt.show()
#%%
import scipy

for bin, grp in plt_df.groupby("intensity_bin"):
    funct = scipy.stats.ks_2samp
    statistic, pvalue = funct(
        grp[grp["Engine"] == "Random Forest"][f"log2({l1}/{l2})"].values,
        grp[grp["Engine"] == "XGBoost"][f"log2({l1}/{l2})"].values,
    )
    print(
        f">> Kolmogorov-Smirnov PeptideForest vs XGBoost @ {bin}: p-value : {pvalue:.4f}"
    )
#%%
import math
top_target_cols = [c for c in df.columns if "top_target_" in c]
#%%
heatmap_data = defaultdict(dict)
for l1, l2 in itertools.combinations(expected_values.keys(), 2):
    if 0 in expected_values[l1] + expected_values[l2]:
        continue
    expected = np.log2(expected_values[l1][0] / expected_values[l2][0])
    print(f"Expected values: {expected:.2f}")
    min_intensity_met = merged[merged[f"ion_intensity({l1}+{l2}) / 2"] > 1.58e6]
    bins = [f"i-bin {i}" for i in range(1, 11)]
    min_intensity_met["intensity_bin"] = pd.qcut(
        min_intensity_met[f"max(ion_intensity({l1},{l2}))"], len(bins), labels=bins
    )
    min_intensity_met = min_intensity_met[
        min_intensity_met[top_target_cols].any(axis=1)
    ]
    min_intensity_met = min_intensity_met[
        (min_intensity_met["species_code"] == 2) & ~min_intensity_met["is_decoy"]
    ]

    plt_df = pd.melt(
        min_intensity_met,
        id_vars=[f"log2({l1}/{l2})", "intensity_bin"],
        value_vars=top_target_cols,
        var_name="Engine",
    )
    plt_df = plt_df[plt_df["value"]]
    plt_df["Engine"] = (
        plt_df["Engine"].str.replace("top_target_", "").replace(engine_repl_dict)
    )

    for bin, grp in plt_df.groupby("intensity_bin"):
        for eng1, eng2 in itertools.combinations(engine_repl_dict.values(), 2):
            funct = scipy.stats.ks_2samp
            statistic, pvalue = funct(
                grp[grp["Engine"] == eng1][f"log2({l1}/{l2})"].values,
                grp[grp["Engine"] == eng2][f"log2({l1}/{l2})"].values,
            )
            heatmap_data[f"log2({l1}/{l2}) @ {bin}"][f"{eng1} vs {eng2}"] = pvalue
#%%

#%%
plt_df = pd.DataFrame(heatmap_data)
#%%
sort_tuples = [(int(re.search(r"\d+$", c).group()), c[:15]) for c in plt_df.columns]
sorted_column_inds = sorted(range(len(sort_tuples)), key=lambda k: sort_tuples[k])
plt_df = plt_df.iloc[:, sorted_column_inds]
#%%
from matplotlib.colors import LogNorm

plt.figure(figsize=(50, 10))
import matplotlib.ticker as tkr

formatter = tkr.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
sns.heatmap(
    data=plt_df,
    cbar=True,
    norm=LogNorm(vmin=0.0001, vmax=1.0),
    cmap=sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True),
    yticklabels=True,
    xticklabels=True,
    linewidths=0.5,
    cbar_kws={"ticks": [0.0, 0.001, 0.01, 0.05, 1.0], "format": formatter},
)
plt.savefig(
    "./pvalues_all_vs_all.png",
    dpi=400,
    bbox_inches="tight",
)