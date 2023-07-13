import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("old_plots/processed_results_500.csv", index_col=0)

for ds in ["E13", "E32", "E41", "E50"]:
    mask = (
        (df["dataset"] == ds) & (df["algorithm"] == "xgboost") & (df["iteration"] >= 0)
    )
    filtered_df = df[mask]

    plt.figure(figsize=(10, 6))
    sns.barplot(x="iteration", y="n_psms", data=filtered_df, errorbar="sd", capsize=0.2)

    plt.title(f"Average n_psms for Dataset {ds} and Algorithm xgboost per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Average n_psms")
    plt.savefig(f"avg_n_psms_{ds}_500est.png", dpi=400, bbox_inches="tight")
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define datasets and iterations
datasets = ["E13", "E32", "E41", "E50"]
iterations = [1, 2, 3]

# Loop through each dataset and iteration
for dataset in datasets:
    for iteration in iterations:
        # Filter the dataframe based on the current dataset and iteration
        filtered_df = df[(df['dataset'] == dataset) & (df['iteration'] == iteration)]

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(x='file', y='n_psms', data=filtered_df)

        # Annotate each bar with corresponding file name
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.2f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, 10),
                             textcoords='offset points')

        # Rotate x axis labels for readability
        plt.xticks(rotation=90)

        # Adjust y limit to make sure all annotations are visible
        plt.ylim(0, 1.1 * filtered_df['n_psms'].max())

        # Set plot title
        plt.title(f'Dataset: {dataset} | Iteration: {iteration}')

        # Save each figure separately
        plt.savefig(f"plot_{dataset}_iteration_{iteration}_500est.png", dpi=300,
                    bbox_inches='tight')

        # Show plot
        plt.show()
