from pathlib import Path
import pandas as pd

consolidated_results_5 = []
consolidated_results_500 = []

# for path in Path('new_data').glob("*"):
for file in Path("new_data").glob("*"):
    if file.name == ".DS_Store":
        continue
    # for file in path.glob('*'):
    try:
        df = pd.read_csv(file, index_col=0)
        n_psms = df[df["top_target_peptide_forest"]].shape[0]
        if "5est" in file.name:
            consolidated_results_5.append(
                {
                    # "dir": path.name,
                    "file": file.name,
                    "n_psms": n_psms,
                }
            )
        elif "500est" in file.name:
            consolidated_results_500.append(
                {
                    "file": file.name,
                    "n_psms": n_psms,
                }
            )
        # print(f"{path.name}>{file.name} | n-psms: {n_psms}")
        print(f"{file.name} | n-psms: {n_psms}")
    except Exception as e:
        print(f"Error in {file.name}: {e}")

df = pd.DataFrame(consolidated_results_5)
df.to_csv("consolidated_results_5.csv")

df = pd.DataFrame(consolidated_results_500)
df.to_csv("consolidated_results_500.csv")
