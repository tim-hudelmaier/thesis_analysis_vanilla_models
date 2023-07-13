import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import pymzml
import itertools
from collections import defaultdict
from unimod_mapper import UnimodMapper
import regex as re

files = {
    "E13": "thesis_results_1/E13_merged.csv",
}
# offset = {k: None for k in files.keys()}
# for file in list(
#     glob.glob(
#         "/Users/tr341516/Downloads/paper/determine_offset_1_0_0_w10_e51ce6f2f119f03fe6ba9c584ffcc28c/*.csv"
#     )
# ):
#     offset_df = pd.read_csv(file)
#     ds = [ds for ds in offset.keys() if ds in offset_df.loc[0, "raw_data_location"]]
#     if len(ds) == 0:
#         continue
#     offset[ds[0]] = offset_df.loc[0, "offset_in_ppm_C12"]
dfs = []
for dataset, df in files.items():
    df = pd.read_csv(df) #, index_col=0)
    df["dataset"] = dataset
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
engine_repl_dict = {
    "comet_2020_01_4": "Comet 2020.01.4",
    "mascot_2_6_2": "Mascot 2.6.2",
    "msamanda_2_0_0_17442": "MSAmanda 2.0.0.17442",
    "msfragger_3_0": "MSFragger 3.0",
    "msgfplus_2021_03_22": "MSGF+ 2021.03.22",
    "omssa_2_1_9": "OMSSA 2.1.9",
    "xtandem_alanine": "X!Tandem Alanine",
    "peptide_forest": "PeptideForest",
}
PROTON = 1.00727646677
NEUTRAL_LOSSES = {
    "A": [{}],
    "C": [{}],
    "D": [{"name": "-H2O", "cc": {"H": -2, "O": -1}, "mass": -18.0105646844}, {}],
    "E": [{"name": "-H2O", "cc": {"H": -2, "O": -1}, "mass": -18.0105646844}, {}],
    "F": [{}],
    "G": [{}],
    "H": [
        {
            "name": "+H2O",
            "cc": {"H": 2, "O": 1},
            "available_in_series": ["b"],
            "mass": 18.0105646844,
        },
        {},
    ],
    "I": [{}],
    "K": [
        {"name": "-NH3", "cc": {"N": -1, "H": -3}, "mass": -17.0265491006},
        {
            "name": "+H2O",
            "cc": {"H": 2, "O": 1},
            "available_in_series": ["b"],
            "mass": 18.0105646844,
        },
        {},
    ],
    "L": [{}],
    "M": [
        {
            "name": "-SOCH4",
            "requires_unimod": ["Oxidation"],
            "cc": {"H": -4, "C": -1, "O": -1, "S": -1},
            "mass": -63.9982859228,
        },
        {},
    ],
    "N": [{"name": "-NH3", "cc": {"N": -1, "H": -3}, "mass": -17.0265491006}, {}],
    "P": [{}],
    "Q": [{"name": "-NH3", "cc": {"N": -1, "H": -3}, "mass": -17.0265491006}, {}],
    "R": [
        {"name": "-NH3", "cc": {"N": -1, "H": -3}, "mass": -17.0265491006},
        {
            "name": "+H2O",
            "cc": {"H": 2, "O": 1},
            "available_in_series": ["b"],
            "mass": 18.0105646844,
        },
        {},
    ],
    "S": [
        {
            "name": "-P",
            "requires_unimod": ["Phospho"],
            "cc": {"H": -3, "O": -4, "P": -1},
            "mass": -97.9768955746,
        },
        {
            "name": "-H2O",
            "requires_unimod": [""],
            "cc": {"H": -2, "O": -1},
            "mass": -18.0105646844,
        },
        {},
    ],
    "T": [
        {
            "name": "-P",
            "requires_unimod": ["Phospho"],
            "cc": {"H": -3, "O": -4, "P": -1},
            "mass": -97.9768955746,
        },
        {"name": "-H2O", "cc": {"H": -2, "O": -1}, "mass": -18.0105646844},
        {},
    ],
    "V": [{}],
    "W": [{}],
    "Y": [
        {
            "name": "-P",
            "requires_unimod": ["Phospho"],
            "cc": {"H": -3, "O": -4, "P": -1},
            "mass": -97.9768955746,
        },
        {},
    ],
}
AA_MASSES = {
    "A": 71.037113805,
    "C": 103.009184505,
    "D": 115.026943065,
    "E": 129.042593135,
    "F": 147.068413945,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084064015,
    "K": 128.094963050,
    "L": 113.084064015,
    "M": 131.040484645,
    "N": 114.042927470,
    "O": 237.147726925,
    "P": 97.052763875,
    "Q": 128.058577540,
    "R": 156.101111050,
    "S": 87.032028435,
    "T": 101.047678505,
    "U": 150.953633405,
    "V": 99.068413945,
    "W": 186.079312980,
    "Y": 163.063328575,
}


def calculate_mz(mass, charge):
    """Calculate m/z function.

    Keyword Arguments:
        mass (float): mass for calculating m/z
        charge (int): charge for calculating m/z
    Returns:
        float: calculated m/z
    """
    mass = float(mass)
    charge = int(charge)
    calc_mz = (mass + (charge * PROTON)) / charge
    return calc_mz


def _fragment_peptide(
    sequence,
    modifications,
    mod_comp,
    max_charge,
    forward_ion_types=("b",),
    reverse_ion_types=("y",),
):
    fragment_starts_forward = {"a": -27.99491462, "b": 0, "c": 17.0265491006}
    fragment_starts_reverse = {
        "x": 43.98982924,
        "y": 18.0105646844,
        "Y": 15.99491462,
        "z": 1.9918406159999993,
    }
    # Positional modifications
    modpos_strlist = modifications.split(";")
    positional_mod_masses = {}
    positional_mods = {}
    if modpos_strlist != [""]:
        for modpos in modpos_strlist:
            mod, pos = modpos.split(":")
            pos = int(pos)
            if pos in positional_mod_masses:
                positional_mod_masses[pos] += mod_comp[mod]
                positional_mods[pos] += [mod]
            else:
                positional_mod_masses[pos] = mod_comp[mod]
                positional_mods[pos] = [mod]
    else:
        positional_mod_masses = {}
    # Get positional masses
    pos_masses = [
        AA_MASSES[aa] + positional_mod_masses.get(pos + 1, 0.0)
        for pos, aa in enumerate(sequence)
    ]
    if 0 in positional_mod_masses:
        pos_masses[0] += positional_mod_masses[0]
    fragment_masses_forward = defaultdict(set)
    fragment_masses_reverse = defaultdict(set)
    # Forward ions
    for fw_ion_type in forward_ion_types:
        possible_neutral_losses = [("",(0.0),)]
        possible_nls = set(possible_neutral_losses)
        for i in range(0, len(sequence)):
            aa = sequence[i]
            for neutral_loss in NEUTRAL_LOSSES.get(aa, [{}]):
                present_unimod = positional_mods.get(i + 1, None)
                required_unimods = neutral_loss.get("requires_unimod", None)
                nl_possible = False
                if (required_unimods is not None) and (present_unimod is not None):
                    nl_possible = True
                if (required_unimods is None) or nl_possible:
                    nl_mass = neutral_loss.get("mass", 0.0)
                    if (nl_mass != 0.0) and (len(possible_neutral_losses) <= 5):
                        possible_neutral_losses += [(neutral_loss["name"], nl_mass)]
                        combs = [("".join(sorted([l[0] for l in combo])),sum([l[1] for l in combo]))
                                 for combo in itertools.chain.from_iterable(
                                itertools.combinations(possible_neutral_losses, j + 1)
                                for j in range(len(possible_neutral_losses))
                            )]
                        possible_nls = set(combs)
                    if fw_ion_type in neutral_loss.get(
                        "available_in_series", [fw_ion_type]
                    ):
                        for charge in range(1, max_charge + 1):
                            for nl in possible_nls:
                                fragment_masses_forward[fw_ion_type + str(i+1) + nl[0]].add(
                                    calculate_mz(
                                        sum(pos_masses[:i+1])
                                        + nl[1]
                                        + fragment_starts_forward[fw_ion_type],
                                        charge,
                                        )
                                )
    # Reverse ions
    for rev_ion_type in reverse_ion_types:
        possible_neutral_losses = [("",(0.0),)]
        possible_nls = set(possible_neutral_losses)
        for i in range(1, len(sequence) + 1):
            aa = sequence[-i]
            for neutral_loss in NEUTRAL_LOSSES.get(aa, [{}]):
                present_unimod = positional_mods.get(len(sequence) - 1, None)
                required_unimods = neutral_loss.get("requires_unimod", None)
                nl_possible = False
                if (required_unimods is not None) and (present_unimod is not None):
                    nl_possible = True
                if (required_unimods is None) or nl_possible:
                    nl_mass = neutral_loss.get("mass", 0.0)
                    if (nl_mass != 0.0) and (len(possible_neutral_losses) <= 5):
                        possible_neutral_losses += [(neutral_loss["name"], nl_mass)]
                        combs = [("".join(sorted([l[0] for l in combo])),sum([l[1] for l in combo]))
                            for combo in itertools.chain.from_iterable(
                                itertools.combinations(possible_neutral_losses, j + 1)
                                for j in range(len(possible_neutral_losses))
                            )]
                        possible_nls = set(combs)

                    if rev_ion_type in neutral_loss.get(
                        "available_in_series", [rev_ion_type]
                    ):
                        for charge in range(1, max_charge + 1):
                            for nl in possible_nls:
                                fragment_masses_reverse[rev_ion_type + str(i) + nl[0]].add(
                                    calculate_mz(
                                        sum(pos_masses[-i:])
                                        + nl[1]
                                        + fragment_starts_reverse[rev_ion_type],
                                        charge,
                                    )
                                )
    return fragment_masses_forward, fragment_masses_reverse


um = UnimodMapper()


def get_ions_list(l):
    pattern = re.compile(r'[by]\d+')
    b_and_y_list = []
    for item in l:
        splitted_items = re.split(r'[+;]', item)
        for splitted_item in splitted_items:
            match = pattern.match(splitted_item)
            if match:
                b_and_y_list.append(match.group())
    return b_and_y_list

def plot_fragments(sequence, highlighted_fragments):
    data = {
        'Combination': [''] + [sequence[:i] + '-' + sequence[i:] for i in
                               range(1, len(sequence))],
        'b': [len(sequence)] + [i for i in range(1, len(sequence))],
        'y': [0] + [len(sequence) - i for i in range(1, len(sequence))],
        'b_name': ['b' + str(len(sequence))] + ['b' + str(i) for i in
                                                range(1, len(sequence))],
        'y_name': [''] + ['y' + str(len(sequence) - i) for i in range(1, len(sequence))]
    }

    df = pd.DataFrame(data)

    df_melt = df[['Combination', 'b', 'y', 'b_name', 'y_name']].melt(
        ['Combination', 'b_name', 'y_name'], var_name='Fragment', value_name='Length')

    df_b = df_melt[df_melt['Fragment'] == 'b']
    df_y = df_melt[df_melt['Fragment'] == 'y']

    fig, ax = plt.subplots(figsize=(10, 6))

    barplot_b = sns.barplot(x='Length', y='Combination', data=df_b, color='lightgray',
                            orient='h', ax=ax)

    sns.barplot(x='Length', y='Combination', data=df_y, color='darkgray',
                left=df_b['Length'], orient='h', ax=ax)

    ax.grid(False)
    ax.axis('off')

    for i, combination in enumerate(df['Combination']):
        if i == 0:
            for j, char in enumerate(sequence):
                ax.text((df_b.iloc[i]['Length'] / len(sequence)) * (j + 0.5), i, char,
                        ha='center', va='center', color='black', fontsize=14)

    plt.draw()

    for i, bar in enumerate(ax.patches[:len(df_b)]):
        if df_b.iloc[i]['b_name'] in highlighted_fragments:
            bar.set_color('red')

    for i, bar in enumerate(ax.patches[len(df_b):]):
        if df_y.iloc[i]['y_name'] in highlighted_fragments:
            bar.set_color('blue')

    for i, bar in enumerate(barplot_b.patches):
        if i == 0:
            bar.set_color('white')

    plt.show()

def best_plotaspec_function_ever(
        mzml,
        spectrum_id,
        sequence,
        modifications,
        charge,
        offset=None,
        tolerance_in_ppm=5,
        only_top_n_peaks=None,
):
    ds = mzml[-14:-11]
    spec_peaks = pymzml.run.Reader(mzml)[spectrum_id].peaks("centroided")
    spec_peaks[:, 0] += spec_peaks[:, 0] * (1e-6)
    if only_top_n_peaks is not None:
        spec_peaks = spec_peaks[spec_peaks[:, 1].argsort()][-only_top_n_peaks:, :]
    # spec_peaks[:, 1] /= spec_peaks[:, 1].max()
    mod_comp = {}
    if modifications != "":
        mods = set(re.sub(r":\d+", "", modifications).split(";"))
        for m in mods:
            mod_comp[m] = um.name_to_mass(m)[0]
    theoretical_peaks = _fragment_peptide(
        sequence=sequence,
        modifications=modifications,
        mod_comp=mod_comp,
        max_charge=charge,
    )
    matched_annotation = []
    for peak in spec_peaks[:, 0]:
        matches = ""
        for fragment, mzs in (theoretical_peaks[0] | theoretical_peaks[1]).items():
            if any([abs(mz-peak) <= peak * tolerance_in_ppm * 1e-6 for mz in mzs]):
                matches += fragment + ";"
        matched_annotation.append(matches.rstrip(";"))
    colors = np.array([m != "" for m in matched_annotation])

    ion_list = get_ions_list(matched_annotation)
    plot_fragments(sequence, ion_list)

    fig, ax = plt.subplots()

    chart = ax.bar(
        x=spec_peaks[colors, 0],
        height=spec_peaks[colors, 1],
        width=5,
        color="red",
    )
    for bar, name in zip(chart, list(itertools.compress(matched_annotation, colors))):
        height = bar.get_height()
        ax.annotate(name, xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0,4), textcoords="offset points", ha="center", va="bottom", rotation=90, fontsize=6)
    unmatched_chart = ax.bar(
        x=spec_peaks[~colors, 0],
        height=spec_peaks[~colors, 1],
        width=5,
        color="black",
    )
    # # plt.yscale("log")
    plt.title(
        f"{ds}: Spectrum {spectrum_id} | {sum(colors)} matched peaks\n {sequence}#{modifications}"
    )
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.savefig(
        f"./{ds}_{spectrum_id}_{sequence}_{modifications}.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()


top_target_cols = [c for c in df.columns if "top_target_" in c]

plt_df = df.groupby(["dataset", "spectrum_id"]).get_group(("E13", 27727))
plt_df = plt_df[plt_df[top_target_cols].any(axis=1)]
for _, row in plt_df.iterrows():
    if _ == 0:
        continue
    print(row[top_target_cols][row[top_target_cols]].index)
    row = row.fillna("")
    print(row[top_target_cols])
    best_plotaspec_function_ever(
        mzml="./" + row["raw_data_location"],
        spectrum_id=row["spectrum_id"],
        sequence=row["sequence"],
        modifications=row["modifications"],
        charge=row["charge"],
        tolerance_in_ppm=200,
        only_top_n_peaks=100,
    )
