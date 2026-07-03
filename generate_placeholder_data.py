"""
generate_placeholder_data.py

IMPORTANT: This script creates SYNTHETIC PLACEHOLDER DATA ONLY.
None of the values here come from real experiments, real databases (GDSC, ChEMBL,
PubChem), or real literature. They exist purely so the OncoTwin-GBM pipeline can
be built and tested end-to-end BEFORE real data is loaded in.

Replace the two output files with your real files when ready:
  - crystal_violet_raw.csv   -> your real crystal violet Excel export (as CSV)
  - docking_results_raw.csv  -> your real AutoDock Vina docking results (as CSV)

The schemas below are a REASONABLE GUESS based on standard crystal violet /
docking result formats. Once you share your real column names, the loader in
src/data_loader.py should be updated to match exactly.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. SYNTHETIC crystal violet viability data
#    Standard layout: per compound, per cell line, per concentration, per replicate
# ---------------------------------------------------------------------------
compounds = ["NSC95397", "CompoundA_synthetic", "CompoundB_synthetic", "TMZ_synthetic"]
cell_lines = ["U87", "U251"]
concentrations = [0, 0.1, 1, 5, 10, 25, 50, 100]  # uM
replicates = [1, 2, 3]

rows = []
for compound in compounds:
    # give each compound a synthetic "true" IC50 so the dose-response curve is realistic
    true_ic50 = rng.uniform(5, 60)
    hill_slope = rng.uniform(0.8, 1.5)
    for cell_line in cell_lines:
        cell_shift = rng.uniform(0.8, 1.2)  # cell-line-specific sensitivity shift
        for conc in concentrations:
            for rep in replicates:
                if conc == 0:
                    viability = 100 + rng.normal(0, 2)
                else:
                    # simple 4-parameter logistic (top=100, bottom=0)
                    ec = true_ic50 * cell_shift
                    viability = 100 / (1 + (conc / ec) ** hill_slope) + rng.normal(0, 4)
                viability = float(np.clip(viability, 0, 105))
                rows.append({
                    "Compound": compound,
                    "CellLine": cell_line,
                    "DrugConc_uM": conc,
                    "Replicate": rep,
                    "OD_570nm": round(0.05 + viability / 100 * rng.uniform(0.8, 1.0), 4),
                    "Viability_percent": round(viability, 2),
                })

crystal_violet_df = pd.DataFrame(rows)
crystal_violet_df.to_csv("/home/claude/oncotwin-gbm/data/crystal_violet_raw.csv", index=False)

# ---------------------------------------------------------------------------
# 2. SYNTHETIC docking results (AutoDock Vina style)
#    Includes the 3 compounds above PLUS extra "virtual library" compounds
#    that have NO wet-lab data (this is the point of virtual screening)
# ---------------------------------------------------------------------------
virtual_library = compounds + [f"VirtualCpd_{i:03d}" for i in range(1, 21)]

docking_rows = []
for compound in virtual_library:
    for target in ["CDC25A", "CDC25B", "CDC25C"]:
        docking_rows.append({
            "Compound": compound,
            "Target": target,
            "PDB_ID": {"CDC25A": "1C25", "CDC25B": "1QB0", "CDC25C": "3OP3"}[target],
            "DockingScore_kcal_mol": round(rng.uniform(-8.5, -3.5), 3),
            "MolecularWeight": round(rng.uniform(180, 550), 2),
            "LogP": round(rng.uniform(-1, 5), 2),
            "HBondDonors": int(rng.integers(0, 5)),
            "HBondAcceptors": int(rng.integers(1, 8)),
            "TPSA": round(rng.uniform(20, 140), 2),
        })

docking_df = pd.DataFrame(docking_rows)
docking_df.to_csv("/home/claude/oncotwin-gbm/data/docking_results_raw.csv", index=False)

print("Synthetic placeholder files written:")
print(" - crystal_violet_raw.csv :", crystal_violet_df.shape)
print(" - docking_results_raw.csv:", docking_df.shape)
print("\nThese are FAKE numbers for pipeline testing only. Replace with real exports.")
