"""
data_loader.py

Loads and validates the two real data sources:
  - Crystal violet viability data  -> data/crystal_violet_raw.csv
  - AutoDock Vina docking results  -> data/docking_results_raw.csv

WHEN YOUR REAL EXCEL FILES ARE READY:
  1. Export each sheet as CSV (or point pd.read_excel directly at your .xlsx)
  2. Rename/save them to the paths above (or pass your own paths to load_*())
  3. Update EXPECTED_CV_COLUMNS / EXPECTED_DOCKING_COLUMNS below to match your
     actual column names exactly, and adjust the rename map if your headers differ.

This module intentionally fails loudly (raises) rather than silently coercing,
so a schema mismatch is caught before it corrupts downstream IC50 fits.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Update these if your real column headers differ
EXPECTED_CV_COLUMNS = {
    "Compound", "CellLine", "DrugConc_uM", "Replicate", "OD_570nm", "Viability_percent"
}
EXPECTED_DOCKING_COLUMNS = {
    "Compound", "Target", "PDB_ID", "DockingScore_kcal_mol",
    "MolecularWeight", "LogP", "HBondDonors", "HBondAcceptors", "TPSA"
}

# If your real Excel headers use different names, map them here:
# e.g. {"Cell Line": "CellLine", "Conc (uM)": "DrugConc_uM"}
CV_RENAME_MAP = {}
DOCKING_RENAME_MAP = {}


def load_crystal_violet(path: str | Path = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_DIR / "crystal_violet_raw.csv"
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if CV_RENAME_MAP:
        df = df.rename(columns=CV_RENAME_MAP)

    missing = EXPECTED_CV_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Crystal violet file is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Update EXPECTED_CV_COLUMNS / CV_RENAME_MAP in data_loader.py to match your file."
        )

    df["CellLine"] = df["CellLine"].astype(str).str.strip()
    df["Compound"] = df["Compound"].astype(str).str.strip()
    df["DrugConc_uM"] = pd.to_numeric(df["DrugConc_uM"], errors="coerce")
    df["Viability_percent"] = pd.to_numeric(df["Viability_percent"], errors="coerce")
    df = df.dropna(subset=["DrugConc_uM", "Viability_percent"])
    return df


def load_docking_results(path: str | Path = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_DIR / "docking_results_raw.csv"
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if DOCKING_RENAME_MAP:
        df = df.rename(columns=DOCKING_RENAME_MAP)

    missing = EXPECTED_DOCKING_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Docking results file is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Update EXPECTED_DOCKING_COLUMNS / DOCKING_RENAME_MAP in data_loader.py to match your file."
        )

    df["Compound"] = df["Compound"].astype(str).str.strip()
    numeric_cols = ["DockingScore_kcal_mol", "MolecularWeight", "LogP",
                     "HBondDonors", "HBondAcceptors", "TPSA"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


if __name__ == "__main__":
    cv = load_crystal_violet()
    docking = load_docking_results()
    print("Crystal violet:", cv.shape, list(cv.columns))
    print("Docking results:", docking.shape, list(docking.columns))
