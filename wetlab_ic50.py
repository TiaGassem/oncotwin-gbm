"""
wetlab_ic50.py

Derives IC50 from crystal violet viability data using a 4-parameter logistic
(4PL) dose-response fit per (Compound, CellLine) pair.

WHY NOT MACHINE LEARNING HERE:
Crystal violet gives you a handful of concentration points per compound/cell
line (typically 6-8), replicated 2-3x. That is a classic dose-response curve-
fitting problem, not a machine-learning regression problem — there isn't
enough independent data per curve to train an ML model, and a 4PL fit is the
pharmacology-standard method reviewers will expect. ML comes in later, at the
virtual-screening stage, where the *features* are per-compound molecular/
docking descriptors and the *samples* are different compounds.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def four_param_logistic(conc, top, bottom, ic50, hill):
    with np.errstate(divide="ignore", invalid="ignore"):
        return bottom + (top - bottom) / (1 + (conc / ic50) ** hill)


def fit_ic50(df_group: pd.DataFrame) -> dict:
    """Fit a 4PL curve to one (Compound, CellLine) group. Returns dict of fit results."""
    conc = df_group["DrugConc_uM"].values.astype(float)
    viab = df_group["Viability_percent"].values.astype(float)

    # avoid log(0) issues in the fit by nudging 0 concentration slightly
    conc_fit = np.where(conc == 0, 1e-3, conc)

    p0 = [100, 0, np.median(conc_fit[conc_fit > 0]) if (conc_fit > 0).any() else 10, 1]
    bounds = ([0, 0, 1e-6, 0.1], [150, 100, 1e6, 10])

    try:
        popt, pcov = curve_fit(
            four_param_logistic, conc_fit, viab, p0=p0, bounds=bounds, maxfev=10000
        )
        top, bottom, ic50, hill = popt
        residuals = viab - four_param_logistic(conc_fit, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((viab - np.mean(viab)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            "IC50_uM": round(float(ic50), 4),
            "Top": round(float(top), 2),
            "Bottom": round(float(bottom), 2),
            "HillSlope": round(float(hill), 3),
            "R2": round(float(r2), 4),
            "n_points": len(conc),
            "fit_success": True,
        }
    except (RuntimeError, ValueError) as e:
        return {
            "IC50_uM": np.nan, "Top": np.nan, "Bottom": np.nan,
            "HillSlope": np.nan, "R2": np.nan, "n_points": len(conc),
            "fit_success": False, "error": str(e),
        }


def compute_ic50_table(crystal_violet_df: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (Compound, CellLine) with fitted IC50 and QC metrics."""
    results = []
    for (compound, cell_line), group in crystal_violet_df.groupby(["Compound", "CellLine"]):
        fit = fit_ic50(group)
        fit["Compound"] = compound
        fit["CellLine"] = cell_line
        results.append(fit)
    out = pd.DataFrame(results)
    cols = ["Compound", "CellLine", "IC50_uM", "R2", "HillSlope", "n_points", "fit_success"]
    return out[cols + [c for c in out.columns if c not in cols]]


if __name__ == "__main__":
    from data_loader import load_crystal_violet
    cv = load_crystal_violet()
    ic50_table = compute_ic50_table(cv)
    ic50_table.to_csv("/home/claude/oncotwin-gbm/data/wetlab_ic50_table.csv", index=False)
    print(ic50_table)
    print("\nFlag any R2 < 0.8 or fit_success=False for manual review before using in the model.")
