"""
screening_model.py

THE BRIDGE BETWEEN WET-LAB AND VIRTUAL SCREENING.

This fixes the original architectural flaw: you cannot feed docking
descriptors into a model trained on assay-readout features. Instead:

  1. Aggregate docking results per compound (across CDC25A/B/C targets)
     into a fixed feature vector: best docking score, per-target scores,
     molecular descriptors (MW, LogP, HBD, HBA, TPSA).
  2. Join that feature vector to the wet-lab-derived IC50 (from
     wetlab_ic50.py) ONLY for compounds that have both — this is your
     small "anchor set" (e.g. NSC95397 + whatever else you've assayed).
  3. Train a model (Random Forest / Gradient Boosting) on the anchor set:
     docking/molecular features -> log(IC50).
  4. Apply that trained model to compounds that ONLY have docking data
     (your virtual library) to get predicted IC50 — this is the actual
     virtual screening output.

HONESTY NOTE FOR YOUR THESIS: with a handful of anchor compounds, this
model has very limited statistical power. Report it as a proof-of-concept
prioritization ranking, not a validated predictive model, unless your
anchor set grows past ~15-20 compounds with measured IC50. Cross-validated
R2 on n<10 should not be presented as generalizable accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error


def build_compound_features(docking_df: pd.DataFrame) -> pd.DataFrame:
    """One row per compound: best/per-target docking scores + molecular descriptors."""
    pivot_scores = docking_df.pivot_table(
        index="Compound", columns="Target", values="DockingScore_kcal_mol", aggfunc="min"
    )
    pivot_scores.columns = [f"DockingScore_{t}" for t in pivot_scores.columns]
    pivot_scores["DockingScore_best"] = pivot_scores.min(axis=1)

    descriptors = docking_df.groupby("Compound").agg({
        "MolecularWeight": "first",
        "LogP": "first",
        "HBondDonors": "first",
        "HBondAcceptors": "first",
        "TPSA": "first",
    })

    features = pivot_scores.join(descriptors).reset_index()
    return features


def build_anchor_dataset(compound_features: pd.DataFrame, wetlab_ic50_table: pd.DataFrame) -> pd.DataFrame:
    """Merge features with measured IC50, averaging across cell lines per compound."""
    ic50_avg = (
        wetlab_ic50_table[wetlab_ic50_table["fit_success"]]
        .groupby("Compound")["IC50_uM"].mean().reset_index()
    )
    anchor = compound_features.merge(ic50_avg, on="Compound", how="inner")
    anchor["log_IC50"] = np.log10(anchor["IC50_uM"])
    return anchor


def train_screening_model(anchor_df: pd.DataFrame, feature_cols: list, model_type="rf"):
    X = anchor_df[feature_cols]
    y = anchor_df["log_IC50"]

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=4)
    else:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=2)

    # With very few anchor compounds, use leave-one-out CV instead of a train/test split
    n = len(anchor_df)
    if n >= 4:
        loo = LeaveOneOut()
        preds = cross_val_predict(model, X, y, cv=loo)
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        print(f"[Anchor set n={n}] Leave-one-out CV: R2={r2:.3f}, MAE(log10 IC50)={mae:.3f}")
        if n < 10:
            print("WARNING: n<10 anchor compounds - treat this as a ranking heuristic, "
                  "not a validated quantitative model, in your writeup.")
    else:
        print(f"WARNING: only {n} anchor compounds available - too few for any CV. "
              "Model will be fit but should NOT be reported with a performance metric.")

    model.fit(X, y)
    return model


def predict_virtual_library(model, compound_features: pd.DataFrame, feature_cols: list,
                             known_compounds: list) -> pd.DataFrame:
    """Predict IC50 for compounds NOT in the anchor (measured) set."""
    virtual_only = compound_features[~compound_features["Compound"].isin(known_compounds)].copy()
    virtual_only["Predicted_log_IC50"] = model.predict(virtual_only[feature_cols])
    virtual_only["Predicted_IC50_uM"] = 10 ** virtual_only["Predicted_log_IC50"]
    return virtual_only.sort_values("Predicted_IC50_uM")


if __name__ == "__main__":
    from data_loader import load_docking_results
    from wetlab_ic50 import compute_ic50_table
    from data_loader import load_crystal_violet

    docking = load_docking_results()
    cv = load_crystal_violet()
    ic50_table = compute_ic50_table(cv)

    features = build_compound_features(docking)
    anchor = build_anchor_dataset(features, ic50_table)

    feature_cols = [c for c in features.columns if c != "Compound"]
    print(f"\nAnchor compounds (measured IC50 + docking data): {list(anchor['Compound'])}")

    model = train_screening_model(anchor, feature_cols, model_type="rf")
    ranked = predict_virtual_library(model, features, feature_cols, known_compounds=list(anchor["Compound"]))

    print(f"\nTop 10 predicted-most-potent virtual compounds:")
    print(ranked[["Compound", "DockingScore_best", "Predicted_IC50_uM"]].head(10).to_string(index=False))

    ranked.to_csv("/home/claude/oncotwin-gbm/data/virtual_screening_ranked.csv", index=False)
    anchor.to_csv("/home/claude/oncotwin-gbm/data/anchor_dataset.csv", index=False)
