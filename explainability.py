"""
explainability.py

SHAP explainability for the virtual screening model (docking/molecular
descriptors -> predicted IC50). Explaining the wet-lab dose-response fits
doesn't need SHAP (a 4PL curve is already fully interpretable - that's the
point of using it instead of a black-box model there).

NOTE: with only a handful of anchor compounds, SHAP values here describe
what the model learned from very little data. Present them as "directional
feature importance in this preliminary model" rather than "validated
mechanistic insight."
"""

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def explain_model(model, X, feature_cols, output_path="/home/claude/oncotwin-gbm/data/shap_summary.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[feature_cols])

    plt.figure()
    shap.summary_plot(shap_values, X[feature_cols], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return shap_values, output_path


if __name__ == "__main__":
    from data_loader import load_docking_results, load_crystal_violet
    from wetlab_ic50 import compute_ic50_table
    from screening_model import build_compound_features, build_anchor_dataset, train_screening_model

    docking = load_docking_results()
    cv = load_crystal_violet()
    ic50_table = compute_ic50_table(cv)
    features = build_compound_features(docking)
    anchor = build_anchor_dataset(features, ic50_table)
    feature_cols = [c for c in features.columns if c != "Compound"]

    model = train_screening_model(anchor, feature_cols, model_type="rf")
    shap_values, path = explain_model(model, anchor, feature_cols)
    print("SHAP summary plot saved to:", path)
