"""
app.py - OncoTwin-GBM Streamlit Dashboard

Run with:  streamlit run app.py   (from the app/ directory)

Three tabs:
  1. Wet-lab dose-response  - crystal violet curves + fitted IC50 per compound/cell line
  2. Virtual screening      - docking-based predicted IC50 ranking, with the
                               anchor-set limitation stated explicitly on screen
  3. Explainability         - SHAP summary for the screening model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_loader import load_crystal_violet, load_docking_results
from wetlab_ic50 import compute_ic50_table, four_param_logistic, fit_ic50
from screening_model import (
    build_compound_features, build_anchor_dataset, train_screening_model,
    predict_virtual_library,
)

st.set_page_config(page_title="OncoTwin-GBM", layout="wide")
st.title("OncoTwin-GBM: Glioblastoma Digital Twin")
st.caption("CDC25 phosphatase inhibitor discovery — wet-lab + virtual screening bridge")

with st.sidebar:
    st.header("Data source")
    st.write("Currently loaded from `data/crystal_violet_raw.csv` and `data/docking_results_raw.csv`.")
    st.info(
        "These are SYNTHETIC PLACEHOLDER values until real Excel exports are dropped in "
        "(see src/data_loader.py for the exact expected schema)."
    )

# --- Load & compute (cached so the dashboard stays responsive) ---
@st.cache_data
def load_all():
    cv = load_crystal_violet()
    docking = load_docking_results()
    ic50_table = compute_ic50_table(cv)
    features = build_compound_features(docking)
    anchor = build_anchor_dataset(features, ic50_table)
    return cv, docking, ic50_table, features, anchor

cv, docking, ic50_table, features, anchor = load_all()
feature_cols = [c for c in features.columns if c != "Compound"]

tab1, tab2, tab3 = st.tabs(["Wet-lab dose-response", "Virtual screening", "Explainability"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Crystal violet dose-response fitting")
    col1, col2 = st.columns([1, 2])
    with col1:
        compound_sel = st.selectbox("Compound", sorted(cv["Compound"].unique()))
        cellline_sel = st.selectbox("Cell line", sorted(cv["CellLine"].unique()))

    group = cv[(cv["Compound"] == compound_sel) & (cv["CellLine"] == cellline_sel)]
    fit = fit_ic50(group)

    with col1:
        st.metric("Fitted IC50 (uM)", f"{fit['IC50_uM']:.2f}" if fit["fit_success"] else "fit failed")
        st.metric("R2", f"{fit['R2']:.3f}" if fit["fit_success"] else "-")
        st.metric("n data points", fit["n_points"])

    with col2:
        fig, ax = plt.subplots()
        ax.scatter(group["DrugConc_uM"], group["Viability_percent"], alpha=0.6, label="observed")
        if fit["fit_success"]:
            x_smooth = np.logspace(-2, np.log10(group["DrugConc_uM"].replace(0, np.nan).max() * 1.2), 200)
            y_smooth = four_param_logistic(x_smooth, fit["Top"], fit["Bottom"], fit["IC50_uM"], fit["HillSlope"])
            ax.plot(x_smooth, y_smooth, color="crimson", label="4PL fit")
        ax.set_xscale("symlog")
        ax.set_xlabel("Concentration (uM)")
        ax.set_ylabel("Viability (%)")
        ax.set_title(f"{compound_sel} on {cellline_sel}")
        ax.legend()
        st.pyplot(fig)

    st.subheader("All fitted IC50 values")
    st.dataframe(ic50_table, use_container_width=True)

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Virtual screening: docking descriptors -> predicted IC50")
    n_anchor = len(anchor)
    if n_anchor < 10:
        st.warning(
            f"Only {n_anchor} anchor compounds have both measured IC50 and docking data. "
            "Treat the ranking below as a preliminary prioritization heuristic, "
            "not a validated predictive model."
        )

    model_type = st.radio("Model", ["rf", "gbm"], horizontal=True, format_func=lambda x: "Random Forest" if x == "rf" else "Gradient Boosting")
    model = train_screening_model(anchor, feature_cols, model_type=model_type)
    ranked = predict_virtual_library(model, features, feature_cols, known_compounds=list(anchor["Compound"]))

    st.write(f"**Anchor (measured) compounds:** {', '.join(anchor['Compound'])}")
    st.dataframe(
        ranked[["Compound", "DockingScore_best", "MolecularWeight", "LogP", "Predicted_IC50_uM"]],
        use_container_width=True,
    )

    st.subheader("Anchor set: docking score vs measured IC50")
    fig2, ax2 = plt.subplots()
    ax2.scatter(anchor["DockingScore_best"], anchor["IC50_uM"])
    for _, row in anchor.iterrows():
        ax2.annotate(row["Compound"], (row["DockingScore_best"], row["IC50_uM"]), fontsize=8)
    ax2.set_xlabel("Best docking score (kcal/mol)")
    ax2.set_ylabel("Measured IC50 (uM)")
    st.pyplot(fig2)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("SHAP feature importance (screening model)")
    st.caption("Interpret directionally only — anchor set is small.")
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(anchor[feature_cols])

    fig3, ax3 = plt.subplots()
    shap.summary_plot(shap_values, anchor[feature_cols], show=False)
    st.pyplot(fig3)
