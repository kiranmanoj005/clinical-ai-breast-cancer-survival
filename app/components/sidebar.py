import streamlit as st
from sklearn.datasets import load_breast_cancer
import numpy as np


def render_sidebar() -> dict:
    """Render the patient input sidebar with demographic + clinical feature inputs."""
    data = load_breast_cancer()
    feature_names = data.feature_names
    X = data.data

    st.sidebar.markdown("## 👤 Patient Demographics")
    st.sidebar.markdown("---")

    # ── Demographic inputs
    st.sidebar.markdown("#### 🧬 Patient Information")
    age = st.sidebar.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=52,
        step=1,
        help="Patient's age in years"
    )

    cancer_stage = st.sidebar.selectbox(
        "Cancer Stage",
        options=["Stage I", "Stage II", "Stage III", "Stage IV", "Unknown"],
        index=1,
        help="TNM clinical staging of the tumour"
    )

    st.sidebar.markdown("#### 🧪 Receptor Status")
    er_status = st.sidebar.radio(
        "ER Status (Estrogen Receptor)",
        options=["Positive", "Negative", "Unknown"],
        index=0,
        horizontal=True,
        help="Oestrogen receptor expression status"
    )

    pr_status = st.sidebar.radio(
        "PR Status (Progesterone Receptor)",
        options=["Positive", "Negative", "Unknown"],
        index=0,
        horizontal=True,
        help="Progesterone receptor expression status"
    )

    her2_status = st.sidebar.radio(
        "HER2 Status",
        options=["Positive", "Negative", "Unknown"],
        index=1,
        horizontal=True,
        help="Human Epidermal Growth Factor Receptor 2 status"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🔬 Cell Nucleus Measurements")
    st.sidebar.caption("Adjust sliders to match patient's biopsy measurements.")

    patient = {
        "__age__": age,
        "__stage__": cancer_stage,
        "__er_status__": er_status,
        "__pr_status__": pr_status,
        "__her2_status__": her2_status,
    }

    for i, name in enumerate(feature_names):
        min_val = float(np.percentile(X[:, i], 1))
        max_val = float(np.percentile(X[:, i], 99))
        mean_val = float(np.mean(X[:, i]))
        patient[name] = st.sidebar.slider(
            label=name.replace(" ", "_"),
            min_value=round(min_val, 4),
            max_value=round(max_val, 4),
            value=round(mean_val, 4),
            format="%.4f",
        )
        if (i + 1) % 10 == 0 and i < len(feature_names) - 1:
            st.sidebar.markdown("---")

    # ── Demographic summary badge
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📋 Patient Summary")
    stage_colours = {
        "Stage I": "🟢", "Stage II": "🟡",
        "Stage III": "🟠", "Stage IV": "🔴", "Unknown": "⚪"
    }
    st.sidebar.markdown(
        f"- **Age:** {age} yrs\n"
        f"- **Stage:** {stage_colours.get(cancer_stage, '')} {cancer_stage}\n"
        f"- **ER:** {er_status} | **PR:** {pr_status} | **HER2:** {her2_status}"
    )

    return patient
