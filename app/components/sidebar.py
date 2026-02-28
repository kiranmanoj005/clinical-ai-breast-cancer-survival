import streamlit as st
from sklearn.datasets import load_breast_cancer
import numpy as np


def render_sidebar() -> dict:
    """Render the patient input sidebar and return feature dict."""
    data = load_breast_cancer()
    feature_names = data.feature_names
    X = data.data

    st.sidebar.markdown("## 👤 Patient Clinical Features")
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🔬 Cell Nucleus Measurements")
    st.sidebar.caption("Adjust sliders to match patient’s biopsy measurements.")

    patient = {}
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

    return patient
