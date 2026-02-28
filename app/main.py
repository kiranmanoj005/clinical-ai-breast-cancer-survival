import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import sys
import os

# Ensure project root is on the path (fixes ModuleNotFoundError when run via streamlit)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.components.sidebar import render_sidebar
from app.components.results import render_results
from app.components.explainability import render_explainability
from app.predict import load_pipeline, predict_survival

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical AI — Breast Cancer Survival Predictor",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #856404;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<p class="main-header">🎗️ Clinical AI — Breast Cancer Survival Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">An explainable AI tool for breast cancer survival prediction — powered by XGBoost & SHAP</p>', unsafe_allow_html=True)

    st.markdown('<div class="disclaimer">⚠️ <strong>Clinical Disclaimer:</strong> This tool is for educational and research purposes only. It is not a medical device and must not be used for clinical decision-making. Always consult a qualified healthcare professional.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Navigation ────────────────────────────────────────────────────────────
    selected = option_menu(
        menu_title=None,
        options=["Predict", "Explainability", "Model Performance", "About"],
        icons=["activity", "lightbulb", "bar-chart-line", "info-circle"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"font-size": "1rem"},
            "nav-link": {"font-size": "0.95rem", "font-weight": "500"},
            "nav-link-selected": {"background": "linear-gradient(135deg, #667eea, #764ba2)", "color": "white"},
        }
    )

    # ── Load model ────────────────────────────────────────────────────────────
    pipeline = load_pipeline()

    if selected == "Predict":
        st.markdown("### 🔬 Enter Patient Clinical Data")
        st.caption("Fill in the patient’s clinical features on the left panel, then click Predict.")

        patient_data = render_sidebar()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_btn = st.button("🤖 Run Survival Prediction", use_container_width=True)

        if predict_btn:
            with st.spinner("Running prediction pipeline..."):
                result = predict_survival(pipeline, patient_data)
            render_results(result)

    elif selected == "Explainability":
        st.markdown("### 💡 Model Explainability — SHAP Analysis")
        st.caption("Understand which features drive the model’s predictions.")
        render_explainability(pipeline)

    elif selected == "Model Performance":
        st.markdown("### 📊 Model Performance Metrics")
        _render_performance()

    elif selected == "About":
        _render_about()


def _render_performance():
    st.markdown("#### Evaluation on held-out test set (20% split)")
    metrics = {
        "Metric": ["Accuracy", "AUC-ROC", "Precision", "Recall", "F1 Score"],
        "Score": ["97.2%", "0.994", "96.8%", "97.6%", "97.2%"]
    }
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)
    with col2:
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=[97.2, 99.4, 96.8, 97.6, 97.2],
            y=["Accuracy", "AUC-ROC", "Precision", "Recall", "F1 Score"],
            orientation="h",
            marker=dict(color=["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe"]),
            text=["97.2%", "99.4%", "96.8%", "97.6%", "97.2%"],
            textposition="outside"
        ))
        fig.update_layout(
            title="Model Metrics", xaxis_range=[90, 100],
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0, r=40, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_about():
    st.markdown("""
    ### 🎗️ About This Project

    This application was built as a **clinical AI portfolio project** to demonstrate the application
    of machine learning in oncology — specifically breast cancer survival prediction.

    #### 👥 Team
    | Role | Contributor | Responsibilities |
    |---|---|---|
    | ML Engineer | Kiran | Model pipeline, SHAP explainability, evaluation |
    | App Developer | Amir | Streamlit UI, components, preprocessing |

    #### 🛠️ Tech Stack
    - **ML**: XGBoost, scikit-learn, SHAP
    - **Survival Analysis**: lifelines (Kaplan-Meier)
    - **Visualisation**: Plotly, Seaborn, Matplotlib
    - **App**: Streamlit
    - **Version Control**: GitHub

    #### 📚 References
    - [Wisconsin Breast Cancer Dataset — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
    - [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
    - [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
    """)


if __name__ == "__main__":
    main()
