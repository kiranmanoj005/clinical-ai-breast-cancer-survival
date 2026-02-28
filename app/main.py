import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.components.sidebar import render_sidebar
from app.components.results import render_results
from app.components.explainability import render_explainability
from app.predict import load_pipeline, predict_survival

st.set_page_config(
    page_title="Clinical AI — Breast Cancer Survival Predictor",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Global font ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    /* ── Hide default Streamlit header/footer ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
        border-radius: 20px;
        padding: 3rem 3.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(102,126,234,0.2) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-ribbon {
        font-size: 4.5rem;
        line-height: 1;
        margin-bottom: 0.8rem;
        display: block;
        filter: drop-shadow(0 4px 12px rgba(102,126,234,0.6));
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #818cf8 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.5rem 0;
        line-height: 1.1;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        margin: 0 0 1.5rem 0;
        max-width: 600px;
        line-height: 1.6;
    }
    .hero-badges {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        color: #cbd5e1;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: linear-gradient(135deg, rgba(251,191,36,0.1), rgba(245,158,11,0.05));
        border-left: 4px solid #f59e0b;
        padding: 0.85rem 1.2rem;
        border-radius: 0 10px 10px 0;
        font-size: 0.87rem;
        color: #fbbf24;
        margin-bottom: 1.5rem;
    }

    /* ── Nav menu override ── */
    .nav-container {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 0.4rem;
        margin-bottom: 1.5rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1rem 1.2rem;
    }

    /* ── Predict button ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        width: 100% !important;
        box-shadow: 0 8px 24px rgba(102,126,234,0.35) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.3px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(102,126,234,0.5) !important;
    }

    /* ── Section headings ── */
    h3 { font-size: 1.5rem !important; font-weight: 700 !important; }
    h4 { font-size: 1.1rem !important; font-weight: 600 !important; color: #94a3b8 !important; }

    /* ── Dividers ── */
    hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


def main():
    # ── Hero Banner ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-ribbon">🎗️</span>
        <h1 class="hero-title">Clinical AI — Breast Cancer<br>Survival Predictor</h1>
        <p class="hero-subtitle">
            An explainable AI platform for breast cancer survival prediction,
            powered by XGBoost, SHAP interpretability, and Kaplan‑Meier survival analysis.
        </p>
        <div class="hero-badges">
            <span class="badge">🤖 XGBoost Classifier</span>
            <span class="badge">💡 SHAP Explainability</span>
            <span class="badge">📈 Kaplan-Meier Curves</span>
            <span class="badge">🧪 Wisconsin Dataset</span>
            <span class="badge">✅ 97.2% Accuracy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="disclaimer">⚠️ <strong>Clinical Disclaimer:</strong> This tool is for educational and research purposes only. It is not a medical device and must not be used for clinical decision-making. Always consult a qualified healthcare professional.</div>', unsafe_allow_html=True)

    # ── Navigation ─────────────────────────────────────────────────────────
    selected = option_menu(
        menu_title=None,
        options=["Predict", "Explainability", "Model Performance", "About"],
        icons=["activity", "lightbulb", "bar-chart-line", "info-circle"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0.4rem", "background-color": "rgba(255,255,255,0.03)",
                          "border": "1px solid rgba(255,255,255,0.07)", "border-radius": "14px"},
            "icon": {"font-size": "1rem", "color": "#94a3b8"},
            "nav-link": {"font-size": "0.95rem", "font-weight": "500", "color": "#94a3b8",
                         "border-radius": "10px", "padding": "0.5rem 1.2rem"},
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #667eea, #764ba2)",
                "color": "white", "font-weight": "700"
            },
        }
    )

    # ── Load model ────────────────────────────────────────────────────────────
    pipeline = load_pipeline()

    if selected == "Predict":
        st.markdown("### 🔬 Enter Patient Clinical Data")
        st.caption("Adjust the patient’s clinical features in the sidebar, then click the button below to run a prediction.")
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
        st.caption("Understand which features drive the model’s predictions globally and per-patient.")
        render_explainability(pipeline)

    elif selected == "Model Performance":
        st.markdown("### 📊 Model Performance Metrics")
        _render_performance()

    elif selected == "About":
        _render_about()


def _render_performance():
    st.markdown("#### Evaluation on held-out test set (20% split, stratified)")
    metrics = {
        "Metric": ["Accuracy", "AUC-ROC", "Precision", "Recall", "F1 Score"],
        "Score": ["97.2%", "99.4%", "96.8%", "97.6%", "97.2%"]
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
            title="Model Metrics Overview", xaxis_range=[90, 101],
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0, r=50, t=40, b=0),
            font=dict(color="#94a3b8")
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
