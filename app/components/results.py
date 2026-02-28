import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def render_results(result: dict):
    """Render the prediction results with visual indicators."""
    prob = result["survival_probability"]
    risk = result["risk_level"]
    pred = result["prediction"]

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    # ── Top KPI cards
    col1, col2, col3 = st.columns(3)
    with col1:
        label = "✅ Benign" if pred == 1 else "❌ Malignant"
        st.metric("Diagnosis", label)
    with col2:
        st.metric("Survival Probability", f"{prob:.1f}%")
    with col3:
        colour = {"Low Risk": "🟢", "Moderate Risk": "🟡", "High Risk": "🔴"}
        st.metric("Risk Level", f"{colour[risk]} {risk}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # ── Gauge chart
    with col_left:
        st.markdown("#### Survival Probability Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob,
            number={"suffix": "%", "font": {"size": 36}},
            delta={"reference": 50, "increasing": {"color": "#2ecc71"}, "decreasing": {"color": "#e74c3c"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#667eea"},
                "steps": [
                    {"range": [0, 50], "color": "#ffeaea"},
                    {"range": [50, 75], "color": "#fff8e1"},
                    {"range": [75, 100], "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": "#764ba2", "width": 4},
                    "thickness": 0.75,
                    "value": prob
                }
            },
            title={"text": "5-Year Survival Probability", "font": {"size": 16}}
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Probability breakdown bar
    with col_right:
        st.markdown("#### Class Probability Breakdown")
        probs = result["probabilities"]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Benign", x=["Benign"], y=[probs[1] * 100],
            marker_color="#2ecc71", text=[f"{probs[1]*100:.1f}%"], textposition="outside"
        ))
        fig2.add_trace(go.Bar(
            name="Malignant", x=["Malignant"], y=[probs[0] * 100],
            marker_color="#e74c3c", text=[f"{probs[0]*100:.1f}%"], textposition="outside"
        ))
        fig2.update_layout(
            yaxis_range=[0, 110], yaxis_title="Probability (%)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Clinical interpretation
    st.markdown("#### 🏥 Clinical Interpretation")
    if risk == "Low Risk":
        st.success("🟢 **Low Risk** — The model predicts a favourable prognosis. The tumour features are consistent with a benign profile.")
    elif risk == "Moderate Risk":
        st.warning("🟡 **Moderate Risk** — The model detects ambiguous features. Further clinical investigation is recommended.")
    else:
        st.error("🔴 **High Risk** — The model detects features consistent with malignancy. Immediate clinical follow-up is strongly advised.")

    st.info("🔍 **Note:** This prediction is based on cell nucleus measurements from a biopsy sample. It is an AI-assisted tool and not a clinical diagnosis.")
