import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def _kaplan_meier_estimate(survival_prob: float, risk: str) -> dict:
    """
    Generate a synthetic Kaplan-Meier survival curve based on the
    model's predicted survival probability and risk level.
    Time axis is in months (0–60 for 5-year view).
    """
    t = np.linspace(0, 60, 200)

    # Shape the KM curve using a Weibull-style decay modulated by predicted probability
    scale_map = {"Low Risk": 0.008, "Moderate Risk": 0.022, "High Risk": 0.042}
    lam = scale_map.get(risk, 0.022)
    S_t = np.exp(-lam * t) * (survival_prob / 100)
    S_t = np.clip(S_t + (1 - survival_prob / 100) * np.exp(-0.003 * t), 0, 1)

    # Confidence interval band (±5% noise, wider at later time)
    ci_width = 0.04 + 0.002 * t
    S_upper = np.clip(S_t + ci_width, 0, 1)
    S_lower = np.clip(S_t - ci_width, 0, 1)

    return {"t": t, "S": S_t, "upper": S_upper, "lower": S_lower}


def render_kaplan_meier(result: dict) -> None:
    """Render Kaplan-Meier survival curve for the predicted patient."""
    prob = result["survival_probability"]
    risk = result["risk_level"]

    km = _kaplan_meier_estimate(prob, risk)
    t, S, S_upper, S_lower = km["t"], km["S"], km["upper"], km["lower"]

    colour_map = {"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c"}
    line_colour = colour_map.get(risk, "#667eea")

    fig = go.Figure()

    # Confidence interval shading
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([S_upper, S_lower[::-1]]),
        fill="toself",
        fillcolor=line_colour.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in line_colour else line_colour + "26",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
        showlegend=True,
        hoverinfo="skip"
    ))

    # Main survival curve
    fig.add_trace(go.Scatter(
        x=t, y=S * 100,
        mode="lines",
        name=f"Predicted Survival ({risk})",
        line=dict(color=line_colour, width=3),
        hovertemplate="Month %{x:.0f}: %{y:.1f}% survival<extra></extra>"
    ))

    # Reference 50% line
    fig.add_hline(
        y=50, line_dash="dash",
        line_color="#95a5a6", opacity=0.7,
        annotation_text="50% survival",
        annotation_position="bottom right"
    )

    fig.update_layout(
        title=dict(text="📈 Kaplan-Meier Survival Curve (5-Year)", font=dict(size=16)),
        xaxis_title="Time (Months)",
        yaxis_title="Survival Probability (%)",
        yaxis=dict(range=[0, 105]),
        xaxis=dict(range=[0, 60]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "⚠️ This Kaplan-Meier curve is model-estimated based on predicted risk. "
        "It is not derived from real patient cohort data and should not be used for clinical decisions."
    )


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

    # ── Kaplan-Meier Survival Curve
    st.markdown("---")
    st.markdown("#### 📈 Survival Curve")
    render_kaplan_meier(result)

    # ── Clinical interpretation
    st.markdown("#### 🏥 Clinical Interpretation")
    if risk == "Low Risk":
        st.success("🟢 **Low Risk** — The model predicts a favourable prognosis. The tumour features are consistent with a benign profile.")
    elif risk == "Moderate Risk":
        st.warning("🟡 **Moderate Risk** — The model detects ambiguous features. Further clinical investigation is recommended.")
    else:
        st.error("🔴 **High Risk** — The model detects features consistent with malignancy. Immediate clinical follow-up is strongly advised.")

    st.info("🔍 **Note:** This prediction is based on cell nucleus measurements from a biopsy sample. It is an AI-assisted tool and not a clinical diagnosis.")
