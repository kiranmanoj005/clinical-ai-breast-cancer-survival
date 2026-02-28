import streamlit as st
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


def render_explainability(pipeline):
    """Render SHAP explainability visualisations."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    st.markdown("#### 🌍 Global Feature Importance (SHAP Summary)")
    st.caption("SHAP values show each feature’s average contribution to the model output across all predictions.")

    with st.spinner("Computing SHAP values (this may take a moment)..."):
        model = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values, X_transformed,
            feature_names=data.feature_names,
            show=False, plot_size=None
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("#### 💡 Top 10 Features by Mean |SHAP|")

    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature": data.feature_names,
        "Mean |SHAP|": mean_shap
    }).sort_values("Mean |SHAP|", ascending=False).head(10)

    import plotly.express as px
    fig2 = px.bar(
        shap_df, x="Mean |SHAP|", y="Feature",
        orientation="h",
        color="Mean |SHAP|",
        color_continuous_scale="Viridis",
        title="Top 10 Most Influential Features"
    )
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    #### 📖 How to Read SHAP Values
    - **Positive SHAP value** → pushes prediction towards *malignant*
    - **Negative SHAP value** → pushes prediction towards *benign*
    - **Feature colour** → red = high feature value, blue = low feature value
    - **Width of distribution** → how often that feature influences predictions
    """)
