<div align="center">

# 🎗️ Breast Cancer Survival Predictor
### *Clinical AI — Explainable Machine Learning in Oncology*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange?style=for-the-badge)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=for-the-badge)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> A fully interactive web application that predicts breast cancer survival probability from clinical biopsy features,
> with explainable AI powered by SHAP and Kaplan‑Meier survival analysis.

</div>

---

## 📌 Overview

This project was built out of a shared interest in how AI can support clinical decision-making in oncology. We wanted to go beyond a standard classification notebook and build something that felt like a real clinical tool — one that doesn’t just give a prediction, but *explains* it in a way that would be meaningful to a clinician.

The app takes biopsy measurements as input, runs them through a trained XGBoost pipeline, and returns a survival probability alongside SHAP-based explanations and a Kaplan‑Meier survival curve. Everything is wrapped in a clean Streamlit interface designed to feel approachable and trustworthy.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **XGBoost Classifier** | Gradient boosted tree model achieving 97.2% accuracy on held-out test data |
| 💡 **SHAP Explainability** | Per-prediction feature attribution — see exactly what drove each outcome |
| 📈 **Kaplan-Meier Curves** | Risk-stratified survival curves visualised interactively with 95% CI bands |
| 🧪 **IQR Outlier Detection** | Automated outlier flagging with visual heatmap output |
| 📊 **Full EDA Notebook** | Class balance, KDE distributions, correlation heatmap, ANOVA feature ranking |
| 🏥 **Clinical Disclaimers** | Responsible AI framing throughout — not a diagnostic tool |
| 🧪 **Unit Tested** | Pytest coverage on prediction pipeline outputs and model accuracy thresholds |

---

## 🖥️ App Preview

The app has four sections accessible from the top navigation bar:

- **Predict** — Adjust patient biopsy features in the sidebar and run a live prediction
- **Explainability** — Global SHAP summary plot and top-10 feature importance bar chart
- **Model Performance** — Accuracy, AUC-ROC, Precision, Recall and F1 with visual bar chart
- **About** — Project background, tech stack and references

---

## 🗂️ Project Structure

```
clinical-ai-breast-cancer-survival/
├── app/
│   ├── main.py                  # Streamlit entry point & layout
│   ├── predict.py               # Pipeline loading & inference logic
│   └── components/
│       ├── sidebar.py           # Patient demographic & biopsy inputs
│       ├── results.py           # Gauge, KM curve & clinical interpretation
│       └── explainability.py    # SHAP summary & feature importance plots
├── model/
│   ├── train.py                 # Model training & pipeline serialisation
│   └── evaluate.py              # Confusion matrix, ROC & PR curves
├── data/
│   └── preprocess.py            # Outlier detection & distribution plots
├── notebooks/
│   └── 01_EDA.ipynb             # Full exploratory data analysis
├── tests/
│   └── test_predict.py          # Unit tests for prediction pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

> Requires Python 3.10+ and [Homebrew](https://brew.sh) on macOS (for XGBoost’s OpenMP dependency).

```bash
# macOS only — install OpenMP if not already present
brew install libomp
```

```bash
# 1. Clone
git clone https://github.com/kiranmanoj005/clinical-ai-breast-cancer-survival.git
cd clinical-ai-breast-cancer-survival

# 2. Virtual environment
python3 -m venv venv && source venv/bin/activate
# Windows: venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Train the model (first time only, ~10 seconds)
python model/train.py

# 5. Launch
streamlit run app/main.py
```

Open **http://localhost:8501** in your browser.

---

## 🧰 Tech Stack

- **ML & Data** — scikit-learn, XGBoost, pandas, NumPy, SciPy
- **Explainability** — SHAP (TreeExplainer)
- **Survival Analysis** — lifelines (Kaplan-Meier)
- **Visualisation** — Plotly, Seaborn, Matplotlib
- **App** — Streamlit, streamlit-option-menu
- **Testing** — pytest

---

## 🤖 Model

The classifier is an **XGBoost gradient boosted tree** trained on the [Wisconsin Breast Cancer Diagnostic Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) (569 samples, 30 features). All features describe cell nucleus properties computed from biopsy images (radius, texture, perimeter, area, smoothness, etc.).

| Metric | Score |
|---|---|
| Accuracy | 97.2% |
| AUC-ROC | 0.994 |
| Precision | 96.8% |
| Recall | 97.6% |
| F1 Score | 97.2% |

The full pipeline is: median imputation → standard scaling → XGBoost (300 estimators, lr=0.05, max_depth=4).

---

## ⚠️ Disclaimer

This project is intended **solely for educational and research purposes**. It is not a certified medical device, has not undergone clinical validation, and must not be used to inform real patient care. Always consult a qualified clinician.

---

## 👥 Authors

Built by **Kiran** and **Amir** as a clinical AI portfolio project.

- **Kiran** — ML pipeline, model training, SHAP explainability, evaluation
- **Amir** — Streamlit UI, patient input components, EDA notebook, survival curves

---

## 📚 References

- Wolberg, W. et al. *Wisconsin Breast Cancer Dataset*, UCI ML Repository (1995)
- Lundberg, S. & Lee, S-I. *A Unified Approach to Interpreting Model Predictions*, NeurIPS (2017)
- Chen, T. & Guestrin, C. *XGBoost: A Scalable Tree Boosting System*, KDD (2016)
- Davidson-Pilon, C. *lifelines: survival analysis in Python* (2019)
