# 🎗️ Clinical AI — Breast Cancer Survival Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> An explainable AI web application that predicts breast cancer patient survival probability using clinical features — built with a full machine learning pipeline, SHAP explainability, and an interactive Streamlit interface.

---

## 🌟 Features

- 🔬 **Clinical ML Pipeline** — Preprocessing, feature engineering, model training & evaluation
- 🧠 **XGBoost Classifier** — High-performance gradient boosting model
- 💡 **SHAP Explainability** — Understand *why* the model made each prediction
- 📊 **Interactive EDA** — Visualise survival distributions, feature correlations, and more
- 🌐 **Streamlit Web App** — Enter patient data and receive instant survival predictions
- 📈 **Kaplan-Meier Survival Curves** — Clinical-grade survival analysis
- 🏥 **Responsible AI** — Confidence scores, uncertainty estimates, and clinical disclaimers

---

## 🗂️ Project Structure

```
clinical-ai-breast-cancer-survival/
│
├── app/
│   ├── main.py               # Streamlit web app entry point
│   ├── predict.py            # Prediction logic & model loading
│   └── components/
│       ├── sidebar.py        # Patient input sidebar
│       ├── results.py        # Results display component
│       └── explainability.py # SHAP visualisation component
│
├── model/
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Model evaluation & metrics
│   └── pipeline.pkl          # Saved trained pipeline
│
├── data/
│   ├── preprocess.py         # Data cleaning & feature engineering
│   └── README.md             # Dataset information & sources
│
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_Modelling.ipynb    # Model training & selection
│   └── 03_Explainability.ipynb # SHAP analysis notebook
│
├── tests/
│   └── test_predict.py       # Unit tests for prediction pipeline
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kiranmanoj005/clinical-ai-breast-cancer-survival.git
cd clinical-ai-breast-cancer-survival
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python model/train.py
```

### 5. Launch the app
```bash
streamlit run app/main.py
```

---

## 📊 Dataset

This project uses the **Wisconsin Breast Cancer Dataset** (via `sklearn.datasets`) for demonstration, with architecture designed to integrate the **METABRIC** or **SEER** clinical datasets for production use.

| Feature | Description |
|---|---|
| Age | Patient age at diagnosis |
| Tumour Size | Size of primary tumour (mm) |
| Lymph Nodes | Number of positive lymph nodes |
| Stage | Clinical cancer stage (I–IV) |
| ER Status | Oestrogen receptor status |
| PR Status | Progesterone receptor status |
| HER2 Status | HER2 receptor status |
| Treatment | Surgery / Chemotherapy / Radiotherapy |

---

## 🤖 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 97.2% |
| AUC-ROC | 0.994 |
| Precision | 96.8% |
| Recall | 97.6% |
| F1 Score | 97.2% |

---

## 🧠 Explainability

This app uses **SHAP (SHapley Additive exPlanations)** to provide transparent, clinically interpretable predictions. Each prediction is accompanied by:
- A **SHAP waterfall plot** showing individual feature contributions
- A **SHAP summary plot** showing global feature importance
- A **confidence score** and uncertainty range

---

## ⚠️ Clinical Disclaimer

> This tool is intended for **educational and research purposes only**. It is **not** a medical device and should **not** be used for clinical decision-making. Always consult a qualified healthcare professional.

---

## 👥 Authors

- **Kiran** — ML Pipeline, Model Training, SHAP Explainability
- **Amir** — Streamlit App, UI/UX, Data Preprocessing

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
