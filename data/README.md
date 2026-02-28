# 📁 Data

## Source

This project uses the **Wisconsin Breast Cancer Diagnostic Dataset** for demonstration purposes, loaded directly via `sklearn.datasets.load_breast_cancer()`.

### For Production Use
To use real clinical data, integrate one of the following:
- **METABRIC** — Molecular Taxonomy of Breast Cancer International Consortium
- **SEER** — Surveillance, Epidemiology, and End Results Program (NIH)
- **TCGA-BRCA** — The Cancer Genome Atlas Breast Cancer dataset

## Features (Wisconsin Dataset)

| Feature Group | Features |
|---|---|
| Geometry | radius, texture, perimeter, area, smoothness |
| Structure | compactness, concavity, concave points, symmetry |
| Fractal | fractal dimension |

Each feature is computed for **mean**, **standard error**, and **worst** (largest) values → 30 features total.

## Ethics & Privacy

All data used is **publicly available** and **fully anonymised**. No patient-identifiable information is stored or processed.
