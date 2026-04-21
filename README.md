# 📊 AutoGluon MLOps Pipeline - Telecom Customer Churn

[![CI/CD](https://github.com/yourusername/autogluon-mlops/actions/workflows/mlops.yml/badge.svg)](https://github.com/yourusername/autogluon-mlops/actions/workflows/mlops.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-grade MLOps pipeline** with automated drift detection, conditional retraining, and model versioning using AutoGluon, Evidently AI, MLflow, FastAPI, and Streamlit.

---

## 🎯 Project Overview

This project implements a complete MLOps lifecycle for **Telecom Customer Churn Prediction**:

```mermaid
graph LR
    A[Data Ingestion] --> B[Train v1]
    B --> C[Drift Detection]
    C --> D{Drift > 0.05?}
    D -->|Yes| E[Retrain v2]
    D -->|No| F[Model Stable]
    E --> G[MLflow Registry]
    F --> G
    G --> H[FastAPI Serving]
    H --> I[Streamlit Dashboard]