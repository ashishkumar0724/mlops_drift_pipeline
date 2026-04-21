"""
MLOps Dashboard: Streamlit (Enhanced)
Shows metrics over time, drift report with custom threshold, and live prediction demo
"""

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(page_title="AutoGluon MLOps Dashboard", layout="wide", page_icon="📊")
st.title("📊 AutoGluon MLOps Dashboard")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ Pipeline Info")
    st.write("**Dataset:** Telecom Customer Churn")
    st.write("**Target:** Churn (Binary)")
    st.write("**Models:** RandomForest, ExtraTrees")
    st.write("**Drift Threshold:** 0.05 (Custom KS Test)")

    # Load metadata if exists
    meta_path = Path("model/metadata.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        st.info(f"**Current Version:** {meta.get('current_version', 'N/A')}")
        st.metric("Latest Accuracy", f"{meta.get('v2_score', 0):.4f}")

# Connect to MLflow
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("telecom-churn-autogluon")

if experiment:
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
else:
    runs = []

tab1, tab2, tab3 = st.tabs(["📈 Experiment Tracking", "🌊 Drift Analysis", "🔮 Live Prediction"])

with tab1:
    st.header("Model Performance Over Versions")

    if runs:
        # Create metrics dataframe
        data = []
        for r in runs:
            name = r.data.tags.get("mlflow.runName", r.info.run_id[:6])
            acc = r.data.metrics.get("val_accuracy", 0.0)
            data.append(
                {
                    "Version": name,
                    "Accuracy": f"{acc:.4f}",
                    "Run ID": r.info.run_id[:8],
                    "Start Time": r.info.start_time.strftime("%Y-%m-%d %H:%M")
                    if hasattr(r.info.start_time, "strftime")
                    else "N/A",
                }
            )

        df_runs = pd.DataFrame(data)
        st.dataframe(df_runs, use_container_width=True, hide_index=True)

        # Chart
        st.subheader("Accuracy Trend")
        df_chart = df_runs.copy()
        df_chart["Accuracy"] = df_chart["Accuracy"].astype(float)
        st.line_chart(df_chart.set_index("Version")["Accuracy"])

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Models", len(df_runs))
        with col2:
            latest_acc = float(df_runs.iloc[0]["Accuracy"])
            st.metric("Latest Accuracy", f"{latest_acc:.4f}")
        with col3:
            if len(df_runs) > 1:
                prev_acc = float(df_runs.iloc[1]["Accuracy"])
                delta = latest_acc - prev_acc
                st.metric("Change vs Previous", f"{delta:+.4f}", delta=delta)
            else:
                st.metric("Change vs Previous", "N/A")
    else:
        st.info("📌 No runs found. Run `uv run python src/pipeline/pipeline.py` first.")

with tab2:
    st.header("Data Drift Report (Evidently AI)")

    # Custom threshold info
    st.info("🎯 **Our Pipeline Threshold:** 0.05 (KS Test) | **Evidently Default:** 0.5")

    report_path = Path("reports/drift_report.html")
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=1000, scrolling=True)
    else:
        st.warning("⚠️ No drift report found. Run pipeline first.")

    # Load and display custom drift metrics
    st.subheader("📐 Custom Drift Metrics (KS Test)")
    meta_path = Path("model/metadata.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        drift_score = meta.get("triggered_by_drift_score", 0.0)
        st.metric("Custom Drift Score", f"{drift_score:.4f}")

        if drift_score > 0.05:
            st.error(f"⚠️ **DRIFT DETECTED!** Score {drift_score:.4f} > 0.05 threshold")
        else:
            st.success(f"✅ **No Drift** Score {drift_score:.4f} ≤ 0.05 threshold")

with tab3:
    st.header("🔮 Test Model Inference")
    st.write("Enter customer details to predict churn probability:")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 80, 35)
        tenure = st.slider("Tenure (Months)", 1, 72, 24)
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])

    with col2:
        monthly = st.number_input("Monthly Charge", 10.0, 150.0, 85.0)
        total = st.number_input("Total Charges", 100.0, 10000.0, 2000.0)
        contract = st.selectbox("Contract", ["Month-to-Month", "One year", "Two year"])

    if st.button("🚀 Predict Churn", type="primary"):
        payload = {
            "Age": age,
            "Tenure in Months": tenure,
            "Monthly Charge": monthly,
            "Total Charges": total,
            "Gender": gender,
            "Married": married,
            "Contract": contract,
        }

        with st.spinner("Contacting model API..."):
            try:
                import requests

                res = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
                res.raise_for_status()
                result = res.json()

                # Display result
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", "CHURN ⚠️" if result["prediction"] == 1 else "STAY ✅")
                with col2:
                    st.metric("Probability", f"{result['churn_probability']:.2%}")
                with col3:
                    st.metric("Risk Level", result["interpretation"])

                # Progress bar
                st.progress(result["churn_probability"])
                st.caption(f"Model Version: {result['version']}")

            except requests.exceptions.ConnectionError:
                st.error("❌ **API Connection Error:** Is FastAPI running on port 8000?")
                st.code("uv run python -m uvicorn src.serve.app:app --reload --port 8000")
            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")
