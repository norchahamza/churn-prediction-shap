import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.explain import explain, make_explainer  # noqa: E402

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("Customer Churn Prediction")

MODEL_PATH = ROOT / "models" / "churn_model.joblib"
if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()


@st.cache_resource
def load():
    art = joblib.load(MODEL_PATH)
    return art, make_explainer(art["model"])


art, explainer = load()
defaults, cats = art["defaults"], art["categories"]

tab_predict, tab_insights = st.tabs(["Predict a customer", "Model insights"])

with tab_predict:
    row = dict(defaults)
    c1, c2, c3 = st.columns(3)

    def pick(col, label, box):
        row[col] = box.selectbox(label, cats[col], index=cats[col].index(defaults[col]))

    pick("Contract", "Contract", c1)
    row["tenure"] = c1.slider("Tenure (months)", 0, 72, int(defaults["tenure"]))
    row["MonthlyCharges"] = c1.slider("Monthly charges", 18.0, 120.0, float(defaults["MonthlyCharges"]))
    pick("InternetService", "Internet service", c2)
    pick("PaymentMethod", "Payment method", c2)
    pick("PaperlessBilling", "Paperless billing", c2)
    pick("TechSupport", "Tech support", c3)
    pick("OnlineSecurity", "Online security", c3)
    row["SeniorCitizen"] = int(c3.selectbox("Senior citizen", ["No", "Yes"]) == "Yes")
    row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]

    X = art["preprocessor"].transform(pd.DataFrame([row])[art["input_columns"]])
    proba = float(art["model"].predict_proba(X)[0, 1])
    level = "High" if proba >= 0.6 else "Medium" if proba >= 0.3 else "Low"

    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{proba:.0%}")
    m2.metric("Risk level", level)

    exp = explain(explainer, X)
    contrib = pd.Series(exp.values[0], index=X.columns).sort_values()
    st.subheader("Why this prediction?")
    shap.plots.waterfall(exp[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Raises the risk: " + ", ".join(contrib.tail(3).index[::-1]) +
               "  |  Lowers the risk: " + ", ".join(contrib.head(3).index))

with tab_insights:
    metrics = ROOT / "reports" / "metrics.json"
    if metrics.exists():
        st.subheader("Model comparison (test set)")
        st.dataframe(pd.DataFrame(json.loads(metrics.read_text())).T)
    left, right = st.columns(2)
    for col, name, title in [(left, "shap_importance.png", "Global feature importance"),
                             (right, "shap_summary.png", "SHAP summary (direction & magnitude)")]:
        img = ROOT / "reports" / name
        if img.exists():
            col.subheader(title)
            col.image(str(img))
