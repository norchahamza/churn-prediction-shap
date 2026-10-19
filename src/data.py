"""Load (download or synthesize) and clean the Telco customer churn dataset."""
import urllib.request

import numpy as np
import pandas as pd

from src.config import DATA_PATH, DATA_URL, RANDOM_STATE, TARGET


def generate_synthetic(n: int = 5000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Offline fallback with the same schema as the real Telco dataset."""
    r = np.random.default_rng(seed)
    yn = lambda p=0.5: r.choice(["Yes", "No"], n, p=[p, 1 - p])
    internet = r.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    has_net = internet != "No"
    addon = lambda: np.where(has_net, yn(0.4), "No internet service")
    tenure = r.integers(0, 73, n)
    monthly = np.round(np.where(internet == "Fiber optic", r.normal(90, 12, n),
                                np.where(internet == "DSL", r.normal(55, 10, n),
                                         r.normal(25, 4, n))).clip(18, 120), 2)
    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": r.choice(["Male", "Female"], n), "SeniorCitizen": r.choice([0, 1], n, p=[.84, .16]),
        "Partner": yn(), "Dependents": yn(0.3), "tenure": tenure,
        "PhoneService": yn(0.9), "MultipleLines": yn(0.4), "InternetService": internet,
        "OnlineSecurity": addon(), "OnlineBackup": addon(), "DeviceProtection": addon(),
        "TechSupport": addon(), "StreamingTV": addon(), "StreamingMovies": addon(),
        "Contract": r.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .21, .24]),
        "PaperlessBilling": yn(0.6),
        "PaymentMethod": r.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                   "Credit card (automatic)"], n),
        "MonthlyCharges": monthly, "TotalCharges": np.round(tenure * monthly, 2)})
    z = (-0.6 + 1.3 * (df["Contract"] == "Month-to-month") - 0.035 * tenure
         + 0.015 * (monthly - 60) + 0.5 * (df["PaymentMethod"] == "Electronic check")
         - 0.6 * (df["TechSupport"] == "Yes") - 0.5 * (df["OnlineSecurity"] == "Yes")
         + 0.3 * (df["SeniorCitizen"] == 1) + r.normal(0, 0.5, n))
    df[TARGET] = np.where(r.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")
    return df


def load_raw() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)
            print(f"Downloaded real Telco dataset -> {DATA_PATH}")
        except Exception as exc:  # offline / blocked network
            print(f"Download failed ({exc}); using synthetic data instead.")
            generate_synthetic().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(),
                                       errors="coerce").fillna(0.0)
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].str.strip()
    df[TARGET] = df[TARGET].astype(str).str.lower().map({"yes": 1, "no": 0})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df.reset_index(drop=True)
