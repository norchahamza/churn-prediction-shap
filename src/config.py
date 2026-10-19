from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")
TARGET = "Churn"
RANDOM_STATE = 42
