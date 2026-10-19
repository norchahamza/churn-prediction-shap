"""Train baseline + gradient boosting, evaluate, and save artifacts + SHAP plots."""
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from src.data import clean, load_raw
from src.explain import make_explainer, save_global_plots


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer(
        [("num", "passthrough", num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        verbose_feature_names_out=False)
    return pre.set_output(transform="pandas")


def evaluate(model, X, y) -> dict:
    proba, pred = model.predict_proba(X)[:, 1], model.predict(X)
    return {"roc_auc": roc_auc_score(y, proba), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred)}


def train(df: pd.DataFrame):
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=RANDOM_STATE)
    pre = build_preprocessor(Xtr)
    Ptr, Pte = pre.fit_transform(Xtr), pre.transform(Xte)

    baseline = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    baseline.fit(Ptr, ytr)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                    random_state=RANDOM_STATE).fit(Ptr, ytr)
    metrics = {"logistic_regression": evaluate(baseline, Pte, yte),
               "gradient_boosting": evaluate(gb, Pte, yte)}
    metrics = {m: {k: round(float(v), 3) for k, v in d.items()} for m, d in metrics.items()}

    cat_cols = X.select_dtypes(exclude="number").columns
    artifacts = {
        "preprocessor": pre, "model": gb, "input_columns": list(X.columns),
        "categories": {c: sorted(X[c].unique().tolist()) for c in cat_cols},
        "defaults": {c: (X[c].mode()[0] if c in cat_cols else float(X[c].median()))
                     for c in X.columns}}
    return artifacts, metrics, Pte


def main():
    df = clean(load_raw())
    print(f"{len(df)} customers, churn rate = {df[TARGET].mean():.1%}")
    artifacts, metrics, Pte = train(df)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(artifacts, MODEL_PATH)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_global_plots(make_explainer(artifacts["model"]), Pte.sample(min(1000, len(Pte)),
                                                                     random_state=RANDOM_STATE),
                      REPORTS_DIR)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
