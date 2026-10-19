from src.data import clean, generate_synthetic
from src.explain import explain, make_explainer
from src.train import train


def test_clean_handles_blank_total_charges_and_target():
    df = generate_synthetic(200)
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    df.loc[0, "TotalCharges"] = " "
    out = clean(df)
    assert out["TotalCharges"].isna().sum() == 0
    assert set(out["Churn"].unique()) <= {0, 1}
    assert "customerID" not in out.columns


def test_train_and_explain():
    df = clean(generate_synthetic(1500))
    art, metrics, Pte = train(df)
    assert metrics["gradient_boosting"]["roc_auc"] > 0.65
    exp = explain(make_explainer(art["model"]), Pte.head(5))
    assert exp.values.shape == (5, Pte.shape[1])
