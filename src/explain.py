"""SHAP helpers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def make_explainer(model):
    return shap.TreeExplainer(model)


def explain(explainer, X):
    """Return a shap.Explanation in log-odds space for the churn class."""
    exp = explainer(X)
    return exp[:, :, 1] if exp.values.ndim == 3 else exp


def save_global_plots(explainer, X, out_dir):
    exp = explain(explainer, X)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=130)
    plt.close()
    plt.figure()
    shap.plots.bar(exp, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=130)
    plt.close()
