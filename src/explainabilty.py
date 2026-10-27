"""

Explainability: which features drive the models' predictions using SHAP

Input:  data/processed/train.pkl                      (src/preprocessing.py)
        data/models/<model>_<set>_<search>.joblib     (src/models.py)

        Output:
        results/shap_lr_importance.csv    mean |SHAP|, coefficient and odds ratio per feature
        results/shap_comparison.csv       feature ranking: logistic regression vs XGBoost
        results/fig_shap_lr_summary.png   SHAP summary (beeswarm) plot, logistic regression
        results/fig_shap_lr_bar.png       mean |SHAP| bar chart, logistic regression

Design notes:
- SHAP (SHapley Additive exPlanations) splits each prediction into a contribution
  from every feature. Positive values push patient's predicted ICU risk up, and
  negative values push it down (values are on the log-odds scale)
- Explanations computed on the training set so testing set stays untouched
- Logistic regression (the primary model): with linear model, SHAP value of
  feature j for patient is exactly  coefficient_j x (patient's value_j - average value_j),
  computed on the preprocessed features. This is returned by shap.LinearExplainer

"""

# Imports

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Settings

search_type = "random"     # which tuned models to load (matches models.py)
top_n = 15                 # features shown in the figures
max_points = 2000          # patients drawn in the summary plot

root = Path(__file__).resolve().parents[1]
processed = root / "data" / "processed"
model_dir = root / "data" / "models"
results_dir = root / "results"

# Readable names for writing report

nice = {
    "age": "Age", "temperature": "Temperature", "heartrate": "Heart rate", "resprate": "Respiratory rate",
    "o2sat": "O2 saturation", "sbp": "Systolic BP", "dbp": "Diastolic BP", "pain_score": "Pain score",
    "male": "Male", "triage_bypassed": "Triage bypassed", "pain_unable": "Pain not assessable",
    "esi_1": "ESI 1", "esi_2": "ESI 2", "esi_3plus": "ESI 3-5", "esi_missing": "ESI missing",
    "arrival_ambulance": "Arrival: ambulance", "arrival_helicopter": "Arrival: helicopter",
    "arrival_walk_in": "Arrival: walk-in", "arrival_unknown": "Arrival: unknown", "arrival_other": "Arrival: other",
    "cc_ich": "Complaint: ICH / head bleed", "cc_sdh_sah": "Complaint: SDH / SAH",
    "cc_transfer": "Complaint: transfer", "cc_impact": "Complaint: high-energy impact",
    "cc_high_energy": "Complaint: high-energy impact", "cc_altered_mental": "Complaint: altered mental state",
    "cc_fall": "Complaint: fall", "shock_index": "Shock index",
}

accent = "#2a78d6"

def label(name):
    return nice.get(name, name)

# SHAP values

def transformed(pipe, X):
    """
    Features after the pipeline's preprocessing (median imputation + scaling), with names
    """
    
    prep = pipe.named_steps["prep"]
    Xt = prep.transform(X)
    return pd.DataFrame(np.asarray(Xt, dtype = float), columns = prep.get_feature_names_out(), index = X.index)

def linear_shap(pipe, X):
    """
    Exact SHAP values for the logistic regression pipeline (log-odds scale)
    """
    
    Xt = transformed(pipe, X)
    coef = pipe.named_steps["model"].coef_.ravel()
    shap_values = (Xt - Xt.mean()) * coef
    return shap_values, Xt, pd.Series(coef, index = Xt.columns)

def check_against_library(pipe, Xt, shap_values):
    """
    If shap is installed, confirm the hand-computed values match shap.LinearExplainer
    """
    
    try:
        import shap
    except ImportError:
        # make sure installed
        print("  (shap not installed: skipping the library cross-check)")
        return
    explainer = shap.LinearExplainer(pipe.named_steps["model"],
                                     shap.maskers.Independent(Xt.values, max_samples=len(Xt)))
    lib = np.asarray(explainer.shap_values(Xt.values))
    gap = float(np.max(np.abs(lib - shap_values.values)))
    print(f"  cross-check with shap.LinearExplainer: max difference {gap:.2e}")

def tree_shap(pipe, X):
    """
    SHAP values for a tree model (XGBoost) via shap.TreeExplainer
    """

    import shap
    Xt = transformed(pipe, X)
    values = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
    values = values[1] if isinstance(values, list) else values      # versions may return one array per class
    return pd.DataFrame(np.asarray(values), columns = Xt.columns, index = Xt.index)

# Explanatory model: adjusted odds ratios with reference categories

reference = ["esi_2", "arrival_ambulance"]   # most common category in each group for comparison

# Continuous features reported per clinically meaningful unit
units = {"age": (10, "per 10 years"), "temperature": (1, "per 1 °F"), "heartrate": (10, "per 10 bpm"),
         "resprate": (1, "per 1 breath/min"), "o2sat": (1, "per 1%"), "sbp": (10, "per 10 mmHg"),
         "dbp": (10, "per 10 mmHg"), "pain_score": (1, "per 1 point"), "shock_index": (0.1, "per 0.1")}


def explanatory_odds_ratios(X, y):
    """
    Unpenalised logistic regression with reference categories - returns ORs with 95% CIs.

    Missing continuous values are filled with training median (triage_bypassed
    flag carries the information that they were missing). Columns that do not vary
    and reference categories are dropped
    """

    from sklearn.linear_model import LogisticRegression
    from scipy.stats import norm
    Z = X.drop(columns = [c for c in reference if c in X.columns]).copy()
    for col, (step, _) in units.items():
        if col in Z:
            Z[col] = Z[col].fillna(Z[col].median()) / step
    Z = Z.loc[:, Z.nunique() > 1].astype(float)

    model = LogisticRegression(penalty = None, max_iter = 10000).fit(Z, y)
    # Standard errors from the inverse of the information matrix X'WX (with intercept)
    D = np.column_stack([np.ones(len(Z)), Z.values])
    p = model.predict_proba(Z)[:, 1]
    cov = np.linalg.pinv(D.T @ (D * (p * (1 - p))[:, None]))
    se = np.sqrt(np.diag(cov))[1:]
    beta = model.coef_.ravel()
    out = pd.DataFrame({
        "feature": [label(c) for c in Z.columns],
        "per": [units[c][1] if c in units else "yes vs no" for c in Z.columns],
        "odds_ratio": np.exp(beta),
        "ci_low": np.exp(beta - 1.96 * se),
        "ci_high": np.exp(beta + 1.96 * se),
        "p_value": 2 * norm.sf(np.abs(beta / se)),
    }, index = Z.columns)
    
    # Separation: category where nearly everyone had same outcome gives estimate
    # that runs off towards 0 or infinity. Flagged not reported
    unstable = (out["ci_high"] > 1e3) | (out["ci_low"] < 1e-3)
    out["note"] = np.where(unstable, "unstable estimate (separation or very few patients)", "")
    return out.sort_values("odds_ratio", ascending = False)


def plot_odds_ratios(table, path):
    """
    Forest plot of adjusted odds ratios (log scale) and reference line at 1
    """

    t = table[table["note"] == ""].iloc[::-1]      # unstable estimates not drawn
    sig = (t["ci_low"] > 1) | (t["ci_high"] < 1)   # CI excludes 1 -> blue
    fig, ax = plt.subplots(figsize = (7.5, 0.3 * len(t) + 1.3))
    ys = np.arange(len(t))
    ax.hlines(ys, t["ci_low"], t["ci_high"], color = np.where(sig, accent, "#b5b4ad"), lw = 1.8)
    ax.scatter(t["odds_ratio"], ys, color = np.where(sig, accent, "#b5b4ad"), s = 22, zorder = 3)
    ax.axvline(1, color = "#8a8984", lw = 0.9, ls = "--")
    ax.set_xscale("log")
    lo, hi = t["ci_low"].min(), t["ci_high"].max()
    ticks = [v for v in [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16] if lo / 1.5 <= v <= hi * 1.5]
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(ys, [f"{f} ({p})" for f, p in zip(t["feature"], t["per"])], fontsize = 8.0)
    ax.set_xlabel("Adjusted odds ratio of ICU admission (95% CI, log scale)  <- lower risk | higher risk ->")
    ax.set_title("Adjusted odds ratios (reference: ESI 2, ambulance arrival)", loc = "left", fontsize = 11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi = 300, bbox_inches = "tight")
    
    plt.close(fig)