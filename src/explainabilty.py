"""

Explainability: which features drive the models' predictions using SHAP

Input:  data/processed/train.pkl                      (src/preprocessing.py)
        data/models/<model>_<set>_<search>.joblib     (src/models.py)

        Output:
        results/shap_lr_importance.csv       mean |SHAP|, coefficient and odds ratio per feature
        results/shap_comparison.csv          feature ranking: logistic regression vs XGBoost
        results/fig_shap_lr_summary.png      SHAP summary (beeswarm) plot, logistic regression
        results/fig_shap_lr_bar.png          mean |SHAP| bar chart, logistic regression
        results/explanatory_odds_ratios.csv  adjusted odds ratios with 95% CIs (explanatory model)

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

# Figures

def plot_bar(importance, path):
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize = (7, 0.32 * len(top) + 1.2))
    ax.barh([label(f) for f in top.index], top.values, color = accent, height = 0.6)
    for i, v in enumerate(top.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize = 8, color = "#3f3f3c")
    ax.set_xlabel("Mean |SHAP value| (average impact on log-odds of ICU admission)")
    ax.set_title("Most influential features, logistic regression (training set)", loc = "left", fontsize = 11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, top.values.max() * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)

def plot_summary(shap_values, X_raw, importance, path, seed=42):
    """
    Beeswarm-style summary: one dot per patient per feature.
    Position = SHAP value, colour = the patient's feature value (light = low, dark = high)
    """
    
    feats = list(importance.head(top_n).index)[::-1]
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(shap_values), size = min(max_points, len(shap_values)), replace = False)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#2a78d6", "#0d366b"])

    fig, ax = plt.subplots(figsize = (7.5, 0.36 * len(feats) + 1.6))
    for i, f in enumerate(feats):
        s = shap_values[f].values[rows]
        raw = X_raw[f].values[rows].astype(float) if f in X_raw else np.full(len(rows), np.nan)
        # colour by rank in feature (robust for outliers) and missing raw values in grey
        known = ~np.isnan(raw)
        colour_val = np.full(len(raw), np.nan)
        if known.sum() > 1:
            colour_val[known] = pd.Series(raw[known]).rank(pct=True).values
        jitter = rng.normal(0, 0.08, len(s))
        ax.scatter(s[~known], i + jitter[~known], s = 6, color = "#c3c2b7", alpha = 0.7, linewidths = 0)
        ax.scatter(s[known], i + jitter[known], s = 6, c = colour_val[known], cmap = cmap, vmin = 0, vmax = 1,
                   alpha = 0.8, linewidths = 0)
        
    ax.axvline(0, color = "#8a8984", lw =  0.8)
    ax.set_yticks(range(len(feats)), [label(f) for f in feats], fontsize = 9)
    ax.set_xlabel("SHAP value (impact on log-odds of ICU admission)  <- lowers risk | raises risk ->")
    ax.set_title("How each feature shifts predicted ICU risk, logistic regression", loc = "left", fontsize = 11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax = ax, fraction = 0.03, pad = 0.02, ticks = [0, 1])
    cbar.ax.set_yticklabels(["Low", "High"], fontsize = 8)
    cbar.set_label("Feature value (grey = missing, imputed)", fontsize = 8)
    fig.tight_layout()
    fig.savefig(path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)

# Run

def main():
    warnings.filterwarnings("ignore")
    train = pd.read_pickle(processed / "train.pkl")
    X = train["X"]
    fs = train.get("feature_set", "A")
    results_dir.mkdir(parents = True, exist_ok = True)

    # Logistic regression (primary model): exact linear SHAP
    lr = joblib.load(model_dir / f"logistic_regression_{fs}_{search_type}.joblib")
    print(f"Logistic regression, feature set {fs}: {len(X)} training stays")
    lr_shap, Xt, coef = linear_shap(lr, X)
    check_against_library(lr, Xt, lr_shap)

    # Check patient's log-odds = the same baseline + the sum of their SHAP values
    baseline = lr.decision_function(X) - lr_shap.sum(axis = 1).values
    print(f"  additivity check (spread of baseline, should be ~0): {baseline.std():.2e}")

    importance = lr_shap.abs().mean().sort_values(ascending = False)
    # Odds ratio per unit of preprocessed feature: per 1 SD for continuous features
    # (they are standardised), yes vs no for 0/1 flags
    binary = {f for f in X.columns if set(X[f].dropna().unique()) <= {0, 1}}
    table = pd.DataFrame({
        "feature": [label(f) for f in importance.index],
        "mean_abs_shap": importance.values,
        "coefficient": coef[importance.index].values,
        "odds_ratio": np.exp(coef[importance.index].values),
        "odds_ratio_per": ["yes vs no" if f in binary else "1 SD" for f in importance.index],
    })

    table.round(4).to_csv(results_dir / "shap_lr_importance.csv", index = False)
    plot_bar(importance, results_dir / "fig_shap_lr_bar.png")
    plot_summary(lr_shap, X, importance, results_dir / "fig_shap_lr_summary.png")

    pd.set_option("display.width", 200)
    print(f"\nTop {top_n} features, logistic regression (training set):")
    print(table.head(top_n).round(3).to_string(index = False))

    # Explanatory model: interpretable adjusted odds ratios
    
    ors = explanatory_odds_ratios(X, train["y"].values)
    ors.round(4).to_csv(results_dir / "explanatory_odds_ratios.csv", index = False)
    plot_odds_ratios(ors, results_dir / "fig_odds_ratios.png")
    print("\nAdjusted odds ratios (reference: ESI 2, ambulance arrival, training set):")
    print(ors.round(3).to_string(index = False))

    # XGBoost: whether non-linear model relies on the same features

    xgb_path = model_dir / f"xgboost_{fs}_{search_type}.joblib"
    try:
        xgb_shap = tree_shap(joblib.load(xgb_path), X)
        xgb_importance = xgb_shap.abs().mean()
        both = pd.DataFrame({"lr_mean_abs_shap": importance, "xgb_mean_abs_shap": xgb_importance})
        both["lr_rank"] = both["lr_mean_abs_shap"].rank(ascending = False).astype(int)
        both["xgb_rank"] = both["xgb_mean_abs_shap"].rank(ascending = False).astype(int)
        both = both.sort_values("lr_rank")
        both.index = [label(f) for f in both.index]
        both.round(4).to_csv(results_dir / "shap_comparison.csv")
        rho = spearmanr(both["lr_rank"], both["xgb_rank"]).correlation
        print(f"\nFeature ranking, logistic regression vs XGBoost (Spearman rho = {rho:.2f}):")
        print(both.head(top_n)[["lr_rank", "xgb_rank"]].to_string())
    
    except Exception as err:
        print(f"\n(XGBoost comparison skipped: {err.__class__.__name__}: {err}. Try: pip install shap)")

    print("\nSaved results/shap_*.csv and results/fig_shap_lr_*.png")

if __name__ == "__main__":
    main()