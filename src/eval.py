# Imports

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from preprocessing import random_state

# Settings

mode = "dev"                       # dev: out-of-fold training predictions, final: test set (run once)
search_type = "random"             # which tuned models to load
primary = "logistic_regression"    # pre-specified primary model (best in CV)
target_recall = 0.80               # catch >80% of ICU patients (triage: misses cost more than false alarms)
n_boot = 1000                      # bootstrap resamples for confidence intervals
min_cell = 10                      # suppress subgroup results based on fewer stays than this
n_folds = 5

root = Path(__file__).resolve().parents[1]
processed = root / "data" / "processed"
model_dir = root / "data" / "models"
cohort_path = root / "data" / "interim" / "cohort.pkl"
results_dir = root / "results"

model_names = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "mlp"]
labels = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
          "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP", "esi": "ESI alone"}
# Validated categorical palette (fixed order); ESI drawn as a grey dashed reference line
colours = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834", "xgboost": "#1baf7a",
           "lightgbm": "#eda100", "mlp": "#e87ba4", "esi": "#6b6b66"}


# Data and model definitions

def load():
    """Load splits, fitted models and cohort (for ESI and subgroups)"""
    
    train = pd.read_pickle(processed / "train.pkl")
    test = pd.read_pickle(processed / "test.pkl")
    fs = train.get("feature_set", "A")

    models = {}
    for name in model_names:
        path = model_dir / f"{name}_{fs}_{search_type}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
        else:
            print(f"  (no saved model for {name}: {path.name} not found, skipping)")

    # The split keeps the cohort's row numbers, so rows can be matched back to it
    cohort = pd.read_pickle(cohort_path).reset_index(drop=True)
    for part in (train, test):
        assert (cohort.loc[part["X"].index, "subject_id"].values == part["groups"].values).all(), \
            "cohort.pkl dfoes not match saved split: rerun preprocessing.py"
        
    return train, test, cohort, models, fs


def out_of_fold(model, X, y, groups):
    """Predicted risk for each training stay from model that has never seen this patient

    Tuned pipeline is cloned (same hyperparameters, unfitted) and refitted in
    each patient-grouped fold
    """

    cv = StratifiedGroupKFold(n_splits = n_folds, shuffle = True, random_state = random_state)
    return cross_val_predict(clone(model), X, y, groups = groups, cv = cv, method = "predict_proba")[:, 1]


def esi_score(cohort, index):
    """ESI as a risk score - lower ESI means more urgent, so scale is flipped.
    Missing acuity (triage bypassed due to urgency) is treated as most urgent level (0)"""

    return -cohort.loc[index, "acuity"].fillna(0).values


def threshold_for_recall(y, p, target):
    """Highest threshold whose recall is still >= target (fewest false alarms)"""

    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]]) if len(ok) else 0.5

# Metrics

def classification_metrics(y, p, threshold):
    """Sensitivity, specificity, PPV, NPV and F1 at given threshold."""

    pred = p >= threshold
    tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
    fn, tn = np.sum(~pred & (y == 1)), np.sum(~pred & (y == 0))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens else np.nan
    return {"sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv, "f1": f1,
            "flagged_%": 100 * pred.mean()}

def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan

def patient_bootstrap(groups, n_boot, seed = random_state):
    """Yield row indices for n_boot resamples of patients (all stays of patient together)"""

    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    patients = np.unique(g)
    rows_of = {pt: np.where(g == pt)[0] for pt in patients}
    for _ in range(n_boot):
        drawn = rng.choice(patients, size = len(patients), replace = True)
        yield np.concatenate([rows_of[pt] for pt in drawn])

def ci(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)

def calibration_slope(y, p):
    """Slope of logistic regression of outcome on logit(predicted risk)
    1 = well calibrated; < 1 = predictions too extreme; > 1 = too cautious"""

    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(C=1e6).fit(logit, y).coef_[0, 0])