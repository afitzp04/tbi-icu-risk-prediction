"""

Evaluation: performance of models compared to ESI baseline

Inputs:  data/processed/train.pkl, data/processed/test.pkl    (src/preprocessing.py)
        data/models/<model>_<set>_<search>.joblib            (src/models.py)
        data/interim/cohort.pkl                              (subgroup variables, ESI)

Outputs: results/eval_<mode>_metrics.csv      performance of each model + ESI, with 95% CIs
        results/eval_<mode>_comparisons.csv  AUROC differences vs ESI and vs logistic regression
        results/eval_<mode>_subgroups.csv    performance of the primary model in patient subgroups
        results/eval_<mode>_thresholds.csv   decision threshold chosen for each model
        results/fig_<mode>_roc.png, results/fig_<mode>_calibration.png
        data/processed/predictions_<mode>.pkl   (patient-level predictions; not committed)


- dev:    evaluates OUT-OF-FOLD predictions on training set, used to build and
          check evaluation (test set untouched)
- final:  evaluates held-out test set, ran once after models and features are frozen
          and report results

Design notes
- Decision thresholds chosen on out-of-fold training predictions (never on test set):
  lowest-risk cut-off that still catches target_recall of ICU patients
- 95% confidence intervals come from bootstrap resampling patients so repeat visits
  stay together. The same resamples are used for every model so differences between
  models will be paired
- Subgroup cells with fewer than min_cell stays are suppressed before saving so saved
  tables can be committed

"""

#  Imports

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
from sklearn.metrics import average_precision_score

from preprocessing import random_state

# Settings

mode = "dev"                       # dev: out-of-fold training predictions, final: test set (run once)
search_type = "random"             # which tuned models to load
primary = "logistic_regression"    # pre-specified primary model (best in CV)
target_recall = 0.80               # catch > 80% of ICU patients (triage: misses cost more than false alarms)
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

    # Split keeps cohort's row numbers so rows can be matched back
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
    1 = well calibrated
    < 1 = predictions too extreme
    > 1 = too cautious
    """

    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(C=1e6).fit(logit, y).coef_[0, 0])

# Figures

def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha = 0.25, linewidth = 0.6)

def plot_roc(y, preds, aucs, path, title):
    fig, ax = plt.subplots(figsize = (6, 5.5))
    ax.plot([0, 1], [0, 1], color = "#c3c2b7", lw = 1, ls = ":", label = "Chance (0.50)")
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        is_esi = name == "esi"
        ax.plot(fpr, tpr, color = colours[name], lw = 2, ls = "--" if is_esi else "-",
                label = f"{labels[name]} ({aucs[name]:.3f})")
    ax.set(xlabel = "1 − specificity (false-positive rate)", ylabel = "Sensitivity (true-positive rate)",
           title = title, xlim = (0, 1), ylim = (0, 1.01))
    ax.legend(loc="lower right", fontsize = 8.5, frameon = False, title = "AUROC", title_fontsize = 8.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)

def plot_calibration(y, preds, path, title):
    fig, ax = plt.subplots(figsize = (6, 5.5))
    ax.plot([0, 1], [0, 1], color = "#c3c2b7", lw = 1, ls = ":", label = "Perfect calibration")
    for name, p in preds.items():
        if name == "esi":
            continue # ESI is a rank, not a probability
        obs, pred = calibration_curve(y, p, n_bins = 10, strategy = "quantile")
        ax.plot(pred, obs, color = colours[name], lw = 2, marker = "o", ms = 4, label = labels[name])
    ax.set(xlabel = "Predicted risk of ICU admission", ylabel = "Observed proportion admitted to ICU",
           title = title, xlim = (0, 1), ylim = (0, 1))
    ax.legend(loc = "upper left", fontsize = 8.5, frameon = False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)

# Decision curve

def net_benefit(y, p, thresholds):
    """
    Net benefit = true positives/n - false positives/n x (odds of the threshold)
    """
    
    n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)

def plot_decision_curve(y, preds, path, title):
    """
    Decision curve: net benefit of each model vs 'treat all' and 'treat none'
    """
    
    thr = np.round(np.arange(0.05, 0.81, 0.01), 2)
    prev = y.mean()
    treat_all = prev - (1 - prev) * thr / (1 - thr)
    table = pd.DataFrame({"threshold": thr, "treat_all": treat_all, "treat_none": 0.0})

    fig, ax = plt.subplots(figsize = (6.5, 5))
    ax.plot(thr, treat_all, color = "#8a8984", lw = 1.2, label = "Treat all as ICU")
    ax.axhline(0, color = "#c3c2b7", lw = 1.2, ls = ":", label = "Treat none")
    for name, p in preds.items():
        if name == "esi":
            continue # ESI is  rank, not probability
        nb = net_benefit(y, p, thr)
        table[labels[name]] = nb
        ax.plot(thr, nb, color = colours[name], lw = 2 , label = labels[name])

    # Currently triage-bypassed patients count as highest acuity in esi_score
    if "esi" in preds:
        rules = {"ESI rule: ICU if ESI 1 or bypassed": -1,
                 "ESI rule: ICU if ESI 1-2 or bypassed": -2,}
        for (rule, cut), ls in zip(rules.items(),["--", "-."]):
            flagged = (preds["esi"] >= cut).astype(float) # 1 = sent to ICU by the rule
            nb = net_benefit(y, flagged, thr)
            table[rule] = nb
            ax.plot(thr, nb, color = colours["esi"], lw = 1.6, ls = ls, label = rule)

    ax.set(xlabel = "Threshold probability of ICU admission", ylabel = "Net benefit",
           title = title, xlim = (0.05, 0.8), ylim = (-0.05, prev + 0.05))
    ax.legend(loc = "upper right", fontsize = 8.5, frameon = False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    return table

# Subgroups

def race_group(race):
    """Collapse detailed race categories from MIMIC into broad groups"""

    r = str(race).upper()
    if r.startswith("WHITE"):
        return "White"
    if r.startswith("BLACK"):
        return "Black"
    if r.startswith("HISPANIC") or "LATINO" in r:
        return "Hispanic/Latino"
    if r.startswith("ASIAN"):
        return "Asian"
    return "Other/unknown"

def subgroup_table(eval_cohort, X, y, p, esi, threshold, groups):
    """Performance of primary model (and ESI) within each patient subgroup

    95% CIs come from a patient-level bootstrap WITHIN each subgroup so small
    subgroups get correctly wide intervals. The gain CI is paired: model and ESI
    are scored on the same resamples

    """

    defs = {
        "Triage": np.where(X["triage_bypassed"] == 1, "Bypassed", "Normally triaged"),
        "Sex": np.where(X["male"] == 1, "Male", "Female"),
        "Age": np.where(eval_cohort["age"].values >= 65, "65+", "18-64"),
        "Race": eval_cohort["race"].map(race_group).values,
    }
    if "cc_transfer" in X:
        defs["Transfer"] = np.where(X["cc_transfer"] == 1, "Transferred in", "Not transferred")

    rows = []
    for variable, group_labels in defs.items():
        for level in pd.unique(group_labels):
            m = group_labels == level
            n = int(m.sum())
            row = {"subgroup": variable, "level": level, "stays": n}
            if n < min_cell:
                row["note"] = f"suppressed (n < {min_cell})"

            else:
                ys, ps, es, gs = y[m], p[m], esi[m], np.asarray(groups)[m]
                boot_model, boot_gain = [], []
                for idx in patient_bootstrap(gs, n_boot):
                    a_model, a_esi = safe_auc(ys[idx], ps[idx]), safe_auc(ys[idx], es[idx])
                    boot_model.append(a_model)
                    boot_gain.append(a_model - a_esi)
                model_lo, model_hi = ci(boot_model)
                gain_lo, gain_hi = ci(boot_gain)
                row.update({"icu_rate": ys.mean(),
                            "auroc_model": safe_auc(ys, ps), "auroc_model_ci_low": model_lo,
                            "auroc_model_ci_high": model_hi,
                            "auroc_esi": safe_auc(ys, es),
                            "gain_vs_esi": safe_auc(ys, ps) - safe_auc(ys, es),
                            "gain_ci_low": gain_lo, "gain_ci_high": gain_hi,
                            **classification_metrics(ys, ps, threshold)})

            rows.append(row)
    return pd.DataFrame(rows)

# Sensitivity analysis for direct arrivals

def sensitivity_direct(train, test, models, cohort):
    """Sensitivity analysis: transferred-in patients removed from training AND evaluation.

    The primary model is refitted (same hyperparameters) on direct arrivals only,
    then compared with ESI on the same patients, with a paired patient bootstrap.
    """
    keep_tr = train["X"]["cc_transfer"] == 0
    xtr = train["X"][keep_tr].drop(columns = "cc_transfer")    # flag is constant once transfers are removed
    ytr = train["y"][keep_tr].values
    gtr = train["groups"][keep_tr]
    model = clone(models[primary])                            # same settings, unfitted

    if mode == "final":
        keep_te = test["X"]["cc_transfer"] == 0
        x = test["X"][keep_te].drop(columns = "cc_transfer")
        y = test["y"][keep_te].values
        g = test["groups"][keep_te].values
        p = model.fit(xtr, ytr).predict_proba(x)[:, 1]
    else:
        x, y, g = xtr, ytr, gtr.values
        p = out_of_fold(model, xtr, ytr, gtr)

    esi = esi_score(cohort, x.index)
    boot_model, boot_esi = [], []
    for idx in patient_bootstrap(g, n_boot):
        boot_model.append(safe_auc(y[idx], p[idx]))
        boot_esi.append(safe_auc(y[idx], esi[idx]))
    boot_model, boot_esi = np.array(boot_model), np.array(boot_esi)
    diff = boot_model - boot_esi

    auc_model, auc_esi = safe_auc(y, p), safe_auc(y, esi)
    return pd.DataFrame([{
        "analysis": "direct arrivals only (transfers excluded)",
        "stays": len(y), "patients": len(np.unique(g)), "icu_rate": y.mean(),
        "auroc_model": auc_model, "model_ci_low": ci(boot_model)[0], "model_ci_high": ci(boot_model)[1],
        "auroc_esi": auc_esi, "esi_ci_low": ci(boot_esi)[0], "esi_ci_high": ci(boot_esi)[1],
        "difference": auc_model - auc_esi, "diff_ci_low": ci(diff)[0], "diff_ci_high": ci(diff)[1],
    }])

# Run

def main():
    warnings.filterwarnings("ignore")
    assert mode in ("dev", "final"), 'mode must be "dev" or "final"'
    train, test, cohort, models, fs = load()
    print(f"Feature set {fs}, {search_type} search, models: {list(models)}")

    # OOF training predictions: used to choose thresholds (both modes)
    # and as the evaluation data in dev mode

    Xtr, ytr, gtr = train["X"], train["y"].values, train["groups"]
    print("Computing out-of-fold training predictions ...")
    oof = {name: out_of_fold(m, Xtr, ytr, gtr) for name, m in models.items()}
    thresholds = {name: threshold_for_recall(ytr, p, target_recall) for name, p in oof.items()}

    # Evaluation data

    if mode == "final":
        print("\n*** Final mode: evaluate the held-out testing sset. Run only once. ***\n")
        X, y, groups = test["X"], test["y"].values, test["groups"].values
        preds = {name: m.predict_proba(X)[:, 1] for name, m in models.items()}
    else:
        print("\nDev mode: evaluate out-of-fold training predictions (test set untouched).\n")
        X, y, groups = Xtr, ytr, gtr.values
        preds = dict(oof)
    preds["esi"] = esi_score(cohort, X.index)
    eval_cohort = cohort.loc[X.index]

    # Point estimates

    rows = []
    for name, p in preds.items():
        row = {"model": labels[name], "auroc": safe_auc(y, p), "auprc": average_precision_score(y, p)}
        if name != "esi":
            row.update({"brier": brier_score_loss(y, p), "calibration_slope": calibration_slope(y, p),
                        "threshold": thresholds[name], **classification_metrics(y, p, thresholds[name])})
        rows.append(row)
    metrics = pd.DataFrame(rows).set_index("model")

    # Bootstrap CIs (same patient resamples for every model -> paired comparisons)
    
    print(f"Bootstrapping {n_boot} patient resamples ...")
    boot = {name: [] for name in preds}
    for idx in patient_bootstrap(groups, n_boot):
        for name, p in preds.items():
            boot[name].append(safe_auc(y[idx], p[idx]))
    boot = {k: np.array(v) for k, v in boot.items()}
    for name in preds:
        lo, hi = ci(boot[name])
        metrics.loc[labels[name], ["auroc_ci_low", "auroc_ci_high"]] = [lo, hi]

    comparisons = []
    for ref in ("esi", primary):
        for name in models:
            if name == ref or ref not in preds:
                continue
            diff = boot[name] - boot[ref]
            lo, hi = ci(diff)
            comparisons.append({"model": labels[name], "compared_with": labels[ref],
                                "auroc_difference": safe_auc(y, preds[name]) - safe_auc(y, preds[ref]),
                                "ci_low": lo, "ci_high": hi,
                                "p_two_sided": min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))})
    comparisons = pd.DataFrame(comparisons)

    # Subgroups (primary model)
    
    print("Bootstrapping subgroups...")
    subgroups = subgroup_table(eval_cohort, X, y, preds[primary], preds["esi"], thresholds[primary], groups) \
        if primary in preds else pd.DataFrame()

    # Sensitivity analysis: direct arrivals only
    
    print("Sensitivity analysis: refitting without transferred patients ...")
    sensitivity = sensitivity_direct(train, test, models, cohort)

    # Save

    results_dir.mkdir(parents = True, exist_ok = True)
    tag = mode

    metrics.round(3).to_csv(results_dir / f"eval_{tag}_metrics.csv")
    comparisons.round(4).to_csv(results_dir / f"eval_{tag}_comparisons.csv", index = False)
    subgroups.round(3).to_csv(results_dir / f"eval_{tag}_subgroups.csv", index = False)
    sensitivity.round(3).to_csv(results_dir / f"eval_{tag}_sensitivity.csv", index = False)
    
    pd.Series(thresholds, name = "threshold").round(4).to_csv(results_dir / f"eval_{tag}_thresholds.csv")
   
    where = "test set" if mode == "final" else "out-of-fold, training set"
    plot_roc(y, preds, metrics["auroc"].rename(index = {v: k for k, v in labels.items()}).to_dict(),
             results_dir / f"fig_{tag}_roc.png", f"ROC curves ({where})")
    plot_calibration(y, preds, results_dir / f"fig_{tag}_calibration.png", f"Calibration ({where})")
    pd.to_pickle({"y": y, "groups": groups, "index": X.index, "preds": preds, "thresholds": thresholds},
                 processed / f"predictions_{tag}.pkl")

    #Decision curve
    decision = plot_decision_curve(y, preds, results_dir / f"fig_{tag}_decision_curve.png",
                                   f"Decision curve ({where})")
    decision.round(4).to_csv(results_dir / f"eval_{tag}_decision_curve.csv", index=False)

    # Print summary

    pd.set_option("display.width", 200)
   
    show = ["auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "brier", "calibration_slope",
            "sensitivity", "specificity", "ppv", "flagged_%"]
    print(f"\nPerformance ({where}); thresholds set for {target_recall:.0%} recall on training data:")
    print(metrics[show].round(3).to_string())
    print("\nAUROC differences (paired bootstrap 95% CI):")
    print(comparisons.round(3).to_string(index = False))
    print(f"\nSubgroups, {labels.get(primary, primary)}:")
    cols = [c for c in ["subgroup", "level", "stays", "icu_rate", "auroc_model", "auroc_model_ci_low",
                        "auroc_model_ci_high", "auroc_esi", "gain_vs_esi", "gain_ci_low", "gain_ci_high",
                        "sensitivity", "specificity", "note"] if c in subgroups]
    print(subgroups[cols].round(3).to_string(index = False))
    print("\nSensitivity analysis (direct arrivals only):")
    print(sensitivity.round(3).T.to_string(header=False))
    print(f"\nSaved results/eval_{tag}_*.csv and results/fig_{tag}_*.png")

if __name__ == "__main__":
    main()
