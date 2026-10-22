"""Model development: train classifier on training set

Input: data/processed/train.pkl     created by src/preprocessing.py

Output: data/models/model.joblib    fitted models, used in src/eval.py
        results/cv_results.csv      cross validation performance
        
Run from project root folder

Notes:
- testing set is not involved here, only in eval.py for final evaluation
- each model = pipeline: preproccesor -> classifier
    - pipeline is re fitted for every CV folds so imputation median and scaling will be learned from training folds only
    - this prevents leakage
- CV folds grouped by patient so one patient's visits are never in both training and validation fold
- Class weighting deals with the mild class imbalance of 44% ICU
    - SMOTE not required due to mildness
-  Hyperparameters chosen by grid search to maximise the AUROC
"""

# Imports

import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from preprocessing import random_state, make_preprocessor

# Paths

root = Path(__file__).resolve().parents[1]
train_path = root / "data" / "processed" / "train.pkl"
model_dir = root / "data" / "models"
results_dir = root / "results"

n_folds = 5

# Scores

scoring = {"auroc": "roc_auc", 
           "recall": "recall",
           "precision": "precision",
           "f1": "f1",
           "brier": "neg_brier_score"}

# Candidate models
def candidate_models():
    """Returns {name: (classifier, parameter grid)}
    
    Gridkeys start with model__ so classifier becomes step called "model" in the pipeline
    Grids are small to avoid fitting noise
    """

    models = {
        # Interpretable baseline where C = inverse regularisation strength
        # Smaller = simpler

        "logistic_regression": (
            LogisticRegression(class_weight = "balanced", max_iter = 2000),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        
        # Averages multiple decision trees to capture non-linear effects

        "random_forest": (
            RandomForestClassifier(n_estimators = 300, class_weight = "balanced",
                                   random_state = random_state, n_jobs = -1),
                                {"model__max_depth": [4, 6, 8, None],
                                 "model__min_samples_leaf": [5, 20]},
        ),
    }

    # Gradient boosting to build trees one after another that correct previous
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = (
            XGBClassifier(n_estimators = 300, subsample = 0.8, colsample_bytree = 0.8,
                          eval_metric = "logloss", random_state = random_state, n_jobs = -1),
                          {"model__max_depth": [2, 3, 4],
                           "model__learning_rate": [0.03, 0.1]},
        )
    except ImportError:
        # Skip with message when library is not installed
        print("xgboost not installed: skip (pip install xgboost)")

    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = (
            LGBMClassifier(n_estimators = 300, class_weight = "balanced", subsample = 0.8,
                           subsample_freq = 1, colsample_bytree = 0.8,
                           random_state = random_state, verbose = -1),
            {"model__num_leaves": [7, 15],
             "model__learning_rate": [0.03, 0.1]},
        )
    except Exception:
        # Skip if not installed
        print("lightgbm not installed: skip (pip install lightgbm)")

    return models

# Tune

def tune(name, classifier, grid, X, y, groups):
    """Grid search model with patient grouped CV
    
    Returns best pipeline that has been refitted on full training set and summary of CV scores
    """

    pipe = Pipeline([("prep", make_preprocessor()), ("model", classifier)])
    cv = StratifiedGroupKFold(n_splits = n_folds, shuffle = True, random_state = random_state)

    search = GridSearchCV(pipe, grid, scoring = scoring, refit = "auroc", cv = cv, n_jobs = -1)
    search.fit(X, y, groups = groups)

    # Scores of best hyperparameter setting, averaged over five folds
    best = search.best_index_
    res = search.cv_results_
    row = {"model": name, "best_params": search.best_params_}
    for metric in scoring:
        mean, sd = res[f"mean_test_{metric}"][best], res[f"std_test_{metric}"][best]
        if metric == "brier":
            mean = -mean
        row[metric] = round(mean, 3)
        row[f"{metric}_sd"] = round(sd, 3)
    return search.best_estimator_, row

# Run

def main():
    warnings.filterwarnings("ignore", category = UserWarning)

    train = pd.read_pickle(train_path)
    X, y, groups = train["X"], train["y"], train["groups"]

    fs = train.get("feature_set", "A") # feature set built by preprocessing
    print(f"Feature set {fs}: {X.shape[1]} features")
    
    print(f"Training set: {len(y)} stays, {groups.nunique()} patients, ICU rate {y.mean():.1%}\n")

    model_dir.mkdir(parents = True, exist_ok = True)
    results_dir.mkdir(parents = True, exist_ok = True)

    rows = []
    for name, (classifier, grid) in candidate_models().items():
        print(f"Tuning {name}...")
        best_model, row = tune(name, classifier, grid, X, y, groups)
        joblib.dump(best_model, model_dir / f"{name}_{fs}.joblib")
        rows.append(row)
        print(f"    best: {row['best_params']}")
        print(f"    Cross-Validation AUROC {row['auroc']:.3f} (sd {row['auroc_sd']:.3f}) | "
              f"recall {row['recall']:.3f} | F1 {row['f1']:.3f} | Brier {row['brier']:.3f}\n")

    summary = pd.DataFrame(rows)
    summary.insert(0, "feature_set", fs)
    out_csv = results_dir / f"cv_results_{fs}.csv" # file per feature set to avoid writing over previous sets
    summary.to_csv(results_dir / out_csv, index = False)

    print("Cross-Validation summary (training set only and ESI baseline AUROC = 0.697):")
    print(summary[["model", "auroc", "auroc_sd", "recall", "precision", "f1", "brier"]]
          .to_string(index = False))
    print(f"\nModels saved to {model_dir}")
    print(f"Summary saved to {out_csv}")

if __name__ == "__main__":
    main()
