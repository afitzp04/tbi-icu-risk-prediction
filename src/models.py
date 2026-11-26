"""Model development: train and tune classifier on training set

Inputs:  data/processed/train.pkl                                   created by src/preprocessing.py
Outputs: data/models/<model>_<feature_set>_<search_type>.joblib    fitted models, used in src/eval.py
         results/cv_results_<feature_set>_<search_type>.csv         cross validation performance
        
Run from project root folder

Design notes:
- testing set not involved, only in eval.py for final evaluation
- each model is a pipeline: of preprocessor -> classifier
    - pipeline re-fitted for every CV fold so imputation medians and scaling are learned from training folds only
    - prevents leakage
- folds grouped by patient so one patient's visits never in both training and validation fold
- class weighting deals with the mild class imbalance of 44% ICU
    - SMOTE not required due to mildness!
-  Hyperparameters chosen by grid or randomised search to maximise AUROC
"""

# Imports

import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier
from scipy.stats import loguniform, randint

from preprocessing import random_state, make_preprocessor

# Paths

root = Path(__file__).resolve().parents[1]
train_path = root / "data" / "processed" / "train.pkl"
model_dir = root / "data" / "models"
results_dir = root / "results"

n_folds = 5

# Hyperparameter search

# "grid" tries combinations in small grids
# "random" samples n_iter combinations from wider randomised search
search_type = "random"
n_iter = 40 # random search

# Scores

# sklearn returns negative Brier (higher -> better) but flipped back later in tune()
scoring = {"auroc": "roc_auc", 
           "recall": "recall",
           "precision": "precision",
           "f1": "f1",
           "brier": "neg_brier_score"}

# Candidate models

def candidate_models():
    """Returns {name: (classifier, parameter grid)}
    
    Gridkeys start with model__ so classifier becomes step called "model" in pipeline
    Grids are small to avoid fitting noise
    """

    models = {
        # LR = interpretable baseline where C = inverse regularisation strength
        "logistic_regression": (
            LogisticRegression(class_weight = "balanced", max_iter = 2000, random_state = random_state),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        
        # RF: averages multiple decision trees to capture non-linear effects
        # and interactions between features
        "random_forest": (
            RandomForestClassifier(n_estimators = 300, class_weight = "balanced",
                                   random_state = random_state, n_jobs = -1),
            {"model__max_depth": [4, 6, 8, None],
             "model__min_samples_leaf": [5, 20]},
        ),
    }
    # XGBoost

    # gradient boosting to build trees one after another that correct previous one
    # class weighting (scale_pos_weight) not used in this mild imbalance scenario
    # skipped when not installed
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = (
            XGBClassifier(n_estimators = 300, subsample = 0.8, colsample_bytree = 0.8,
                          eval_metric = "logloss", random_state = random_state, n_jobs = -1),
            {"model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.03, 0.1]},
        )
    except Exception:
        print("xgboost not installed: skipped (pip install xgboost)")

    # LightGBM

    # also gradient boosting lib
    # should be faster than XGBoost
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
        print("lightgbm not installed: skipped (pip install lightgbm)")

    # Neural network

    # NN (multilayer perceptron): connected unit layers learn non linear patterns
    # small to match small dataset of approx 1800 stays
    # early_stopping holds training data back and stops when improvement stops
    # MLPClassifier has no class_weight option which is acceptable with this imbalance
    models["mlp"] = (
        MLPClassifier(max_iter = 1000, early_stopping = True, random_state = random_state),
        {"model__hidden_layer_sizes":[(16,), (32,), (32, 16)], # one/two hidden layers
         "model__alpha": [0.01, 0.1, 1.0, 10.0]}, # regularisation strength
    
    )


    return models

# Random distribution

def random_distributions():
    """Wider hyperparameter ranges, sampled at random n_iter times per model

    loguniform(a, b) samples evenly on log scale (good for values spanning
    several orders of magnitude). randint(a, b) samples whole numbers a to b-1.
    """
    return {
        "logistic_regression": {
            "model__C": loguniform(1e-3, 1e2),
            "model__penalty": ["l1", "l2"],            # l1 can shrink weak features to 0
            "model__solver": ["liblinear"],            # supports both l1 and l2
        },
        "random_forest": {
            "model__n_estimators": randint(200, 800),
            "model__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
            "model__min_samples_leaf": randint(1, 50),
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        "xgboost": {
            "model__n_estimators": randint(100, 800),
            "model__max_depth": randint(2, 7),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__min_child_weight": randint(1, 20),
            "model__reg_lambda": loguniform(0.1, 10),
        },
        "lightgbm": {
            "model__n_estimators": randint(100, 800),
            "model__num_leaves": randint(4, 32),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__min_child_samples": randint(5, 60),
            "model__reg_lambda": loguniform(0.1, 10),
        },
        "mlp": {
            "model__hidden_layer_sizes": [(8,), (16,), (32,), (64,), (16, 8), (32, 16)],
            "model__alpha": loguniform(1e-3, 1e2),
            "model__learning_rate_init": loguniform(1e-4, 1e-2),
        },
    }

# Tuning

def tune(name, classifier, grid, X, y, groups):
    """Tune model with patient grouped CV
    
    Returns best pipeline refitted on full training set
    and summary of CV scores
    """

    pipe = Pipeline([("prep", make_preprocessor()), ("model", classifier)])
    cv = StratifiedGroupKFold(n_splits = n_folds, shuffle = True, random_state = random_state)

    # refit = "auroc" -> best setting chosen by AUROC and refitted on all training data
    if search_type == "random":
        search = RandomizedSearchCV(pipe, random_distributions()[name], n_iter = n_iter,
                                    scoring = scoring, refit = "auroc", cv = cv, n_jobs = -1,
                                    random_state = random_state)
    else:
        search = GridSearchCV(pipe, grid, scoring = scoring, refit = "auroc", cv = cv, n_jobs = -1)

    # groups is passed so patient stays not split across folds
    search.fit(X, y, groups = groups)

    # mean and sd of scores for best setting, averaged over five folds
    best = search.best_index_
    results = search.cv_results_
    row = {"model": name, "best_params": search.best_params_}
    
    for metric in scoring:
        mean = results["mean_test_" +metric][best]
        sd = results["std_test_" + metric][best]
        if metric == "brier":
            mean = -mean    # flipped back
        row[metric] = round(mean, 3)
        row[metric + "_sd"] = round(sd, 3)
    return search.best_estimator_, row

# Run

def main():
    # sklearn warnings eg convergence hidden to keep reliable output
    warnings.filterwarnings("ignore", category = UserWarning)

    train = pd.read_pickle(train_path)
    X, y, groups = train["X"], train["y"], train["groups"]

    fs = train.get("feature_set", "A") # feature set built by preprocessing.py
    print("Feature set", fs, ":", X.shape[1], "features")
    
    print("Training set:", len(y), "stays", groups.nunique(), "patients, ICU rate", round(y.mean() * 100, 1), "%")

    model_dir.mkdir(parents = True, exist_ok = True)
    results_dir.mkdir(parents = True, exist_ok = True)

    # tune models and save best version with scores
    rows = []
    for name, (classifier, grid) in candidate_models().items():
        print("Tuning", name, "...")
        best_model, row = tune(name, classifier, grid, X, y, groups)
        joblib.dump(best_model, model_dir / f"{name}_{fs}_{search_type}.joblib")
        rows.append(row)
        print("    best:", row["best_params"])
        print("    Cross-Validation AUROC", row["auroc"], "(sd", row["auroc_sd"], ")",
              "|   recall", row["recall"], "|  F1", row["f1"], "|   Brier", row["brier"])
        print()

    # one results file per feature set and search type -> earlier runs not overwritten
    summary = pd.DataFrame(rows)
    summary.insert(0, "feature_set", fs)
    out_csv = results_dir / ("cv_results_" + fs + "_" + search_type + ".csv")
    summary.to_csv(out_csv, index = False)

    print("Cross-Validation summary (training set only and ESI baseline AUROC = 0.697):")
    print(summary[["model", "auroc", "auroc_sd", "recall", "precision", "f1", "brier"]]
          .to_string(index = False))
    print("\nModels saved to", model_dir)
    print("Summary saved to", out_csv)

if __name__ == "__main__":
    main()
