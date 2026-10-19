""" Pre-processing: turn the cleaned TBI cohort into model-ready training

Input: data/interim/cohort.pkl      (saved at the end of notebooks/explorations/ipynb)
Output: data/processed/train.pkl    (features, outcome, and patientts IDs for training)
        data/processed/test.pkl     (same applies for held-out test set)

Design notes:
- only information available at ED triage is used as feature
- steps that learn from data (filling missing values, scaling) are not applied
- steps defined by make_preprocessor() and fitted later on training data only, inside cross validation, so no information leaks from test set
- train/test split is by patient so no patient appears in both sets
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Settings

root = Path(__file__).resolve().parents[1] # project root folder
cohort_path = root / "data" / "interim" / "cohort.pkl"
out_dir = root / "data" / "processed"

outcome = "icu_24h"
random_state = 42 # fixed seed for identical splits every run

# Continuous triage measurements: missing values will be filled and scaled
numeric = ["age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain_score"]

# Columns describing what occurs after triage -> never used as features
leakage = ["hadm_id", "outtime", "disposition", "icu_intime", "hours_to_icu",
           "icu_any", "icu_24h", "first_vitals_mins"]

# Features
def build_features(cohort:pd.DataFrame) -> pd.DataFrame:
    """Create feature matrix from triage time variables
    
    Encoding categories as 0 and 1 does not learn from data,
    so can be completed before train and test split"""

    X = pd.DataFrame(index = cohort.index)

    # Continuous measurements (still containing missing values)
    X[numeric] = cohort[numeric]

    # Binary flags
    X["male"] = (cohort["gender"] == "M").astype(int)
    X["triage_bypassed"] = cohort["triage_bypassed"].astype(int)   # whole triage form blank
    X["pain_unable"] = cohort["pain_unable"].astype(int)           # pain could not be assessed
    
    # ESI acuity as categories and Missing acuity is informative (bypassed triage),
    # so it is its own category
    # ESI 3-5 are combined because ESI 4-5 are too rare to estimate separately
        
    esi = cohort["acuity"].map({1: "1", 2: "2", 3: "3plus", 4: "3plus", 5: "3plus"}).fillna("missing")
    X = X.join(pd.get_dummies(esi, prefix = "esi", dtype = int))
    
    # Arrival mode as categories (missing treated as "unknown")
    arrival = (cohort["arrival_transport"].fillna("UNKNOWN")
                .str.lower().str.replace(" ", "_"))
    X = X.join(pd.get_dummies(arrival, prefix="arrival", dtype = int))

    return X

# Train/test split by patient
def split_by_patient(X, y, groups, test_folds=3, n_folds=10):
    """Return a True/False mask marking the test set (about 30% of stays).
 
    StratifiedGroupKFold divides patients into 10 folds while keeping the ICU
    rate similar in each fold; 3 folds become the test set. All stays of a
    patient fall in the same fold, so no patient is in both sets.
    """

    sgkf = StratifiedGroupKFold(n_splits = n_folds, shuffle = True, random_state = random_state)
    fold = np.empty(len(y), dtype = int)
    for k, (_, idx) in enumerate(sgkf.split(X, y, groups)):
        fold[idx] = k
    return fold < test_folds

# Pre-processor (fitted later on training data only)
def make_preprocessor() -> ColumnTransformer:
    """Fill missing numeric values with training median and standardise.
 
    Binary and category columns pass through unchanged. Used inside model
    pipelines in models.py, so it is re-fitted within every cross-validation fold.
    """

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy = "median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [("numeric", numeric_steps, numeric)],
        remainder = "passthrough",              # keep flags and categories as is
        verbose_feature_names_out = False,
    )

# Run
def main():
    cohort = pd.read_pickle(cohort_path).reset_index(drop=True)
    assert cohort["stay_id"].is_unique, "duplicate ED stays in cohort"
 
    X = build_features(cohort)
    y = cohort[outcome].astype(int)
    groups = cohort["subject_id"]
 
    # Safety check: no post-triage information among the features
    leaked = set(X.columns) & set(leakage)
    assert not leaked, f"leakage columns in features: {leaked}"
 
    is_test = split_by_patient(X, y, groups)
    train = {"X": X[~is_test], "y": y[~is_test], "groups": groups[~is_test]}
    test = {"X": X[is_test], "y": y[is_test], "groups": groups[is_test]}
 
    # Checks: no patient in both sets, similar ICU rates
    overlap = set(train["groups"]) & set(test["groups"])
    assert not overlap, f"{len(overlap)} patients appear in both sets"
 
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(train, out_dir / "train.pkl")
    pd.to_pickle(test, out_dir / "test.pkl")
 
    print(f"Features ({X.shape[1]}): {list(X.columns)}\n")
    print(f"{'':<8}{'stays':>7}{'patients':>10}{'ICU rate':>10}{'bypassed':>10}")
    for name, part in [("train", train), ("test", test)]:
        print(f"{name:<8}{len(part['y']):>7}{part['groups'].nunique():>10}"
              f"{part['y'].mean():>10.1%}{part['X']['triage_bypassed'].mean():>10.1%}")
    print("\nPatients in both sets:", len(overlap))
    print("Saved to", out_dir)
 
 
if __name__ == "__main__":
    main()

