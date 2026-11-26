"""Pre-processing: make the cohort ready for model training

Inputs: data/interim/cohort.pkl      (saved at end of notebooks/exploration.ipynb)
Outputs: data/processed/train.pkl    (features, outcome, and patients IDs for training)
        data/processed/test.pkl     (same for held-out test set)

Design notes:
- only information available at ED triage is used as feature
- steps that learn from data (filling missing values, scaling) not applied
- steps defined by make_preprocessor() and fitted later on training data only, inside cross validation -> no information leaks from test set
- train/test split by patient so no patient appears in both sets
"""

# Imports

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Settings

# locations of script and root
root = Path(__file__).resolve().parents[1] # project root folder
cohort_path = root / "data" / "interim" / "cohort.pkl"
out_dir = root / "data" / "processed"

# column predicted by model where 1 = yes, 0 = no
outcome = "icu_24h"

# Secondary feature sets

# A = baseline (vitals, age, sex...), B = A + chief complaint flags, C = B + derived vitals
feature_set = ("B")

# Ablation
# drop_esi = True -> removes ESI acuity columns: how much does model add beyond ESI
drop_esi = False

# chief complaint flags = 1 when free-text complaint matches pattern, otherwise = 0
# groups chosen for clinical reasons and checked only on training set
# patterns:
    # \b = word boundary eg "ich" for intracranial hemorrhage will not match "which"
    # | = "or" or other matched terms
    # partial words will match full words eg confus means confused, confusion

chief_complaint_flags = {
    "cc_ich":            r"\bich\b|head bleed|intracerebral|intraparenchymal",  # brain tissue bleed
    "cc_sdh_sah":        r"\bsdh\b|\bsah\b|subdural|subarachnoid",              # bleed around brain
    "cc_transfer":       r"transfer",                                           # from another hospital
    "cc_impact":         r"\bmvc\b|\bmva\b|ped struck|pedestrian|\bmca\b|motorcycle|bicycl|\bbike", # high energy accident like road crash
    "cc_altered_mental": r"altered mental|\bams\b|unresponsive|seizure|confus|letharg",
    "cc_fall":           r"fall",
}

random_state = 42 # fixed seed for identical splits every run

# continuous triage measurements: missing values will be filled and scaled
numeric = ["age", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain_score"]

# columns describing what occurs after triage -> never used as features!
leakage = ["hadm_id", "outtime", "disposition", "icu_intime", "hours_to_icu",
           "icu_any", "icu_24h", "first_vitals_mins"]

# Features

def build_features(cohort:pd.DataFrame) -> pd.DataFrame:
    """Create feature matrix X from triage variables
    
    Encoding categories as 0/1 does not learn from data,
    so can be completed before train/test split
    """

    # matches cohort row order
    X = pd.DataFrame(index = cohort.index)

    # continuous measurements (still contain missing values)
    X[numeric] = cohort[numeric]

    # binary flags
    X["male"] = (cohort["gender"] == "M").astype(int)
    X["triage_bypassed"] = cohort["triage_bypassed"].astype(int)   # whole triage form blank
    X["pain_unable"] = cohort["pain_unable"].astype(int)           # pain could not be assessed
    
    # ESI acuity as categories and Missing acuity is assumed to be informative
    # ie, linked to bypassed triage so it is its own category and becomes missingness indicator
    # ESI 3-5 are combined because ESI 4-5 are too rare to estimate separately
    esi = cohort["acuity"].map({1: "1", 2: "2", 3: "3plus", 4: "3plus", 5: "3plus"}).fillna("missing")
    # get_dummies -> turns categories into own 0/1 columns
    X = X.join(pd.get_dummies(esi, prefix = "esi", dtype = int))
    
    # arrival mode as categories eg ambulance, helicopter, walk in
    # missing treated as "unknown"
    arrival = (cohort["arrival_transport"].fillna("UNKNOWN")
                .str.lower().str.replace(" ", "_"))
    X = X.join(pd.get_dummies(arrival, prefix="arrival", dtype = int))

    # Feature set definitions
    # Feature set B - chief complaints

    if feature_set in ("B", "C"):
        complaint = cohort["chiefcomplaint"].fillna("").str.lower()
        for name, pattern in chief_complaint_flags.items():
            X[name] = complaint.str.contains(pattern, regex = True).astype(int)

    # Feature set C - vitals

    # standard clinical thresholds for vital signs
    # missing vital gives zero, but already accounted for by "triage_bypassed" when all missing
    # shock_index remains missing when either value is missing
    # and later filled with training median like other continuous columns
    if feature_set == "C":
        X["shock_index"] = cohort["heartrate"] / cohort["sbp"]          # > 0.9 -> shock
        X["hypoxia"] = (cohort["o2sat"] < 94).astype(int)               # low oxygen saturation
        X["hypotension"] = (cohort["sbp"] < 90).astype(int)             # low blood pressure
        X["hypertension"] = (cohort["sbp"] > 160).astype(int)           # raised BP (possible Cushing response where BP rises to keep blood from brain)
        X["tachycardia"] = (cohort["heartrate"] > 100).astype(int)      # fast heart rate
        X["age_65plus"] = (cohort["age"] >= 65).astype(int)             # older patients at higher risk
    
    return X

# Train/test split by patient

def split_by_patient(X, y, groups, test_folds=3, n_folds=10):
    """Return True/False mask marking test set (about 30% of stays)
 
    StratifiedGroupKFold divides patients into 10 folds but keeps ICU
    rate similar in each fold; 3 folds become the test set. All stays of a
    patient fall in the same fold, so no patient is in both sets
    """

    sgkf = StratifiedGroupKFold(n_splits = n_folds, shuffle = True, random_state = random_state)
    fold = np.zeros(len(y), dtype = int)
    k = 0
    for train_idx, test_idx in sgkf.split(X, y, groups):
        fold[test_idx] = k # held out in this fold
        k = k + 1
    return fold < test_folds

# Shock index - continuous columns for pre-processor

def numeric_columns(X):
    """Return continuous columns present in X

    shock_index will only exist for feature set C. Columns chosen
    from X so that the pre-processor always matches feature set
    saved, regardless of what feature_set is set to in script
    """

    cols = []
    for c in numeric + ["shock_index"]:
        if c in X.columns:
            cols.append(c)
    return cols

# Pre-processor (fitted later on training data only)
def make_preprocessor() -> ColumnTransformer:
    """Fill missing numeric values with training median and standardise
 
    Standardising rescales column to mean 0 and standard deviation 1,
    so features measured in varying units can be compared.
    Binary and category columns pass through unchanged. Used inside model
    pipelines in models.py, so it is re-fitted within every cross-validation fold
    but never sees data it is evaluated on.
    """

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy = "median")),
        ("scale", StandardScaler()),
    ])

    # numeric_columns becomes function not list so picks up present column
    return ColumnTransformer(
        [("numeric", numeric_steps, numeric_columns)],
        remainder = "passthrough",              # keep flags and categories as is
        verbose_feature_names_out = False,      # keep names without numeric__ prefix
    )

# Run

def main():
    # reset_index to give clean row
    cohort = pd.read_pickle(cohort_path).reset_index(drop=True)
    assert cohort["stay_id"].is_unique, "duplicate ED stays in cohort"

    # fs_tag to record which feature set is being used for saving separately
    X = build_features(cohort)
    fs_tag = feature_set
    if drop_esi:
        esi_cols = []
        for c in X.columns:
            if c.startswith("esi_"):
                esi_cols.append(c)
        X = X.drop(columns = esi_cols)
        fs_tag = feature_set + "_noesi"
    y = cohort[outcome].astype(int)
    groups = cohort["subject_id"] # ID keeps patient stays together
 
    # safety check: stop when post-triage information is in features
    leaked = []
    for c in leakage:
        if c in X.columns:
            leaked.append(c)
    assert len(leaked) == 0, "leakage columns in features: " + str(leaked)

    # save them as dictionaries with features, outcomes, IDs, tags
    is_test = split_by_patient(X, y, groups)
    train = {"X": X[~is_test], "y": y[~is_test], "groups": groups[~is_test], "feature_set": fs_tag}
    test = {"X": X[is_test], "y": y[is_test], "groups": groups[is_test], "feature_set": fs_tag}
    
    # checks: no patient in both sets
    overlap = set(train["groups"]) & set(test["groups"])
    assert not overlap, f"{len(overlap)} patients appear in both sets"

    # save both sets and create folder if non existing
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(train, out_dir / "train.pkl")
    pd.to_pickle(test, out_dir / "test.pkl")

    # summary with feature list, size, ICU rate, bypassed triage rate
    # when balanced, the train and test rates should be similar!
    print("Features (" + str(X.shape[1]) + "):", list(X.columns))
    print()
    for name, part in [("train", train), ("test", test)]:
        print(name, ":", len(part["y"]), "stays,", part["groups"].nunique(), "patients",
              "ICU rate", round(part["y"].mean() * 100, 1), "%,",
              "bypassed", round(part["X"]["triage_bypassed"].mean() * 100, 1), "%")
    print("\nPatients in both sets:", len(overlap))
    print("Saved to", out_dir)

    # for secondary feature sets: number of training stays for each flag and ICU rate
    # training data only
    if feature_set in ("B", "C"):
        Xtr, ytr = train["X"], train["y"]
        flags = list(chief_complaint_flags)

        if feature_set == "C":
            flags += ["hypoxia", "hypotension", "hypertension", "tachycardia", "age_65plus"]
    
        for name in flags:
            on = Xtr[name] == 1
            print(name, ":", on.sum(), "stays, ICU rate", round(ytr[on].mean() * 100, 1), "%")
        print()
 
if __name__ == "__main__":
    main()