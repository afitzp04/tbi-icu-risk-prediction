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
