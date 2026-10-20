"""Model development: train classifier on training set

Input: data/processed/train.pkl     created by src/preprocessing.py

Output: data/models/model.joblib    fitted models, used in src/eval.py
        results/cv_results.csv      cross validation performance
        
Run from project root folder

Design notes:
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
