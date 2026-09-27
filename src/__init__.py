"""
book-market-intelligence source package.

Modules:
    config          — load/validate config.yaml, resolve absolute paths
    utils            — logging, reproducibility, small shared helpers
    data_loader      — read raw CSV, basic schema checks
    preprocessing    — cleaning, missing-value policy, train/val/test split
    features         — feature engineering, target encoding, ColumnTransformer
    train            — model training + hyperparameter search
    evaluate         — metrics, calibration, interpretability (SHAP/permutation)
    predict          — load trained artifacts and score new records
"""

__version__ = "1.0.0"
