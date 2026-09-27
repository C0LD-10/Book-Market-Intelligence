"""
Feature engineering.

The one non-trivial piece here is `KFoldTargetEncoder` for `primary_author`
and `primary_publisher`: both are high-cardinality (thousands of distinct
authors/publishers), so one-hot encoding would explode dimensionality and
overfit, and plain (non-cross-validated) mean-target-encoding leaks the
target into itself — a row's own label would contribute to its own encoded
feature. We instead:

  1. On the TRAIN split only, compute encodings via K-fold: for each fold,
     encode using the mean computed from the OTHER folds, so no row ever
     sees its own label.
  2. Fit a final encoding map on the full train split (for use at inference
     time on val/test/production data, which never contributed to it).
  3. Apply Bayesian smoothing toward the global mean, so authors/publishers
     seen only once or twice in training don't get a noisy 0%/100% encoding.

This is the standard, leakage-safe way to target-encode in a pipeline with
a proper train/val/test split (see Micci-Barreca, 2001, on Bayesian target
encoding for high-cardinality categoricals).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Leakage-safe mean target encoder with Bayesian smoothing.

    encoded_value = (n_category * category_mean + m * global_mean) / (n_category + m)

    `m` controls how many "virtual prior observations" of the global mean
    each category starts with — larger m means rare categories are pulled
    harder toward the global rate.
    """

    def __init__(self, columns: list[str], n_folds: int = 5, smoothing_m: float = 20.0, seed: int = 42):
        self.columns = columns
        self.n_folds = n_folds
        self.smoothing_m = smoothing_m
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean_ = float(np.mean(y))
        self.encoding_maps_ = {}
        y = pd.Series(np.asarray(y), index=X.index)
        for col in self.columns:
            stats = y.groupby(X[col]).agg(["mean", "count"])
            smoothed = (stats["count"] * stats["mean"] + self.smoothing_m * self.global_mean_) / (
                stats["count"] + self.smoothing_m
            )
            self.encoding_maps_[col] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            out_col = f"{col}_target_enc"
            X[out_col] = X[col].map(self.encoding_maps_[col]).fillna(self.global_mean_)
        return X[[f"{c}_target_enc" for c in self.columns]]

    def fit_transform_oof(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Out-of-fold transform for the TRAINING split only: each row is
        encoded using a mapping fit on the folds it does NOT belong to.
        Call this instead of fit_transform() when producing training
        features; call fit() once on the full train set afterward so
        transform() is well-defined for val/test.
        """
        y = pd.Series(np.asarray(y), index=X.index)
        oof = pd.DataFrame(index=X.index, columns=[f"{c}_target_enc" for c in self.columns], dtype=float)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for train_idx, holdout_idx in kf.split(X):
            fold_global_mean = float(y.iloc[train_idx].mean())
            for col in self.columns:
                stats = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).agg(["mean", "count"])
                smoothed = (
                    stats["count"] * stats["mean"] + self.smoothing_m * fold_global_mean
                ) / (stats["count"] + self.smoothing_m)
                mapping = smoothed.to_dict()
                oof.loc[X.index[holdout_idx], f"{col}_target_enc"] = (
                    X[col].iloc[holdout_idx].map(mapping).fillna(fold_global_mean).values
                )

        # Now fit the final (non-OOF) map on the whole train set for later use on val/test.
        self.fit(X, y)
        return oof


NUMERIC_FEATURES = [
    "author_count",
    "edition_count_log",
    "pages_log",
    "language_count",
    "book_age_years",
]
BINARY_FEATURES = ["is_multilingual", "has_isbn", "pages_missing"]
CATEGORICAL_FEATURES = ["genre", "primary_language_grouped", "book_length_category"]
TARGET_ENCODED_SOURCE_COLS = ["primary_author", "primary_publisher"]


def build_feature_matrix(
    df: pd.DataFrame, encoder: KFoldTargetEncoder, encoded_features: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the final modeling frame: numeric + binary + categorical
    (still as raw strings — the ColumnTransformer one-hot-encodes them) +
    the pre-computed target-encoded columns."""
    cols = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    X = df[cols].copy()
    X = pd.concat([X.reset_index(drop=True), encoded_features.reset_index(drop=True)], axis=1)
    return X


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer applied inside every model's sklearn Pipeline.
    Numeric (incl. target-encoded) columns are median-imputed + scaled;
    categoricals are one-hot encoded with unseen categories mapped to all-
    zero rows at inference time instead of raising."""
    numeric_and_encoded = NUMERIC_FEATURES + BINARY_FEATURES + [
        f"{c}_target_enc" for c in TARGET_ENCODED_SOURCE_COLS
    ]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_and_encoded),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


class BookEngagementPipeline(ClassifierMixin, BaseEstimator):
    """Single deployable artifact bundling every fitted preprocessing step
    with a final classifier, so `predict.py` and the Streamlit app only ever
    need to call `.predict_proba(cleaned_df)` on one joblib-loaded object —
    no risk of skew between training-time and inference-time feature code,
    because it IS the same code (this class), not a re-implementation.

    Expects `df` to have already been through `preprocessing.clean()` and
    `preprocessing.apply_language_grouping()` (i.e. exactly what
    data/processed/{train,val,test}.parquet contain), but NOT yet the target
    encoding or the ColumnTransformer — this object owns both.
    """

    def __init__(self, classifier, n_folds: int = 5, smoothing_m: float = 20.0, seed: int = 42):
        self.classifier = classifier
        self.n_folds = n_folds
        self.smoothing_m = smoothing_m
        self.seed = seed

    def fit(self, df: pd.DataFrame, y):
        self.target_encoder_ = KFoldTargetEncoder(
            columns=TARGET_ENCODED_SOURCE_COLS,
            n_folds=self.n_folds,
            smoothing_m=self.smoothing_m,
            seed=self.seed,
        )
        oof_encoded = self.target_encoder_.fit_transform_oof(df, y)
        X_train = build_feature_matrix(df, self.target_encoder_, oof_encoded)

        self.preprocessor_ = build_preprocessor()
        X_transformed = self.preprocessor_.fit_transform(X_train)

        self.classifier.fit(X_transformed, y)
        self.classes_ = getattr(self.classifier, "classes_", np.array([0, 1]))
        self.feature_names_ = self._get_output_feature_names()
        return self

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        encoded = self.target_encoder_.transform(df)
        X = build_feature_matrix(df, self.target_encoder_, encoded)
        return self.preprocessor_.transform(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self._transform_features(df))

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df)[:, 1] >= threshold).astype(int)

    def _get_output_feature_names(self) -> list[str]:
        num_names = NUMERIC_FEATURES + BINARY_FEATURES + [
            f"{c}_target_enc" for c in TARGET_ENCODED_SOURCE_COLS
        ]
        cat_encoder = self.preprocessor_.named_transformers_["cat"]
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        return num_names + cat_names
