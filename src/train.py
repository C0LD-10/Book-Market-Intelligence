"""
Model training and selection.

Strategy:
  1. Fit four candidates on TRAIN only: a majority-class baseline (the floor
     any real model must beat), logistic regression (interpretable, linear
     reference), random forest, and XGBoost (with a light randomized
     hyperparameter search under 5-fold CV on train).
  2. Select the winner by VALIDATION average precision (PR-AUC) — chosen
     over accuracy/ROC-AUC as the primary metric because the positive class
     is a minority (~32%) and business cost is asymmetric: a false "will
     not engage" (missed opportunity) is different from a false "will
     engage" (wasted marketing spend) — PR-AUC is the standard threshold-
     free metric for imbalanced binary classification.
  3. Refit the winning configuration on TRAIN+VAL combined (more data for
     the artifact that actually gets deployed), and reserve TEST — which
     contributed to none of steps 1-3 — purely for the final, one-shot,
     unbiased performance report (see evaluate.py).
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config import load_config
from src.features import BookEngagementPipeline
from src.utils import get_logger, set_global_seed

logger = get_logger(__name__)


def _val_scores(model, val_df: pd.DataFrame, y_val: np.ndarray) -> dict:
    proba = model.predict_proba(val_df)[:, 1]
    return {
        "average_precision": float(average_precision_score(y_val, proba)),
        "roc_auc": float(roc_auc_score(y_val, proba)),
    }


def train_all_candidates(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Fit every candidate on train, score on val. Returns a dict of
    {name: {"model": fitted_pipeline, "val_scores": {...}}}."""
    seed = cfg["project"]["random_seed"]
    y_train = train_df[cfg["target"]["name"]].values
    y_val = val_df[cfg["target"]["name"]].values
    results = {}

    # --- 0. Baseline floor: predicts the training positive rate for everyone.
    baseline = DummyClassifier(strategy="stratified", random_state=seed)
    # Baseline needs no feature engineering; fit directly on a dummy 1-col frame.
    baseline.fit(np.zeros((len(train_df), 1)), y_train)
    baseline_proba = baseline.predict_proba(np.zeros((len(val_df), 1)))[:, 1]
    results["baseline_stratified"] = {
        "model": baseline,
        "val_scores": {
            "average_precision": float(average_precision_score(y_val, baseline_proba)),
            "roc_auc": float(roc_auc_score(y_val, baseline_proba)),
        },
    }
    logger.info("baseline_stratified: %s", results["baseline_stratified"]["val_scores"])

    # --- 1. Logistic Regression ---
    lr_cfg = cfg["models"]["logistic_regression"]
    lr = BookEngagementPipeline(
        classifier=LogisticRegression(
            C=lr_cfg["C"], class_weight=lr_cfg["class_weight"], max_iter=lr_cfg["max_iter"]
        ),
        n_folds=cfg["target_encoding"]["n_folds"],
        smoothing_m=cfg["target_encoding"]["smoothing_m"],
        seed=seed,
    )
    t0 = time.time()
    lr.fit(train_df, y_train)
    results["logistic_regression"] = {
        "model": lr,
        "val_scores": _val_scores(lr, val_df, y_val),
        "fit_seconds": round(time.time() - t0, 1),
    }
    logger.info("logistic_regression: %s", results["logistic_regression"]["val_scores"])

    # --- 2. Random Forest ---
    rf_cfg = cfg["models"]["random_forest"]
    rf = BookEngagementPipeline(
        classifier=RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            class_weight=rf_cfg["class_weight"],
            random_state=seed,
            n_jobs=-1,
        ),
        n_folds=cfg["target_encoding"]["n_folds"],
        smoothing_m=cfg["target_encoding"]["smoothing_m"],
        seed=seed,
    )
    t0 = time.time()
    rf.fit(train_df, y_train)
    results["random_forest"] = {
        "model": rf,
        "val_scores": _val_scores(rf, val_df, y_val),
        "fit_seconds": round(time.time() - t0, 1),
    }
    logger.info("random_forest: %s", results["random_forest"]["val_scores"])

    # --- 3. XGBoost with a light randomized hyperparameter search (5-fold CV on train) ---
    xgb_cfg = cfg["models"]["xgboost"]
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate  # counter class imbalance

    base_xgb = BookEngagementPipeline(
        classifier=XGBClassifier(
            eval_metric=xgb_cfg["eval_metric"],
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=-1,
        ),
        n_folds=cfg["target_encoding"]["n_folds"],
        smoothing_m=cfg["target_encoding"]["smoothing_m"],
        seed=seed,
    )
    param_dist = {
        "classifier__n_estimators": [200, 300, 400, 600],
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.02, 0.05, 0.08, 0.1],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "classifier__min_child_weight": [1, 3, 5, 10],
    }
    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=param_dist,
        n_iter=20,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cfg["evaluation"]["cv_folds"], shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=1,  # inner XGBoost already parallelizes; avoid oversubscription
        verbose=0,
    )
    t0 = time.time()
    search.fit(train_df, y_train)
    best_xgb = search.best_estimator_
    results["xgboost"] = {
        "model": best_xgb,
        "val_scores": _val_scores(best_xgb, val_df, y_val),
        "cv_best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "fit_seconds": round(time.time() - t0, 1),
    }
    logger.info("xgboost (tuned): %s | cv_best=%.4f", results["xgboost"]["val_scores"], search.best_score_)

    return results


def select_and_refit_best(
    cfg: dict, results: dict, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> tuple[str, BookEngagementPipeline, dict]:
    """Pick the candidate with the best validation PR-AUC among the
    non-baseline models, then refit that exact configuration on
    train+val combined for the final deployable artifact."""
    candidates = {k: v for k, v in results.items() if k != "baseline_stratified"}
    best_name = max(candidates, key=lambda k: candidates[k]["val_scores"]["average_precision"])
    logger.info("Selected model: %s", best_name)

    seed = cfg["project"]["random_seed"]
    target = cfg["target"]["name"]
    combined_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    y_combined = combined_df[target].values

    winner = candidates[best_name]["model"]
    # Clone the winning hyperparameters into a fresh pipeline and refit on more data.
    refit_pipeline = BookEngagementPipeline(
        classifier=winner.classifier.__class__(**winner.classifier.get_params()),
        n_folds=cfg["target_encoding"]["n_folds"],
        smoothing_m=cfg["target_encoding"]["smoothing_m"],
        seed=seed,
    )
    refit_pipeline.fit(combined_df, y_combined)

    return best_name, refit_pipeline, results[best_name]


def run_training_pipeline(cfg: dict) -> dict:
    set_global_seed(cfg["project"]["random_seed"])
    train_df = pd.read_parquet(cfg["paths"]["train_data"])
    val_df = pd.read_parquet(cfg["paths"]["val_data"])

    results = train_all_candidates(cfg, train_df, val_df)
    best_name, final_model, best_result = select_and_refit_best(cfg, results, train_df, val_df)

    model_path = f"{cfg['paths']['model_dir']}engagement_model.joblib"
    joblib.dump(final_model, model_path)
    logger.info("Saved final model (%s) to %s", best_name, model_path)

    leaderboard = {
        name: {k: v for k, v in r.items() if k != "model"} for name, r in results.items()
    }
    metadata = {
        "selected_model": best_name,
        "leaderboard": leaderboard,
        "target": cfg["target"]["name"],
        "random_seed": cfg["project"]["random_seed"],
    }
    with open(f"{cfg['paths']['model_dir']}model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {"model_path": model_path, "metadata": metadata, "final_model": final_model}


if __name__ == "__main__":
    cfg = load_config()
    run_training_pipeline(cfg)
