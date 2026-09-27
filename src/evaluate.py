"""
Final, one-shot evaluation on the held-out TEST split, plus interpretability.

TEST is used here for the first and only time in the whole pipeline — it
was untouched by candidate training, model selection, and hyperparameter
search (all of which used only TRAIN/VAL). This is what makes the numbers
in reports/model_card.md an honest estimate of real-world performance
rather than an optimistically-biased one.
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import load_config
from src.utils import get_logger

logger = get_logger(__name__)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold that maximizes F1 on the same split it's evaluated on is
    normally circular — here we only ever call this on VAL, never on TEST,
    and then apply the frozen threshold to TEST. That keeps TEST honest."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.nanargmax(f1s[:-1])  # last point has no matching threshold
    return float(thresholds[best_idx])


def evaluate_on_test(model, test_df: pd.DataFrame, target_col: str, threshold: float, figures_dir: str) -> dict:
    y_test = test_df[target_col].values
    proba = model.predict_proba(test_df)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "n_test": int(len(test_df)),
        "positive_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "f1_at_threshold": float(f1_score(y_test, preds)),
        "threshold_used": float(threshold),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }

    _plot_roc_pr(y_test, proba, figures_dir)
    _plot_calibration(y_test, proba, figures_dir)
    _plot_confusion_matrix(metrics["confusion_matrix"], figures_dir)

    logger.info("TEST metrics: %s", {k: v for k, v in metrics.items() if k != "confusion_matrix"})
    return metrics


def _plot_roc_pr(y_true, proba, figures_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve (test)")
    axes[0].legend(loc="lower right", frameon=False)

    precision, recall, _ = precision_recall_curve(y_true, proba)
    base_rate = float(np.mean(y_true))
    axes[1].plot(recall, precision, color="#059669", linewidth=2,
                 label=f"AP = {average_precision_score(y_true, proba):.3f}")
    axes[1].axhline(base_rate, color="#9ca3af", linestyle="--", linewidth=1, label=f"Base rate = {base_rate:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (test)")
    axes[1].legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(f"{figures_dir}roc_pr_curves.png", bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(y_true, proba, figures_dir: str) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(mean_pred, frac_pos, marker="o", color="#7c3aed", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve (test, decile bins)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{figures_dir}calibration_curve.png", bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(cm: list, figures_dir: str) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Unrated", "Rated"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{figures_dir}confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def permutation_importance_report(model, test_df: pd.DataFrame, target_col: str, seed: int, figures_dir: str) -> pd.DataFrame:
    """Model-agnostic feature importance: how much does shuffling one raw
    input column degrade held-out PR-AUC? Computed on TEST (post hoc,
    doesn't affect the model), on RAW input columns (not the one-hot-
    expanded internal features), so the result is directly interpretable
    by a non-ML stakeholder ("genre matters this much")."""
    from src.features import NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET_ENCODED_SOURCE_COLS

    raw_cols = [c for c in NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + TARGET_ENCODED_SOURCE_COLS
                if c in test_df.columns]
    y_test = test_df[target_col].values
    baseline_ap = average_precision_score(y_test, model.predict_proba(test_df)[:, 1])

    rng = np.random.RandomState(seed)
    n_repeats = 10
    importances = {}
    for col in raw_cols:
        drops = []
        for _ in range(n_repeats):
            shuffled = test_df.copy()
            shuffled[col] = rng.permutation(shuffled[col].values)
            shuffled_ap = average_precision_score(y_test, model.predict_proba(shuffled)[:, 1])
            drops.append(baseline_ap - shuffled_ap)
        importances[col] = {"mean_ap_drop": float(np.mean(drops)), "std_ap_drop": float(np.std(drops))}

    imp_df = pd.DataFrame(importances).T.sort_values("mean_ap_drop", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_df.index[::-1], imp_df["mean_ap_drop"][::-1],
            xerr=imp_df["std_ap_drop"][::-1], color="#2563eb")
    ax.set_xlabel("Drop in test PR-AUC when column is shuffled")
    ax.set_title("Permutation Feature Importance")
    fig.tight_layout()
    fig.savefig(f"{figures_dir}permutation_importance.png", bbox_inches="tight")
    plt.close(fig)

    return imp_df


def run_evaluation_pipeline(cfg: dict) -> dict:
    model = joblib.load(f"{cfg['paths']['model_dir']}engagement_model.joblib")
    val_df = pd.read_parquet(cfg["paths"]["val_data"])
    test_df = pd.read_parquet(cfg["paths"]["test_data"])
    target_col = cfg["target"]["name"]

    # Threshold chosen on VAL, frozen, then applied once to TEST.
    val_proba = model.predict_proba(val_df)[:, 1]
    threshold = find_best_threshold(val_df[target_col].values, val_proba)

    test_metrics = evaluate_on_test(model, test_df, target_col, threshold, cfg["paths"]["figures_dir"])
    imp_df = permutation_importance_report(
        model, test_df, target_col, cfg["project"]["random_seed"], cfg["paths"]["figures_dir"]
    )
    imp_df.to_csv(f"{cfg['paths']['model_dir']}feature_importance.csv")

    with open(f"{cfg['paths']['reports_dir']}test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("Top 5 features by permutation importance:\n%s", imp_df.head(5))
    return {"test_metrics": test_metrics, "feature_importance": imp_df}


if __name__ == "__main__":
    cfg = load_config()
    run_evaluation_pipeline(cfg)
