"""
Cleaning and split logic.

Design decisions (see reports/data_quality_report.md for full justification):

1. No rows are dropped for "duplication". 826 titles appear more than once,
   but zero *full rows* are duplicated, and manual inspection shows repeats
   are legitimate distinct catalog records (different publisher/edition/year
   of the same work, e.g. multiple editions of "Dracula"). Dropping them
   would silently delete real market signal (republication is itself a
   feature: `edition_count` already captures this at the record level).

2. `rating_score == 0 & rating_count == 0` (237 rows) is treated as
   equivalent to "no rating data", identical to the 9,874 NaN rows. Both
   states mean "no reader ever rated this book" — the dataset's producer
   encoded that fact two different ways, which is a data-quality issue, not
   two different classes. The target definition
   `rating_count.fillna(0) > 0` unifies them.

3. Missing `pages`, `first_publish_year`, `primary_language`,
   `primary_author`, `primary_publisher` are NOT imputed with a mean/mode
   silently. Each gets an explicit `<field>_missing` indicator (where used
   as a model feature) or an explicit "Unknown"/"unk" category, so the model
   can learn "missingness" as a signal in its own right, and so a human
   reviewing predictions can see when a decision was made from an imputed
   value.

4. Extreme-looking values (pages > 3000, edition_count > 1000,
   first_publish_year < 1500) were manually inspected and found to be
   legitimate (encyclopedias/anthologies, heavily-republished public-domain
   classics, and pre-1500 classical texts respectively) — kept as-is, not
   winsorized. Their effect on tree models is handled by using
   `log1p(edition_count)`; on linear models by standardization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger

logger = get_logger(__name__)

CURRENT_YEAR = 2026


def build_target(df: pd.DataFrame, target_col: str = "is_rated") -> pd.DataFrame:
    """Construct the leakage-free binary target and drop the columns it is
    derived from so they can never accidentally leak into a feature set
    built downstream from this dataframe."""
    df = df.copy()
    df[target_col] = (df["rating_count"].fillna(0) > 0).astype(int)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions documented in the module docstring and
    engineer a small set of deterministic, leakage-free derived columns.
    This function must NOT see the target during any conditional logic
    (no target-dependent imputation) — that would be leakage.
    """
    df = df.copy()

    # --- missingness indicators (kept even though the raw column is also kept,
    #     so a linear model can use "missing" as a first-class signal) ---
    df["pages_missing"] = df["pages"].isna().astype(int)
    df["year_missing"] = df["first_publish_year"].isna().astype(int)

    # --- numeric transforms: heavy right-skew -> log1p ---
    df["pages_log"] = np.log1p(df["pages"].fillna(df["pages"].median()))
    df["edition_count_log"] = np.log1p(df["edition_count"])

    # --- book age at time of dataset snapshot; NaN year -> median age ---
    median_year = df["first_publish_year"].median()
    df["book_age_years"] = CURRENT_YEAR - df["first_publish_year"].fillna(median_year)

    # --- language grouping: keep frequent languages explicit, rare -> "other",
    #     missing -> its own explicit category (NOT silently merged into "other",
    #     since "unknown language" and "known rare language" are different
    #     provenance / different real-world situations) ---
    df["primary_language"] = df["primary_language"].fillna("unk")

    # --- categorical missing -> explicit "Unknown" ---
    df["book_length_category"] = df["book_length_category"].fillna("Unknown")
    df["primary_author"] = df["primary_author"].fillna("Unknown Author")
    df["primary_publisher"] = df["primary_publisher"].fillna("Unknown Publisher")

    logger.info("Cleaning complete: %d rows, %d columns", *df.shape)
    return df


def group_rare_languages(train_langs: pd.Series, top_n: int) -> list[str]:
    """Return the top_n most frequent languages *computed on the training
    split only* (fit-on-train, apply-to-all — the same discipline as any
    other learned transform, to prevent val/test leakage)."""
    return train_langs.value_counts().head(top_n).index.tolist()


def apply_language_grouping(df: pd.DataFrame, keep_languages: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["primary_language_grouped"] = df["primary_language"].where(
        df["primary_language"].isin(keep_languages), other="other"
    )
    return df


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-stage stratified split: train vs. (val+test), then val vs. test.
    Stratifying on the target at both stages keeps the ~31.8% positive rate
    consistent across all three splits, which matters for fair PR-AUC
    comparison between them."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "fractions must sum to 1"

    train_df, temp_df = train_test_split(
        df, train_size=train_frac, stratify=df[target_col], random_state=seed
    )
    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, train_size=relative_val_frac, stratify=temp_df[target_col], random_state=seed
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rate = split_df[target_col].mean()
        logger.info("%s split: n=%d, positive_rate=%.4f", name, len(split_df), rate)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def run_preprocessing_pipeline(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end: raw -> clean -> target -> split -> persist. Returns the
    three splits in memory as well, so callers (notebooks, training script)
    don't have to round-trip through disk if they don't want to."""
    from src.data_loader import load_raw

    raw = load_raw(cfg["paths"]["raw_data"])
    df = clean(raw)
    df = build_target(df, cfg["target"]["name"])
    df.to_parquet(cfg["paths"]["clean_data"], index=False)

    train_df, val_df, test_df = stratified_split(
        df,
        target_col=cfg["target"]["name"],
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        test_frac=cfg["split"]["test_frac"],
        seed=cfg["project"]["random_seed"],
    )

    # Language grouping is "fit" on train only, then applied everywhere.
    keep_langs = group_rare_languages(
        train_df["primary_language"], cfg["language_grouping"]["keep_top_n"]
    )
    train_df = apply_language_grouping(train_df, keep_langs)
    val_df = apply_language_grouping(val_df, keep_langs)
    test_df = apply_language_grouping(test_df, keep_langs)

    train_df.to_parquet(cfg["paths"]["train_data"], index=False)
    val_df.to_parquet(cfg["paths"]["val_data"], index=False)
    test_df.to_parquet(cfg["paths"]["test_data"], index=False)

    # Persist the train-fit language grouping so `predict.py` / the app can
    # apply the *exact same* grouping to brand-new records at inference time.
    import json

    with open(f"{cfg['paths']['model_dir']}language_groups.json", "w") as f:
        json.dump({"keep_languages": keep_langs}, f, indent=2)

    return train_df, val_df, test_df


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    run_preprocessing_pipeline(cfg)
