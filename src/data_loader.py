"""
Raw data loading and schema validation.

Keeping this as its own module (rather than inlining `pd.read_csv` everywhere)
means every entry point — notebooks, training script, app — validates the
same schema contract and fails loudly and early if the upstream data changes
shape, instead of failing silently three steps downstream.
"""
from __future__ import annotations

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = {
    "title": "object",
    "primary_author": "object",
    "author_count": "int64",
    "primary_publisher": "object",
    "edition_count": "int64",
    "first_publish_year": "float64",
    "pages": "float64",
    "rating_score": "float64",
    "rating_count": "float64",
    "primary_language": "object",
    "language_count": "int64",
    "has_isbn": "bool",
    "genre": "object",
    "publication_decade": "float64",
    "book_length_category": "object",
    "rating_tier": "object",
    "is_multilingual": "bool",
}


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV and validate that the schema matches expectations.

    Raises
    ------
    ValueError if a required column is missing. Dtype mismatches are logged
    as warnings (not raised) because pandas' inferred dtype can legitimately
    shift (e.g. int64 -> float64) if a fresh export introduces a single NaN
    in a previously-complete integer column — that's informative, not fatal.
    """
    df = pd.read_csv(path)

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw data is missing expected columns: {missing_cols}")

    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        logger.warning("Raw data has unexpected extra columns: %s", extra_cols)

    for col, expected_dtype in EXPECTED_COLUMNS.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            logger.warning(
                "Column '%s': expected dtype %s, got %s", col, expected_dtype, actual_dtype
            )

    logger.info("Loaded raw data: %d rows x %d columns from %s", *df.shape, path)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("/src", 1)[0])
    from src.config import load_config

    cfg = load_config()
    df = load_raw(cfg["paths"]["raw_data"])
    print(df.head())
    print(df.dtypes)
