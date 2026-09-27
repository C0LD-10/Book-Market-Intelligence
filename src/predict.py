"""
Inference on new records.

This is the only module the Streamlit app (app/app.py) needs to import.
It takes care of applying the *exact* same cleaning + language-grouping
logic used at training time, so a user typing in one book's metadata gets
a prediction from features built identically to how the model was trained.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from src.config import load_config
from src.preprocessing import apply_language_grouping, clean
from src.utils import get_logger

logger = get_logger(__name__)

RAW_INPUT_COLUMNS = [
    "title", "primary_author", "author_count", "primary_publisher", "edition_count",
    "first_publish_year", "pages", "primary_language", "language_count", "has_isbn",
    "genre", "book_length_category", "is_multilingual",
]


class EngagementPredictor:
    """Thin, cached wrapper around the saved BookEngagementPipeline.

    Usage
    -----
    >>> predictor = EngagementPredictor()
    >>> predictor.predict_one({
    ...     "title": "Example", "primary_author": "Jane Doe", "author_count": 1,
    ...     "primary_publisher": "Acme Press", "edition_count": 3,
    ...     "first_publish_year": 2015, "pages": 320, "primary_language": "eng",
    ...     "language_count": 1, "has_isbn": True, "genre": "Fiction",
    ...     "book_length_category": "Standard", "is_multilingual": False,
    ... })
    {'probability_rated': 0.41, 'predicted_label': 'Unrated', ...}
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        model_path = f"{self.cfg['paths']['model_dir']}engagement_model.joblib"
        self.model = joblib.load(model_path)

        with open(f"{self.cfg['paths']['model_dir']}language_groups.json") as f:
            self.keep_languages = json.load(f)["keep_languages"]

        with open(f"{self.cfg['paths']['model_dir']}model_metadata.json") as f:
            self.metadata = json.load(f)

        with open(f"{self.cfg['paths']['reports_dir']}test_metrics.json") as f:
            self.test_metrics = json.load(f)
        self.threshold = self.test_metrics["threshold_used"]

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(RAW_INPUT_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")
        df = clean(df)
        df = apply_language_grouping(df, self.keep_languages)
        return df

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare(df)
        proba = self.model.predict_proba(prepared)[:, 1]
        out = df.copy()
        out["probability_rated"] = proba
        out["predicted_label"] = pd.Series(proba >= self.threshold).map(
            {True: "Likely to be rated", False: "Likely to stay unrated"}
        )
        return out

    def predict_one(self, record: dict) -> dict:
        df = pd.DataFrame([record])
        result = self.predict_batch(df).iloc[0]
        return {
            "probability_rated": round(float(result["probability_rated"]), 4),
            "predicted_label": result["predicted_label"],
            "decision_threshold": self.threshold,
            "model_used": self.metadata["selected_model"],
        }


if __name__ == "__main__":
    predictor = EngagementPredictor()
    example = {
        "title": "Example Title", "primary_author": "Unknown Author", "author_count": 1,
        "primary_publisher": "Unknown Publisher", "edition_count": 3,
        "first_publish_year": 2015, "pages": 320, "primary_language": "eng",
        "language_count": 1, "has_isbn": True, "genre": "Fiction",
        "book_length_category": "Standard", "is_multilingual": False,
    }
    print(predictor.predict_one(example))
