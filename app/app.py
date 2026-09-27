"""
Streamlit app: Book Market Intelligence Explorer.

Two tabs:
  1. Market Explorer  — descriptive, filterable view of the cleaned catalog
     (genre landscape, engagement rates, temporal trends). Read directly
     from data/processed/books_clean.parquet, no model involved.
  2. Engagement Predictor — interactive form wrapping src.predict.EngagementPredictor
     so a user can score a hypothetical/new title and see which factors
     drove the prediction (via the model's own feature importances).

Run locally with:
    cd book-market-intelligence
    pip install -r requirements.txt
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.predict import EngagementPredictor

st.set_page_config(page_title="Book Market Intelligence", page_icon="📚", layout="wide")


@st.cache_resource
def get_predictor():
    return EngagementPredictor()


@st.cache_data
def get_clean_data(_cfg):
    return pd.read_parquet(_cfg["paths"]["clean_data"])


@st.cache_data
def get_feature_importance(_cfg):
    return pd.read_csv(_cfg["paths"]["model_dir"] + "feature_importance.csv", index_col=0)


cfg = load_config()

st.title("📚 Global Book Market Intelligence")
st.caption(
    "Explore the catalog, and predict whether a title is likely to gain reader engagement "
    "(any ratings at all) based on its metadata."
)

tab_explore, tab_predict, tab_about = st.tabs(["📊 Market Explorer", "🔮 Engagement Predictor", "ℹ️ About this model"])

# =============================================================================
# TAB 1 — Market Explorer
# =============================================================================
with tab_explore:
    df = get_clean_data(cfg)

    col_filters, col_main = st.columns([1, 3])
    with col_filters:
        st.subheader("Filters")
        genres = st.multiselect("Genre", sorted(df["genre"].unique()), default=sorted(df["genre"].unique()))
        decade_min, decade_max = int(df["publication_decade"].min()), int(df["publication_decade"].max())
        decade_range = st.slider("Publication decade", decade_min, decade_max, (1950, decade_max), step=10)
        multilingual_only = st.checkbox("Multilingual editions only", value=False)

    filtered = df[df["genre"].isin(genres)]
    filtered = filtered[filtered["publication_decade"].between(*decade_range)]
    if multilingual_only:
        filtered = filtered[filtered["is_multilingual"]]

    with col_main:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{len(filtered):,}")
        c2.metric("Engagement rate", f"{filtered['is_rated'].mean():.1%}")
        c3.metric("Median editions", f"{filtered['edition_count'].median():.0f}")
        c4.metric("Genres shown", f"{filtered['genre'].nunique()}")

        st.subheader("Engagement rate by genre")
        genre_rate = filtered.groupby("genre")["is_rated"].mean().sort_values(ascending=False)
        st.bar_chart(genre_rate)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Records by publication decade")
            decade_counts = filtered["publication_decade"].value_counts().sort_index()
            st.bar_chart(decade_counts)
        with col_b:
            st.subheader("Top languages")
            lang_counts = filtered["primary_language"].value_counts().head(10)
            st.bar_chart(lang_counts)

    with st.expander("Show filtered data table"):
        st.dataframe(
            filtered[[
                "title", "primary_author", "primary_publisher", "genre",
                "first_publish_year", "edition_count", "is_rated",
            ]].reset_index(drop=True),
            use_container_width=True,
        )

# =============================================================================
# TAB 2 — Engagement Predictor
# =============================================================================
with tab_predict:
    st.subheader("Score a title")
    st.caption(
        "Enter a title's catalog metadata to estimate the probability it accumulates "
        "any reader ratings. This predicts *engagement/reach*, not quality — see the "
        "'About this model' tab for why."
    )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            title = st.text_input("Title", "Untitled Work")
            primary_author = st.text_input("Primary author", "Unknown Author")
            author_count = st.number_input("Author count", min_value=0, max_value=50, value=1)
            genre = st.selectbox(
                "Genre",
                ["Fiction", "Mystery", "Romance", "History", "Biography", "Self-Help", "Children", "Science Fiction"],
            )
        with c2:
            primary_publisher = st.text_input("Primary publisher", "Unknown Publisher")
            edition_count = st.number_input("Edition count", min_value=0, max_value=3000, value=3)
            pages = st.number_input("Pages", min_value=1, max_value=8000, value=300)
            book_length_category = st.selectbox(
                "Length category", ["Novella", "Short", "Standard", "Long", "Epic", "Unknown"]
            )
        with c3:
            first_publish_year = st.number_input("First publish year", min_value=1400, max_value=2026, value=2020)
            primary_language = st.text_input("Primary language (ISO code, e.g. eng)", "eng")
            language_count = st.number_input("Number of languages published in", min_value=1, max_value=40, value=1)
            has_isbn = st.checkbox("Has ISBN", value=True)
            is_multilingual = st.checkbox("Multilingual edition", value=(language_count > 1))

        submitted = st.form_submit_button("Predict engagement", type="primary")

    if submitted:
        predictor = get_predictor()
        record = {
            "title": title, "primary_author": primary_author, "author_count": author_count,
            "primary_publisher": primary_publisher, "edition_count": edition_count,
            "first_publish_year": first_publish_year, "pages": pages,
            "primary_language": primary_language, "language_count": language_count,
            "has_isbn": has_isbn, "genre": genre, "book_length_category": book_length_category,
            "is_multilingual": is_multilingual,
        }
        result = predictor.predict_one(record)

        st.divider()
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.metric("P(reader engagement)", f"{result['probability_rated']:.1%}")
            st.metric("Prediction", result["predicted_label"])
            st.caption(f"Decision threshold: {result['decision_threshold']:.3f} · Model: {result['model_used']}")
        with col_r2:
            st.info(
                "This estimates whether the title will accumulate ANY reader ratings, "
                "based on catalog metadata alone — genre, publisher/author track record, "
                "and republication/translation reach are the strongest drivers (see the "
                "'About this model' tab). It does **not** estimate how well the book will "
                "be reviewed."
            )

# =============================================================================
# TAB 3 — About / Model Card summary
# =============================================================================
with tab_about:
    st.subheader("What this model does and does not do")

    try:
        with open(cfg["paths"]["reports_dir"] + "test_metrics.json") as f:
            test_metrics = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("Test ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
        c2.metric("Test PR-AUC", f"{test_metrics['average_precision']:.3f}")
        c3.metric("Test base rate", f"{test_metrics['positive_rate_test']:.1%}")
    except FileNotFoundError:
        st.warning("Run `python -m src.evaluate` to generate test_metrics.json.")

    st.markdown(
        """
**Target:** `is_rated` — has this title accumulated any reader ratings at all
(a proxy for market reach), evaluated on a fully held-out test split never
used for training or model selection.

**What drives predictions** (see `reports/model_card.md` for full detail):
1. Genre — up to ~2x spread in predicted engagement rate
2. Author / publisher track record (leakage-safe K-fold target encoding)
3. Republication reach (`edition_count`)
4. Translation reach (`language_count`, `is_multilingual`)

**What this model deliberately does NOT predict:** reader *rating/quality*.
A confirmatory experiment (`notebooks/03_modeling.ipynb`, §4) found R² ≈ 0.008
when the same modeling approach is pointed at `rating_score` instead —
catalog metadata contains essentially no signal about how much readers will
*like* a book, only whether they engage with it at all. Predicting quality
would require content-level features (review text, cover, sample text) not
present in this dataset.

**Calibration caveat:** the model is mildly overconfident in the mid-probability
range (see `reports/figures/calibration_curve.png`) — treat predicted
probabilities as a ranking signal rather than a precise percentage.
        """
    )

    if os.path.exists(cfg["paths"]["figures_dir"] + "roc_pr_curves.png"):
        st.image(cfg["paths"]["figures_dir"] + "roc_pr_curves.png", caption="ROC / Precision-Recall (test set)")
