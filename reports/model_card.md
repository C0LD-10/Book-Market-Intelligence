# Model Card — Reader Engagement Predictor
**Model file:** `models/engagement_model.joblib` (XGBoost, selected by validation PR-AUC)
**Full training/evaluation code:** `src/train.py`, `src/evaluate.py` · **Notebooks:** `03_modeling.ipynb`, `04_interpretability_and_insights.ipynb`

---

## 1. Task definition

**Target:** `is_rated` = whether a catalog title has accumulated **any**
reader ratings (`rating_count > 0`), evaluated as binary classification.

**This is deliberately not a quality/rating-value model.** See §7 and
`reports/eda_report.md` §2 for why: catalog metadata has essentially no
recoverable signal for how *well* a book is rated (R² ≈ 0.008 in a
confirmatory experiment), but real, explainable signal for *whether it is
rated at all* — a legitimate market-reach question in its own right ("which
titles are likely to gain reader traction").

**Inputs used** (all available at/near catalog-entry time, none derived
from ratings): `genre`, `primary_language` (grouped), `book_length_category`,
`author_count`, `edition_count` (log), `pages` (log, + missing-indicator),
`language_count`, `book_age_years`, `is_multilingual`, `has_isbn`, and
K-fold target-encoded `primary_author` / `primary_publisher`.

**Inputs explicitly excluded:** `rating_score`, `rating_count`,
`rating_tier` (these define the target — including them would be leakage),
and `title` (free-text identifier, not generalizable metadata).

## 2. Data splits

70% train (n=10,379) / 15% validation (n=2,224) / 15% test (n=2,225),
stratified on `is_rated`. Positive rate is consistent across all three
splits (31.79–31.82%), and zero overlap was confirmed on a
(title, author, year) natural key (`notebooks/02_feature_engineering.ipynb`, §2).
**The test split was used exactly once**, after model selection was
finalized on train/validation — the numbers in §4 are an honest,
un-tuned estimate.

## 3. Candidates and selection

| Model | Val PR-AUC | Val ROC-AUC | Notes |
|---|---:|---:|---|
| Majority/stratified baseline | 0.319 | 0.502 | Floor — matches base rate by construction |
| Logistic Regression | 0.702 | 0.829 | `class_weight="balanced"`; recovers most of the achievable signal |
| Random Forest | 0.717 | 0.840 | 400 trees, `max_depth=12` |
| **XGBoost (tuned, selected)** | **0.722** | **0.844** | 20-iteration randomized search, 5-fold CV on train, `scoring="average_precision"` |

Selection metric: **validation average precision (PR-AUC)**, chosen over
accuracy/ROC-AUC because the positive class is a ~32% minority and the
business cost of missed-engagement vs. wasted-attention errors is
asymmetric — PR-AUC is the standard threshold-free metric for this setting.

The gap between logistic regression (0.702) and the tuned tree ensemble
(0.722) is modest — most of the signal is linear/additive (a genre effect,
a log-edition-count effect, a language-reach effect, each contributing
close to independently), and the winning model is best understood as a
believable refinement of that structure, not a discovery of some hidden
nonlinear pattern the linear model missed.

**Final artifact:** the winning XGBoost configuration was refit on
train+validation combined (12,603 rows) for more training data in the
deployed model; test performance below reflects that refit model.

## 4. Final test performance (held out, one-shot)

| Metric | Value |
|---|---:|
| n (test) | 2,225 |
| Positive rate | 31.8% |
| **ROC-AUC** | **0.854** |
| **PR-AUC (average precision)** | **0.746** |
| Brier score | 0.156 |
| F1 (at threshold tuned on validation, 0.624) | 0.643 |
| Precision / Recall at that threshold | 67.4% / 61.4% |

Confusion matrix (test, threshold=0.624):

| | Predicted Unrated | Predicted Rated |
|---|---:|---:|
| **Actual Unrated** | 1,307 | 210 |
| **Actual Rated** | 273 | 435 |

See `reports/figures/roc_pr_curves.png` and `confusion_matrix.png`.

**Calibration:** `reports/figures/calibration_curve.png` shows the model is
**mildly overconfident in the mid-probability range** (e.g. a predicted
~46% bucket observes ~31% actual engagement). Brier score of 0.156 is
reasonable but not excellent. **Recommendation:** treat raw probabilities
as a ranking/triage signal; apply Platt scaling or isotonic regression on a
calibration split before using probabilities for anything requiring
precise percentages (e.g. expected-value calculations).

## 5. What drives the predictions

Permutation importance (drop in test PR-AUC when a raw column is shuffled,
10 repeats) and SHAP (feature-level, on the transformed model input) agree:

| Feature | Permutation importance (ΔPR-AUC) |
|---|---:|
| `genre` | 0.103 |
| `primary_author` (target-encoded) | 0.076 |
| `edition_count_log` | 0.062 |
| `language_count` | 0.027 |
| `primary_publisher` (target-encoded) | 0.026 |
| `book_age_years` | 0.022 |
| (all others) | < 0.005 each |

A counterfactual sweep (`notebooks/04_interpretability_and_insights.ipynb`,
§2) holding every other feature fixed per-book shows genre alone shifts
average predicted engagement probability by roughly 2× between the
highest genres (Mystery, Romance, History) and lowest (Self-Help, Science
Fiction) — see `reports/figures/counterfactual_genre_effect.png`.

## 6. Error analysis

At the deployed threshold, false negatives (missed engagement) cluster in
Fiction, Children, and Romance — titles the feature set predicts as
lower-odds but which still engaged readers for reasons the model can't
see (a well-known author writing outside their usual genre, a marketing
event, word-of-mouth). False positives cluster in History, Mystery, and
Romance — genres the model favors that didn't pan out for a specific
title. **Practical implication: the model is a prior, not a verdict** —
useful for triage/prioritization at scale, not for overriding
title-specific editorial judgment.

## 7. What this model cannot do (explicit limitations)

1. **Cannot predict rating quality.** A same-methodology regression
   experiment on `rating_score` (the ~32% of records with real ratings)
   achieved R² ≈ 0.008 — essentially no signal. Predicting quality would
   require content-level features (review text, cover image, sample text,
   sales data) absent from this dataset.
2. **Correlational, not causal.** Genre/author/publisher effects are
   associations within one catalog snapshot. We cannot claim that
   *changing* a book's genre would change its engagement — only that,
   within this catalog, that association holds.
3. **English-centric.** 85.5% of records are English-language; language-
   specific effects for German/French/Spanish/other are estimated from
   much smaller samples (each < 2% of data) and are correspondingly noisier.
4. **Calibration is imperfect** (§4) — use ranked probabilities, or
   recalibrate, rather than trusting raw percentages at face value.
5. **No temporal validation.** The train/val/test split is a random
   stratified split, not a time-based split (e.g. train on pre-2015,
   test on 2015+). If this model were deployed to score *newly acquired*
   titles going forward, a time-based backtest is recommended before
   trusting it in production — random splitting can be optimistic when
   the true deployment setting is forecasting forward in time.

## 8. Reproducibility

- Random seed: 42 (fixed in `config.yaml`, applied via `src.utils.set_global_seed`)
- All hyperparameters and feature definitions: `config.yaml`
- Full pipeline: `python -m src.preprocessing && python -m src.train && python -m src.evaluate`
- Environment: see `requirements.txt` (exact versions pinned)

---
*See `reports/executive_summary.md` for a one-page, non-technical version
of this document.*
