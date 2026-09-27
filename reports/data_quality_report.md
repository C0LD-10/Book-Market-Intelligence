# Data Quality Report
**Dataset:** `global_book_market_intelligence_2026.csv` (14,828 rows × 17 columns)
**Prepared for:** Global Book Market Intelligence project
**Scope:** Every non-trivial cleaning decision made in `src/preprocessing.py`, with the evidence that motivated it.

---

## 1. Structural integrity

| Check | Result |
|---|---|
| Full-row duplicates | **0** |
| Duplicated `title` values | 826 rows (5.6%) |
| Duplicated on (`title`, `primary_author`, `first_publish_year`) | **0** |

**Interpretation:** 826 titles appear more than once, but never with the same
author *and* publication year — these are distinct catalog records for
different editions of the same work (e.g. multiple publisher records for
*Dracula*, one per republication). **No rows were dropped on the basis of
title repetition.** Removing them would have deleted genuine market signal:
republication frequency (`edition_count`, and the repeat-record pattern
itself) is one of the strongest predictors found later in this project.

## 2. Missing data

| Column | Missing | % | Handling |
|---|---:|---:|---|
| `rating_score` / `rating_count` | 9,874 | 66.6% | See §3 — unified into the target definition, not imputed |
| `pages` | 1,701 | 11.5% | Kept as-is; `pages_missing` indicator added; `pages_log` (median-filled) used by the model |
| `primary_language` | 755 | 5.1% | Filled with explicit category `"unk"` |
| `book_length_category` | 1,701 | 11.5% (identical rows to `pages`, by construction) | Filled with explicit category `"Unknown"` |
| `first_publish_year` | 128 | 0.9% | Kept as-is; `year_missing` indicator added; `book_age_years` (median-filled) used by the model |
| `primary_author` | 338 | 2.3% | Filled with `"Unknown Author"` |
| `primary_publisher` | 205 | 1.4% | Filled with `"Unknown Publisher"` |

**Policy, and why:** no column is silently mean/mode-imputed in a way that
erases the fact that a value was missing. Categoricals get an explicit
"unknown" level (so a one-hot encoder or target encoder treats "we don't
know the language" as its own learnable category, distinct from any real
language). Numerics get a same-scale imputed value used *only* by the model,
paired with a binary indicator column so the model can also learn a
"missingness effect" directly, and so any downstream audit can immediately
tell an observed value from an inferred one. This logic lives in
`src.preprocessing.clean()` and is unit-verified in
`notebooks/02_feature_engineering.ipynb`, §1.

## 3. Critical finding: two encodings of "no rating" (the reason the project's target is *not* `rating_score`)

Two structurally different groups both represent "nobody has rated this
book," but are encoded differently in the raw data:

| Group | n | `rating_score` | `rating_count` | `rating_tier` (raw) |
|---|---:|---|---|---|
| Missing | 9,874 | `NaN` | `NaN` | `"Unrated"` |
| Explicit zero | 237 | `0.0` | `0.0` | `"Unrated"` |

Both are labeled `"Unrated"` in the raw `rating_tier` column, confirming they
represent the same real-world fact through two different encodings — a
genuine data-quality defect. A model naively trained on raw `rating_score`
would treat the 237 explicit-zero rows as "the worst-rated books in the
catalog" (score 0.0 out of 5.0), when in fact they simply have no rating at
all, identical in meaning to the 9,874 `NaN` rows.

**Fix:** `src.preprocessing.build_target()` unifies both into a single
boolean, `is_rated = rating_count.fillna(0) > 0`. This is the single most
consequential line of preprocessing in the project — see
`reports/eda_report.md` §2 and `reports/model_card.md` for why it also
became the modeling target itself, rather than `rating_score`.

## 4. Outlier inspection (kept, not winsorized)

Every candidate "outlier" below was individually inspected and found to be a
**legitimate value**, not a data-entry error:

| Field | Extreme values | Example records | Verdict |
|---|---|---|---|
| `pages` | up to 6,500 | *Encyclopedia of World Biography* (6,500 pp.), *Modern Indian History* (6,160 pp.) | Legitimate reference works / anthologies |
| `edition_count` | up to 2,425 | *The Scarlet Letter* (2,425), *Emma* (2,263), *Dracula* (1,917) | Legitimate — heavily-republished public-domain classics |
| `first_publish_year` | as early as 1469 | *Naturalis Historia*, *Ecclesiastical History* | Legitimate classical/foundational texts |

**Decision:** none of these were removed or capped. Instead, `edition_count`
is log-transformed (`edition_count_log = log1p(edition_count)`) so a handful
of extreme republication counts don't dominate a linear model's scale, while
tree-based models simply split on the value directly — both approaches
preserve the information rather than discarding it.

## 5. Class balance of the modeling target

`is_rated` is `True` for **31.8%** of records (train: 31.81%, val: 31.79%,
test: 31.82% — near-identical rates across the stratified split, confirming
the split is working correctly). This is a moderate imbalance, handled via
`class_weight="balanced"` (logistic regression, random forest),
`scale_pos_weight` (XGBoost), and by using **PR-AUC (average precision)**,
not accuracy, as the primary evaluation metric (see `reports/model_card.md`).

## 6. Leakage audit

Three leakage risks were identified and explicitly guarded against:

1. **Target-derived columns** (`rating_score`, `rating_count`, `rating_tier`)
   are excluded from the feature set entirely (`config.yaml`,
   `features.excluded_leakage`) — they define the target, so using them as
   inputs would make the "prediction" trivial and useless.
2. **High-cardinality categorical leakage** (`primary_author`,
   `primary_publisher`): naive mean-target-encoding these would let a row's
   own label leak into its own feature (especially for authors/publishers
   appearing only once). Fixed with out-of-fold K-fold target encoding
   (`src.features.KFoldTargetEncoder`) — quantitatively demonstrated in
   `notebooks/02_feature_engineering.ipynb`, §4 (naive in-sample correlation
   with the label: 0.884; correct out-of-fold correlation: 0.290).
3. **Split-dependent transforms** (rare-language grouping): the set of
   "frequent enough to keep" languages is computed from the **training
   split only**, then applied unchanged to validation/test — verified in
   `notebooks/02_feature_engineering.ipynb`, §3.

---
*Generated as part of the Global Book Market Intelligence project. See
`reports/eda_report.md` for descriptive findings and `reports/model_card.md`
for modeling results.*
