# 📚 Global Book Market Intelligence

Predicting **reader engagement** (not quality) from book catalog metadata —
a full, reproducible ML project: data cleaning → EDA → leakage-safe feature
engineering → model selection → honest held-out evaluation → interpretability
→ an interactive app.

**Start here:** `reports/executive_summary.md` (1 page, non-technical) or
`reports/model_card.md` (full technical write-up).

---

## The headline finding

The dataset's `rating_score` field looks like a "book quality" target, but
**it isn't modelable from catalog metadata** — a confirmatory experiment
found R² ≈ 0.008 (essentially zero). Instead, this project models
**`is_rated`**: whether a title has accumulated *any* reader ratings at all
— a genuine, well-signaled market-reach question (PR-AUC 0.746 / ROC-AUC
0.854 on a fully held-out test set, vs. 0.318 / 0.500 for a naive baseline).

That distinction — choosing the target the data can actually support, over
the target that would look most impressive — is the central methodological
decision in this project. See `reports/eda_report.md` §2 and
`reports/model_card.md` §7 for the full reasoning and evidence.

---

## Project structure

```
book-market-intelligence/
├── data/
│   ├── raw/                      # original CSV, untouched
│   └── processed/                 # cleaned + train/val/test parquet splits
├── notebooks/
│   ├── 01_eda.ipynb                          # data quality + descriptive findings
│   ├── 02_feature_engineering.ipynb          # cleaning/split/encoding, verified
│   ├── 03_modeling.ipynb                     # candidate models, selection, negative-result experiment
│   └── 04_interpretability_and_insights.ipynb # SHAP, counterfactuals, error analysis
├── src/
│   ├── config.py           # loads config.yaml, resolves paths
│   ├── utils.py             # logging, seeding
│   ├── data_loader.py       # raw CSV load + schema validation
│   ├── preprocessing.py     # cleaning, target construction, stratified split
│   ├── features.py          # K-fold target encoding, ColumnTransformer, BookEngagementPipeline
│   ├── train.py             # candidate training, hyperparameter search, selection
│   ├── evaluate.py          # held-out test metrics, calibration, permutation importance
│   └── predict.py           # inference on new records (used by the app)
├── models/
│   ├── engagement_model.joblib   # final trained pipeline (preprocessing + classifier, one artifact)
│   ├── model_metadata.json        # leaderboard, selected model, hyperparameters
│   ├── feature_importance.csv     # permutation importance table
│   └── language_groups.json       # train-fit language grouping (needed for inference)
├── app/
│   └── app.py                # Streamlit: market explorer + engagement predictor
├── reports/
│   ├── figures/                        # all generated plots (ROC/PR, calibration, SHAP, ...)
│   ├── data_quality_report.md          # every cleaning decision, with evidence
│   ├── eda_report.md                    # descriptive market findings
│   ├── model_card.md                    # methodology, performance, limitations
│   ├── executive_summary.md             # 1-page non-technical summary
│   └── test_metrics.json                # machine-readable final test metrics
├── README.md
├── requirements.txt
└── config.yaml               # single source of truth: paths, target, splits, hyperparameters
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing everything from scratch

```bash
# 1. Clean, build target, stratified split
python -m src.preprocessing

# 2. Train + select model (baseline, logistic regression, random forest, tuned XGBoost)
python -m src.train

# 3. Final held-out evaluation, plots, permutation importance
python -m src.evaluate

# 4. (optional) Score a single record from the command line
python -m src.predict
```

Each step reads exclusively from `config.yaml` and the outputs of the
previous step — no notebook or script hardcodes a path or hyperparameter
that isn't defined there.

## Interactive app

```bash
streamlit run app/app.py
```

Two tabs: a filterable **Market Explorer** (genre/decade/language trends
over the cleaned catalog) and an **Engagement Predictor** (score a
hypothetical or real title's metadata and see the driving factors).

## Notebooks

Notebooks are reports, not alternate implementations — every notebook
imports and calls the actual `src/` functions rather than re-deriving logic,
so what you read is exactly what runs in production. Run in order
(01 → 04); each is self-contained and can also be re-executed independently
against the artifacts already in `data/processed/` and `models/`.

## Key design decisions (see linked reports for full justification)

| Decision | Why | Detail |
|---|---|---|
| Target = `is_rated`, not `rating_score` | Metadata has no quality signal (R²≈0.008) but real reach signal (PR-AUC 0.746) | `reports/eda_report.md` §2, `reports/model_card.md` §7 |
| Unify `NaN` and explicit `(0,0)` ratings | Both raw-encode the same fact ("never rated") | `reports/data_quality_report.md` §3 |
| K-fold target encoding for author/publisher | Naive mean encoding leaks the label (in-sample r=0.88 vs. correct oof r=0.29) | `reports/data_quality_report.md` §6, `notebooks/02` §4 |
| PR-AUC as primary metric | ~32% positive rate; threshold-free, imbalance-appropriate | `reports/model_card.md` §3 |
| Single deployable pipeline object (`BookEngagementPipeline`) | Guarantees training-time and inference-time features can never skew apart | `src/features.py` |
| No temporal validation performed | Random split may be optimistic vs. true forward-deployment; flagged as future work | `reports/model_card.md` §7 |

## Limitations (full list in `reports/model_card.md` §7)

This model predicts reach, not quality; is correlational, not causal;
under-represents non-English titles; is imperfectly calibrated in the
mid-probability range; and has not been validated on a time-based
(forward-looking) split. None of these are hidden — each is documented
with the evidence that surfaced it.

## Suggested next steps

1. **Time-based backtest** — retrain on titles published/catalogued before
   a cutoff date, test on titles after it, to validate the model under
   realistic forward-deployment conditions rather than a random split.
2. **Probability recalibration** — Platt scaling or isotonic regression on
   a dedicated calibration split, given the mid-range overconfidence found
   in `reports/figures/calibration_curve.png`.
3. **Content features for a genuine quality model** — if review text, cover
   images, or sales data become available, a *separate* model targeting
   `rating_score` becomes worth attempting; this dataset alone cannot
   support it.
4. **Non-English-specific evaluation** — report per-language metrics once
   more non-English data is available, given the current sample-size
   imbalance (85.5% English).
