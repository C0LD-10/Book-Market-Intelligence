# Executive Summary — Global Book Market Intelligence

## The question

Given only catalog metadata (genre, publisher, author, how many editions
and languages a title has been published in, length, publication year) —
**can we predict which titles are likely to gain reader engagement, and
which are likely to sit unread?**

## The answer

**Yes, with useful accuracy.** The model correctly ranks engaged vs.
unengaged titles 85.4% of the time (ROC-AUC 0.854) on titles it has never
seen, a substantial lift over guessing (which would score 50%). On a
harder, imbalance-aware metric (PR-AUC), it scores 0.746 vs. 0.318 for a
naive guess — more than double.

## What matters most

1. **Genre** — the single biggest lever. Holding everything else about a
   title fixed, Mystery/Romance/History titles are predicted to engage
   readers at roughly double the rate of Self-Help/Science Fiction titles
   *in this catalog*.
2. **Author and publisher track record** — titles from authors/publishers
   with a history of engaged readers are more likely to engage readers
   again (learned safely — no title's own outcome contaminates its own
   prediction).
3. **Reach** — how many editions and languages a title has been published
   in is strongly associated with engagement (more distribution → more
   chances to be read and rated).

## What this does **not** tell you

**This model cannot tell you whether a book is good.** We tested this
directly: predicting how *well* readers rated a book from the same
catalog metadata failed almost completely (near-zero predictive power).
Quality is a property of the content — the writing, the story — which
this dataset simply does not capture. This model answers "will this book
get noticed," not "is this book worth reading." Treat any comparison
between two specific titles as informative for market-reach questions
(marketing/distribution prioritization) — not as an editorial
recommendation.

## Practical use

- **Prioritization/triage at scale**, e.g. flagging a large backlist for
  which titles might benefit most from a translation or reprint push.
- **Not** a substitute for editorial judgment on any single title — the
  model is a statistical prior based on catalog-level patterns, and
  title-specific factors (a timely topic, an author's other work, a
  marketing event) it cannot see will and should override it.

## Where to go for more detail

- `reports/data_quality_report.md` — data cleaning decisions and evidence
- `reports/eda_report.md` — descriptive market findings
- `reports/model_card.md` — full technical methodology, performance, and limitations
- `notebooks/` — the analysis, reproducible end-to-end
- `app/app.py` — interactive explorer + predictor (`streamlit run app/app.py`)
