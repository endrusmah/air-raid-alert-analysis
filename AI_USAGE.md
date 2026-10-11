# AI Usage

## Summary

AI was the primary engineering tool for this Python project. The assistant
researched public data sources, implemented acquisition and forecasting code,
wrote adversarial leakage tests, executed experiments, diagnosed model behavior,
and produced the README, notebook, reports, and reproducible project archive.
The human supplied the problem and constraints, reviewed the results, and
requested the major modeling revision after observing nearly constant forecasts.
This document records the work performed in the project conversation; it does
not imply an independent human audit or invent an agent delegation history.

The project forecasts Kyiv city's next-day air-alert start count for civilian
shelter staffing and disruption planning. Historical alert counts are combined
with archived news titles and metadata. Every feature and training label must
have been available before the forecast origin. AI assistance does not establish
operational reliability; the reported evaluation is evidence for a pet project,
not a validated warning system.

The completed revision uses the frozen data vintage of **2026-10-10 03:19:39
UTC**, with complete target days through October 8. It does not claim to have
refreshed the data when the modeling revision or this document was prepared.

## Delegated Tasks & Workflow

The human delegated implementation and verification to one AI assistant.
**No autonomous subagents were used.** The tasks below describe responsibilities
delegated to that assistant, rather than separate agents or fictitious reviewers.

| Task | AI contribution | Human direction or review |
| --- | --- | --- |
| Problem formulation | Defined the Kyiv city count target, forecast timing, and civilian planning use case. | Requested a real-world Ukraine alert forecasting project. |
| Data and provenance | Implemented versioned public alert acquisition, archived news extraction, source hashes, and availability evidence. | Required recent available data and text/media inputs with strict temporal validity. |
| Feature engineering | Built per-origin historical features and training-only text transformations. | Identified excessive dimensionality relative to roughly 150 training rows. |
| Modeling | Implemented baselines, the original Ridge alternative, then compact log-link Poisson candidates and a recent-level offset. | Challenged flat forecasts and requested Poisson regression without negative-value clipping. |
| Model selection | Implemented validation-only comparisons and separate locked selection/final-evaluation stages. | Required the test window to stay outside model selection. |
| Verification and delivery | Wrote and ran tests, inspected outputs, executed notebook Python cells, documented limitations, and packaged the project. | Reviewed the delivered behavior and requested this AI usage record. |

The workflow was iterative: inspect sources and their timestamps, establish the
temporal contract, implement a small baseline, evaluate chronologically, then
respond to review with a controlled revision. The revision retained the same
data vintage so changes could be compared without a simultaneous data refresh.
Validation experiments preceded the final revised evaluation. Test outcomes
were not used to choose the revised configuration.

The implementation is organized into acquisition, as-of feature construction,
model fitting, validation selection, final evaluation, forecasting, and
descriptive analysis modules under `src/ua_alerts/`. Recorded source and fit
audits make the assistant's assumptions inspectable rather than relying only
on its explanations.

## AI Mistakes Caught & Architectural Pivots

### Oversized model and nearly constant forecasts

The original learned alternative allowed **54 numeric features plus 300 text
features**, against **139 initially eligible training outcomes**. This was an
unnecessarily large feature space for the available sample. The user caught the
practical symptom: the winning forecast looked like a constant near 4.4.

Investigation showed that the winner was a **28-day rolling mean**, selected
legitimately on validation, whose long smoothing window naturally produced
slowly changing forecasts. The estimator was not stuck. Repeated calls using
the same frozen default origin also necessarily returned the same forecast.
The available label mean shifted from approximately 1.85 in initial training
to 3.75 in validation, further exposing adaptation limits.

The revision reduced the design to **10 history/calendar inputs**, **3 optional
news aggregates**, and at most **50 unigram TF-IDF terms**. The text cap is also
limited to `floor(n_train / 3)` within each eligible training window. Nearby
regions are aggregated, weekdays use cyclic coordinates, and constant numeric
features are removed using training data alone. The selected final model uses
63 features in total; 50 is the text cap, not the total feature count.

### Count support and adaptation

The original Ridge model could predict negative counts and clipped them to
zero. The replacement uses **Poisson regression with a log link**, yielding
positive expected counts without clipping or integer rounding. Direct Poisson
validation trials still struggled with the changing level, so the candidate
set also included a recent seven-day rate offset derived exclusively from
known historical counts. This learns multiplicative adjustments to a recent
level while retaining the Poisson likelihood.

The deployment pool was explicitly restricted to Poisson candidates to satisfy
the requested model change. Averages remain reported benchmarks. This is a
model-family constraint, not evidence that Poisson beats every alternative.

### Publication and availability assumptions

Using the latest corrected alert history for every past origin would expose
information that was unavailable at the time. Git commit timestamps alone
also do not establish public availability. The final pipeline reconstructs
each origin from eligible snapshots and corroborates alert publication with
successful GitHub publisher workflow records, plus a six-hour safety buffer.
It uses alert start times, not later-known end times or durations.

Live article bodies or current Telegram previews could contain later edits.
The project instead uses archived GDELT titles and metadata with explicit
publication timestamps. A record must satisfy both publication and batch
availability checks, including a two-hour buffer; unknown publication times
are excluded. The fixed daily news sample is not a complete news archive.

During acquisition, large GDELT fields exceeded Python's default CSV field-size
limit. The assistant raised the limit and retried the failed batches. Downloads
fail closed rather than silently treating acquisition errors as zero news.

### Selection isolation and holdout honesty

The revision physically separates selection from final scoring. Selection
constructs no test rows, excludes final revised truth, and uses only first
usable validation label vintages available before selection. It searches 36
compact Poisson configurations, then saves the chosen configuration with an
integrity hash and a pipeline hash. Final evaluation rejects a changed lock or
changed pipeline.

An important limitation remains: **version 1 already evaluated the same
September 11–October 8 historical test period**. Version 2 kept that period out
of its selection procedure, froze the choice, passed tests, and then scored
it. That procedural separation cannot make previously seen data genuinely
unseen. The result is a reused-holdout evaluation; an independent claim requires
fresh prospective data.

## Verification & Quality Assurance

### Temporal contract

For target day D, the forecast is issued at **12:00 UTC on D−1** and predicts
alert starts in **[D 00:00, D+1 00:00)**. Each historical feature row uses the
snapshot and news content available strictly before its own origin. A training
row is eligible only when its origin, target-day end, and label availability
all precede the fit origin. Scalers, constant-feature filtering, and vocabulary
are fitted only on those eligible rows. Final revised outcomes are reserved
for reporting, not fitting.

Validation covers **August 13–September 9, 2026**. Selection is locked at
**September 10, 12:00 UTC**. September 10's target is purged because its outcome
is not available at the first test origin. The test covers **September
11–October 8**. Daily test refits may use earlier test-day labels only after
they become available; the selected family and hyperparameters remain fixed.

### Executed checks

**All 25 tests passed before final revised scoring.** The suite covers:

- Future snapshots, alert counts, publication times, ingestion/index times,
  equal-origin timestamps, unknown publication times, and delayed labels.
- Actual frozen-source availability and server-recorded alert publication proof.
- Training-only vocabulary and feature budgets, exclusion of revised final truth,
  and invariance to changes in future labels or text.
- Positive Poisson predictions, response to a varying synthetic signal, numeric
  fallback for missing text, and correctness of the recent-level offset.
- Absence of test rows during selection, invariance to corrupted future test
  sources, and rejection of modified selection or pipeline hashes.

Tests reside in `tests/`. The pre-evaluation run is recorded in
`reports/tests_before_final.log`. The locked configuration is in
`reports/selection.json` and `reports/selection.sha256`; fit/source audits and
prediction files provide per-origin evidence.

The assistant also checked the forecast CLI's origin restrictions and positive
output, executed the notebook's Python cells with CPython, visually inspected
generated plots, and verified archive CRCs and packaged file hashes. Notebook
outputs were not produced by a Jupyter kernel, and no independent external
review is claimed. For this documentation-only change, existing reports and
test records were checked; the historical experiment was not rerun.

To reproduce the intended selection/test order after installing dependencies:

```bash
python -m ua_alerts.evaluate --stage select --data-dir data --out-dir reports
python -m unittest discover -s tests -v
python -m ua_alerts.evaluate --stage test --data-dir data --out-dir reports
```

### Results and limits

The locked choice is history + news Poisson, a recent seven-day offset,
regularization alpha 1, and all eligible past training rows.

| Model | Validation MAE | Historical test MAE | Historical test RMSE |
| --- | ---: | ---: | ---: |
| 28-day average benchmark | 2.186 | 1.297 | 1.513 |
| History Poisson | 2.350 | 1.472 | 1.793 |
| Selected history + news Poisson | 2.307 | 1.518 | 1.810 |

The selected Poisson forecasts span **3.021–4.663** starts/day, with 24 distinct
values at two decimal places across 28 days. The revision satisfies the
responsive, positive count-model requirements but **does not improve accuracy
over the average benchmark** on this period. The sampled news does not
demonstrate an accuracy benefit. These outcomes, rather than a favorable
AI-generated narrative, are the basis for assessing the project.

Evidence and further reproduction instructions are in `README.md`,
`reports/revision_report.md`, `reports/metrics.json`, and the test suite.
