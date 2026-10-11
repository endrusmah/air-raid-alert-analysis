# Revision results

The v1 rolling mean was selected legitimately on validation, but its 28-day
smoothing looked nearly constant. Up to 354 inputs for 139 initial outcomes made
the learned alternative unnecessarily large. The mean shift from 1.85 to 3.75
in the available training/validation labels also exposed adaptation limits.
The frozen forecast command always uses the same default origin; rerunning it
without new data is not a new day's forecast.

The replacement is a compact log-link Poisson model with an optional recent-rate
offset. History/calendar inputs are capped at 10, optional news aggregates at 3,
and the unigram vocabulary at min(50, n_train//3). The selected model uses 63
features. No clipping or rounding is applied to its expected count.

Validation alone selected news + history, a seven-day recent-level offset,
alpha=1 and all eligible training rows. Its validation MAE is 2.307 versus 2.186
for the mean benchmark. The explicitly requested deployment pool comprises
Poisson models; averages remain reported benchmarks and were more accurate.

All 25 temporal/provenance/model tests passed before final scoring. Selection
has a separate stage, zero test rows, an integrity lock and a pipeline hash.
Future test snapshots/headlines cannot change it; estimator fitting independently
excludes future or unpublished labels. Final scoring did not change selection.

On September 11–October 8, the Poisson model predicts 3.021–4.663 (24 distinct
rounded values) with MAE 1.518. The average benchmark achieves 1.297. The change
fulfills the responsive positive count-model constraints; it does not demonstrate
better forecasting accuracy. Since v1 already scored this same historical
period, this is a reused-holdout evaluation; future unseen data is required for
an independent claim.
