# Frozen data and provenance

`acquisition.json` records the query cutoff, fixed sampling rule, availability
buffers, and acquisition failures (none in the delivered run).

`snapshots/*.json` are compact, commit-pinned extracts of the volunteer CSV:

- All six input regions' daily counts for a 70-day window.
- The original full CSV's source URL, byte size, and SHA-256 digest.
- Repository commit time and matched GitHub publisher workflow proof.
- Buffered `available_at`, used by the as-of feature builder.

Only `region` and `started_at` are read during extraction. Alert end times and
the `naive` indicator are intentionally absent from model inputs.

`news/*.json` contain Ukraine-related GKG records from one fixed 00:00 UTC
15-minute batch each day. Only records with explicit archived headline and
publication timestamp are retained. Each batch includes its source URL,
full ZIP's SHA-256 digest/size, HTTP Last-Modified, and record/filter counts.
Each record retains publisher URL/name, GKG ID, archived title/themes/tone,
publication time, GKG index time, and buffered availability time.

`raw/commits_page_*.json` and `raw/workflow_runs_*.json` preserve public
GitHub provenance responses. `raw/update_datasets.yml` is the upstream
publisher workflow at commit `2f115548a8bd0816c901dbf422082091b3e5c34c`.
Large original CSV/ZIP transfers are not distributed; compact inputs are
sufficient to reproduce every feature and prediction. Originals can be
downloaded using the stored URLs and checked against their digests.

Alert inputs are derived from the MIT-licensed Vadimkin dataset; its license
is in `UPSTREAM_ALERTS_LICENSE.txt`. GDELT metadata and original publisher
URLs are retained for attribution. Article bodies are not redistributed.

These are public research sources, not a verified official alert archive.
The latest snapshot only proves events recorded by the collector; a silent
collector cannot be distinguished perfectly from a genuinely quiet period.
