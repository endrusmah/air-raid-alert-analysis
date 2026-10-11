"""Fetch versioned public alerts and archived (never live-edited) news titles."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile
from email.utils import parsedate_to_datetime

import pandas as pd

# GKG extras (embedded metadata) can exceed Python's default CSV field limit.
csv.field_size_limit(10_000_000)

REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
REGIONS = ["Kyiv City", "Kyivska oblast", "Chernihivska oblast", "Sumska oblast",
           "Zhytomyrska oblast", "Poltavska oblast"]


def fetch(url: str) -> tuple[bytes, dict]:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ua-alerts-research/0.1"})
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read(), dict(response.headers)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("unreachable")


def alert_snapshot(commit: dict, folder: Path) -> dict:
    sha = commit["sha"]
    dest = folder / f"{sha}.json"
    proof = commit["_workflow_proof"]
    public_by = pd.Timestamp(proof["updated_at"])
    available_at = max(public_by, pd.Timestamp(commit["commit"]["committer"]["date"])) + pd.Timedelta(hours=6)
    if dest.exists():
        payload = json.loads(dest.read_text())
        payload.update(workflow_proof=proof, available_at=str(available_at))
        dest.write_text(json.dumps(payload, ensure_ascii=False))
        return payload
    url = f"https://raw.githubusercontent.com/{REPO}/{sha}/datasets/volunteer_data_en.csv"
    body, headers = fetch(url)
    # Input features use ONLY starts. Do not filter by naive/finished_at, because
    # both can be decided after an alert starts and would leak information.
    frame = pd.read_csv(io.BytesIO(body), usecols=["region", "started_at"])
    frame["started_at"] = pd.to_datetime(frame.started_at, utc=True)
    recorded_at = pd.Timestamp(commit["commit"]["committer"]["date"])
    # Conservative ingestion buffer after the public repository's commit time.
    cutoff = recorded_at.floor("D") - pd.Timedelta(days=70)
    frame = frame[(frame.started_at >= cutoff) & (frame.started_at < recorded_at)]
    max_start = frame.started_at.max()
    frame = frame[frame.region.isin(REGIONS)].drop_duplicates(["region", "started_at"])
    frame["day"] = frame.started_at.dt.floor("D")
    counts = frame.groupby(["day", "region"]).size()
    payload = {
        "commit": sha, "commit_at": str(recorded_at), "available_at": str(available_at),
        "workflow_proof": proof,
        "url": url, "raw_sha256": hashlib.sha256(body).hexdigest(),
        "raw_bytes": len(body), "max_start": str(max_start),
        "counts": {str(d): {r: int(counts.get((d, r), 0)) for r in REGIONS}
                   for d in pd.date_range(cutoff, recorded_at.floor("D"), freq="D")},
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False))
    return payload


def tag(extra: str, name: str) -> str | None:
    match = re.search(fr"<{name}>(.*?)</{name}>", extra, flags=re.S)
    return html.unescape(match.group(1)).strip() if match else None


def news_batch(day: pd.Timestamp, folder: Path) -> dict:
    # Fixed, label-independent sample: one 15-minute GKG batch at 00:00 UTC/day.
    # This is a sample of international reporting, NOT all news for the day.
    batch = day.strftime("%Y%m%d000000")
    dest = folder / f"{batch}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    url = f"https://data.gdeltproject.org/gdeltv2/{batch}.gkg.csv.zip"
    body, headers = fetch(url)
    modified = pd.Timestamp(parsedate_to_datetime(headers["Last-Modified"]))
    archive_at = max(day, modified) + pd.Timedelta(hours=2)
    rows, total, no_publication, no_title = [], 0, 0, 0
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        stream = io.TextIOWrapper(archive.open(archive.namelist()[0]), encoding="utf-8", errors="replace")
        for record in csv.reader(stream, delimiter="\t"):
            total += 1
            if len(record) != 27 or record[2] != "1":
                continue
            # GDELT uses the FIPS code UP, not the ISO code UA.
            if not any(len(x.split("#")) > 2 and x.split("#")[2] == "UP"
                       for x in record[9].split(";")):
                continue
            publication = tag(record[26], "PAGE_PRECISEPUBTIMESTAMP")
            title = tag(record[26], "PAGE_TITLE")
            if not publication:
                no_publication += 1
                continue
            if not title:
                no_title += 1
                continue
            try:
                published = pd.to_datetime(publication, format="%Y%m%d%H%M%S", utc=True)
                indexed = pd.to_datetime(record[1], format="%Y%m%d%H%M%S", utc=True)
                tone = float(record[15].split(",")[0])
            except (ValueError, TypeError):
                continue
            rows.append({"id": record[0], "url": record[4], "source": record[3],
                         "published_at": str(published), "indexed_at": str(indexed),
                         "available_at": str(max(archive_at, indexed + pd.Timedelta(hours=2))),
                         "title": title, "themes": record[7], "tone": tone})
    payload = {"url": url, "batch_at": str(day), "last_modified": str(modified),
               "available_at": str(archive_at), "raw_sha256": hashlib.sha256(body).hexdigest(),
               "raw_bytes": len(body), "all_records": total,
               "dropped_missing_publication": no_publication, "dropped_missing_title": no_title,
               "records": rows}
    dest.write_text(json.dumps(payload, ensure_ascii=False))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", default="2026-03-24")
    parser.add_argument("--as-of", default="2026-10-10T03:19:39Z")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    start, as_of = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.as_of)
    raw = args.data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    commits = []
    for page in range(1, 10):
        url = f"https://api.github.com/repos/{REPO}/commits?path=datasets/volunteer_data_en.csv&per_page=100&page={page}"
        cache = raw / f"commits_page_{page}.json"
        # --refresh is intentionally unnecessary: rerun with a new data directory
        # for a new vintage. The frozen project always reproduces the same run.
        if cache.exists():
            part = json.loads(cache.read_text())
        else:
            body, _ = fetch(url)
            part = json.loads(body)
            cache.write_bytes(body)
        commits.extend(c for c in part if start <= pd.Timestamp(c["commit"]["committer"]["date"]) < as_of)
        if not part or pd.Timestamp(part[-1]["commit"]["committer"]["date"]) < start:
            break
    snapshots = args.data_dir / "snapshots"
    news = args.data_dir / "news"
    snapshots.mkdir(exist_ok=True)
    news.mkdir(exist_ok=True)
    # Git commit dates alone do not establish public availability. Match each
    # output commit to a successful publisher workflow: its parent is the run's
    # head SHA, and creation of the output commit falls inside that run. GitHub's
    # server-recorded completion time supplies a conservative public-by bound.
    runs = []
    for page in range(1, 10):
        cache = raw / f"workflow_runs_{page}.json"
        if cache.exists():
            run_page = json.loads(cache.read_text())
        else:
            body, _ = fetch(f"https://api.github.com/repos/{REPO}/actions/runs?per_page=100&page={page}")
            run_page = json.loads(body)
            cache.write_bytes(body)
        part = run_page.get("workflow_runs", [])
        runs.extend(part)
        if not part or pd.Timestamp(part[-1]["created_at"]) < start:
            break
    for commit in commits:
        parents = [p["sha"] for p in commit["parents"]]
        stamp = pd.Timestamp(commit["commit"]["committer"]["date"])
        candidates = [r for r in runs if r["head_sha"] in parents and r["conclusion"] == "success"
                      and r["status"] == "completed" and r["name"] == "Update datasets"
                      and pd.Timestamp(r["created_at"]) <= stamp <= pd.Timestamp(r["updated_at"])]
        if not candidates:
            raise ValueError(f"No successful public publisher workflow for {commit['sha']}; fail closed")
        run = min(candidates, key=lambda r: r["updated_at"])
        commit["_workflow_proof"] = {k: run[k] for k in ["id", "head_sha", "created_at", "updated_at", "html_url", "path", "conclusion"]}
    jobs = [("alerts", c) for c in commits]
    last = min(as_of.floor("D") - pd.Timedelta(days=1),
               max(pd.Timestamp(c["commit"]["committer"]["date"]) for c in commits).floor("D"))
    jobs += [("news", d) for d in pd.date_range(start, last, freq="D")]
    errors = []
    def run(job):
        kind, item = job
        result = alert_snapshot(item, snapshots) if kind == "alerts" else news_batch(item, news)
        return kind, result.get("commit_at", result.get("batch_at"))
    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, job): job for job in jobs}
        for n, future in enumerate(cf.as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                kind, item = futures[future]
                errors.append({"kind": kind, "item": item if kind == "news" else item["sha"], "error": str(exc)})
                print("FAILED", errors[-1], flush=True)
            if n % 20 == 0 or n == len(jobs):
                print(f"Acquired {n}/{len(jobs)}; failures={len(errors)}", flush=True)
    (args.data_dir / "acquisition.json").write_text(json.dumps({
        "as_of": str(as_of), "start": str(start), "last_batch": str(last),
        "commits": len(commits), "errors": errors,
        "news_sampling": "00:00 UTC 15-minute GKG batch each day",
        "alert_buffer_hours": 6, "news_buffer_hours": 2,
        "alert_publication_proof": "successful GitHub publisher run, parent/head SHA and commit-inside-run match",
    }, indent=2, default=str))
    if errors:
        raise SystemExit("Acquisition incomplete; rerun to retry failed files. Do not evaluate silently.")


if __name__ == "__main__":
    main()
