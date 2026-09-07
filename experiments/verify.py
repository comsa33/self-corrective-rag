"""Checks a result directory must pass before its numbers enter the paper.

Every check here corresponds to a way a past run was silently wrong:

  manifest / preflight     the March 2026 runs recorded no conditions at all
  snapshot                 a re-run on another model snapshot is not a re-run
  cache                    a warm response cache deflates latency to ms
  git dirty                the commit hash does not identify the code that ran
  n / unique ids           a resumed run can drop or duplicate questions
  errors                   an error row has an empty prediction and counts
  latency < 1 s share      cache contamination (see latency_analysis)
  required fields          a table needing a value that was never stored
  metered calls            usage that was not captured cannot be re-measured
  usage complete           a late callback would bill one question to the next
  provenance               a resumed row was made by another process

`verify_run_dir` returns the checks; `scripts/verify_campaign.py` prints
them as a table and exits non-zero when any FAIL remains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiments.analysis.latency_analysis import (
    CACHE_CONTAMINATION_LIMIT,
    CACHE_HIT_THRESHOLD_SECONDS,
)
from experiments.manifest import REQUIRED_ROW_FIELDS, RunManifest, load_manifest_or_none

FAIL = "FAIL"
WARN = "WARN"
OK = "OK"

# Keys `summary.json` must carry so a table can be rebuilt from disk alone.
REQUIRED_SUMMARY_SETTINGS = (
    "max_passages",
    "passage_cap",
    "enabled_tools",
    "models",
    "seed",
    "top_k",
)


@dataclass
class Check:
    run: str
    name: str
    status: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def _result_files(run_dir: Path) -> list[Path]:
    """The per-variant result files, without judge re-scorings or checkpoints."""
    return sorted(
        p for p in run_dir.glob("*.jsonl") if "_judged" not in p.name and "checkpoint" not in p.name
    )


def _read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def verify_run_dir(
    run_dir: str | Path,
    *,
    n: int | None = None,
    allow_dirty: bool = False,
) -> list[Check]:
    """Run every check on one result directory."""
    run_dir = Path(run_dir)
    run = run_dir.name
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str = "", *, warn: bool = False) -> None:
        status = OK if ok else (WARN if warn else FAIL)
        checks.append(Check(run, name, status, detail))

    if not run_dir.is_dir():
        add("run dir", False, f"not a directory: {run_dir}")
        return checks

    manifest = load_manifest_or_none(run_dir)
    add("manifest", manifest is not None, "" if manifest else "manifest.json missing or unreadable")
    if manifest is not None:
        checks.extend(_manifest_checks(run, manifest, allow_dirty))
    target_n = n if n is not None else (manifest.n if manifest else None)
    expected = _expected_snapshots(manifest)

    files = _result_files(run_dir)
    add("result files", bool(files), "" if files else "no *.jsonl result file")
    if manifest is not None:
        add(
            "variants complete",
            len(files) == len(manifest.variants),
            f"{len(files)} result files for {len(manifest.variants)} variants",
        )

    for path in files:
        checks.extend(_rows_checks(run, path, target_n, expected, manifest))
        checks.extend(_summary_checks(run, path))

    return checks


def verify_repeat_set(run_dirs: list[str | Path]) -> list[Check]:
    """Checks across several result directories that form repeated runs.

    Runs are grouped by config, dataset and model. Within a group the
    repeat indices must be unique and contiguous from 1 and every run must
    have the same n, or a mean over the repeats would be computed over an
    incomplete or uneven set.
    """
    groups: dict[str, list[tuple[str, RunManifest]]] = {}
    for run_dir in run_dirs:
        m = load_manifest_or_none(Path(run_dir))
        if m is None or not m.run_key:
            continue
        base = "|".join(m.run_key.split("|")[:3])  # stem|dataset|model
        groups.setdefault(base, []).append((Path(run_dir).name, m))

    out: list[Check] = []
    for base, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        label = f"repeat set {base}"
        ks = [m.repeat_index for _, m in members]
        missing_k = [name for name, m in members if m.repeat_index is None]
        out.append(
            Check(label, "repeat index present", FAIL if missing_k else OK, f"no k: {missing_k}")
        )
        present = sorted(k for k in ks if k is not None)
        unique = len(set(present)) == len(present)
        out.append(Check(label, "repeat index unique", OK if unique else FAIL, f"k={present}"))
        contiguous = present == list(range(1, len(present) + 1))
        out.append(
            Check(label, "repeat index contiguous", OK if contiguous else FAIL, f"k={present}")
        )
        ns = sorted({m.n for _, m in members})
        out.append(Check(label, "same n", OK if len(ns) == 1 else FAIL, f"n={ns}"))
    return out


def _manifest_checks(run: str, m: RunManifest, allow_dirty: bool) -> list[Check]:
    out: list[Check] = []

    def add(name, ok, detail="", *, warn=False):
        out.append(Check(run, name, OK if ok else (WARN if warn else FAIL), detail))

    add("preflight", m.preflight.get("status") == "ok", f"status={m.preflight.get('status')}")
    add("run finished", m.finished_at is not None, "finished_at missing: run did not complete")

    unpinned = [r.spec for r in m.models if not r.expected_response_model]
    add(
        "snapshot pinned",
        not unpinned,
        f"unpinned: {sorted(set(unpinned))}" if unpinned else "",
        warn=True,
    )
    for r in m.models:
        if r.expected_response_model:
            add(
                f"snapshot {r.slot}",
                r.observed_response_model == r.expected_response_model,
                f"observed={r.observed_response_model!r} expected={r.expected_response_model!r}",
            )
    expected = _expected_snapshots(m)
    if expected and m.observed_response_models:
        stray = sorted(set(m.observed_response_models) - expected)
        add("snapshot during run", not stray, f"unexpected response.model: {stray}")

    add(
        "cache disabled",
        bool(m.cache.get("llm_cache_disabled")),
        "DISABLE_LLM_CACHE was off: latency is not measurable",
    )
    add(
        "git clean",
        allow_dirty or not m.git.get("dirty"),
        f"commit {str(m.git.get('commit'))[:10]} with uncommitted changes: "
        f"{m.git.get('modified_files')}",
    )
    return out


def _expected_snapshots(m: RunManifest | None) -> set[str]:
    if m is None:
        return set()
    return {r.expected_response_model for r in m.models if r.expected_response_model}


def _rows_checks(
    run: str,
    path: Path,
    n: int | None,
    expected: set[str],
    manifest: RunManifest | None,
) -> list[Check]:
    out: list[Check] = []
    tag = path.stem

    def add(name, ok, detail="", *, warn=False):
        out.append(Check(run, f"{tag}: {name}", OK if ok else (WARN if warn else FAIL), detail))

    rows = _read_rows(path)
    valid = [r for r in rows if "error" not in r]
    n_err = len(rows) - len(valid)

    if n is None:
        add("n", True, f"{len(rows)} rows (no target given)", warn=True)
    else:
        add("n", len(rows) == n, f"{len(rows)} rows, target {n}")
    ids = [str(r.get("id")) for r in rows]
    add("unique ids", len(set(ids)) == len(ids), f"{len(set(ids))} unique of {len(ids)}")
    add("errors", n_err == 0, f"{n_err} error row(s)")

    missing = sorted({f for r in valid for f in REQUIRED_ROW_FIELDS if f not in r})
    add("required fields", not missing, f"missing in some rows: {missing}")

    latencies = [r["latency_seconds"] for r in valid if "latency_seconds" in r]
    if latencies:
        fast = sum(1 for x in latencies if x < CACHE_HIT_THRESHOLD_SECONDS)
        share = fast / len(latencies)
        add(
            "latency < 1s share",
            share <= CACHE_CONTAMINATION_LIMIT,
            f"{fast}/{len(latencies)} = {share:.0%} (limit {CACHE_CONTAMINATION_LIMIT:.0%})",
        )

    unmetered = sum(1 for r in valid if not r.get("metered_calls"))
    add("usage metered", unmetered == 0, f"{unmetered} row(s) with no metered call")
    incomplete = sum(1 for r in valid if r.get("usage_complete") is False)
    add("usage complete", incomplete == 0, f"{incomplete} row(s) with unreported calls")

    if expected:
        stray = sorted(
            {m for r in valid for m in r.get("response_models", []) if m not in expected}
        )
        add("row snapshots", not stray, f"unexpected response.model in rows: {stray}")

    if manifest is not None:
        out.extend(_provenance_checks(run, tag, valid, manifest))
        out.extend(_passage_cap_checks(run, tag, path, valid, manifest))
    return out


def _variant_of(path: Path) -> str | None:
    """Variant name of a result file, from its summary.json."""
    summary_path = path.with_name(f"{path.stem}_summary.json")
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8")).get("extra", {}).get("variant")
    except json.JSONDecodeError:
        return None


def _passage_cap_checks(
    run: str, tag: str, path: Path, rows: list[dict], m: RunManifest
) -> list[Check]:
    """Every row must respect the passage cap the manifest declares for its variant.

    This is the mechanical basis of the paper's "controlled for M" scope: a
    capped pipeline may never hand the generator more than M passages, and
    an uncapped one must have used everything it retrieved.
    """
    out: list[Check] = []

    def add(name, ok, detail="", *, warn=False):
        out.append(Check(run, f"{tag}: {name}", OK if ok else (WARN if warn else FAIL), detail))

    variant = _variant_of(path)
    if variant is None or variant not in m.max_passages_by_pipeline:
        add("passage cap", False, f"no cap declared for variant {variant!r}", warn=True)
        return out
    cap = m.max_passages_by_pipeline[variant]
    used = [r.get("passages_used") for r in rows if r.get("passages_used") is not None]
    if not used:
        return out
    if cap is None:
        truncated = sum(
            1 for r in rows if r.get("passages_used") != r.get("total_passages_retrieved")
        )
        add("passage cap", truncated == 0, f"uncapped, but {truncated} row(s) used < retrieved")
    else:
        over = sum(1 for u in used if u > cap)
        add("passage cap", over == 0, f"cap {cap}: {over} row(s) over, max used {max(used)}")
        add("passage cap reached", max(used) >= cap, f"cap {cap} never reached", warn=True)
    return out


def _provenance_checks(run: str, tag: str, rows: list[dict], m: RunManifest) -> list[Check]:
    """Rows reused from a checkpoint must have been made under this manifest's code.

    A resumed run carries rows written by an earlier process. They are
    acceptable only if that process ran the same commit with the cache off
    and got answers from the pinned snapshot (checked with the other rows);
    otherwise the manifest would vouch for conditions it did not produce.
    """
    out: list[Check] = []

    def add(name, ok, detail="", *, warn=False):
        out.append(Check(run, f"{tag}: {name}", OK if ok else (WARN if warn else FAIL), detail))

    unstamped = sum(1 for r in rows if not r.get("run_id") or not r.get("git_commit"))
    add("provenance", unstamped == 0, f"{unstamped} row(s) without run_id/git_commit stamp")

    cached = sum(1 for r in rows if r.get("llm_cache_disabled") is False)
    add("row cache", cached == 0, f"{cached} row(s) made with the response cache on")

    foreign = [r for r in rows if r.get("run_id") and r.get("run_id") != m.run_id]
    if not foreign:
        return out
    sources = sorted({str(r.get("run_id")) for r in foreign})
    other_commit = sorted(
        {str(r.get("git_commit")) for r in foreign if r.get("git_commit") != m.git.get("commit")}
    )
    add(
        "resumed rows commit",
        not other_commit,
        f"{len(foreign)} resumed row(s) from {sources}; commits differ from manifest: "
        f"{[c[:10] for c in other_commit]}",
    )
    if not other_commit:
        add("resumed rows", False, f"{len(foreign)} row(s) resumed from {sources}", warn=True)
    return out


def _summary_checks(run: str, path: Path) -> list[Check]:
    out: list[Check] = []
    tag = path.stem
    summary_path = path.with_name(f"{tag}_summary.json")
    if not summary_path.exists():
        out.append(Check(run, f"{tag}: summary", FAIL, f"{summary_path.name} missing"))
        return out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    has_metrics = bool(summary.get("metrics"))
    settings_used = summary.get("settings") or {}
    missing = [k for k in REQUIRED_SUMMARY_SETTINGS if k not in settings_used]
    ok = has_metrics and not missing
    detail = (
        "" if ok else f"metrics={'ok' if has_metrics else 'missing'} settings missing={missing}"
    )
    out.append(Check(run, f"{tag}: summary", OK if ok else FAIL, detail))
    return out
