"""Verify result directories before their numbers are used.

Usage:
    uv run python scripts/verify_campaign.py data/results-smoke/<run_dir> [...]
    uv run python scripts/verify_campaign.py data/results/paper/*rq1* --n 200
    uv run python scripts/verify_campaign.py <run_dir> --allow-dirty --only-failures

Prints one row per check and exits 1 when any check FAILs. WARN rows do not
fail the run; they mark conditions the paper must state (e.g. an unpinned
model snapshot such as Gemini, whose response.model carries no version).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from experiments.verify import FAIL, OK, WARN, verify_judged, verify_repeat_set, verify_run_dir

STYLE = {OK: "green", WARN: "yellow", FAIL: "bold red"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--n", type=int, default=None, help="Target rows per result file")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="Do not fail a run made from a dirty tree"
    )
    parser.add_argument("--only-failures", action="store_true", help="Hide OK rows")
    parser.add_argument(
        "--require-judge",
        metavar="TAG",
        default=None,
        help="Also require <file>_judged_<TAG>.jsonl with the same ids for every result file",
    )
    args = parser.parse_args()

    console = Console()
    all_checks = []
    for run_dir in args.run_dirs:
        all_checks.extend(verify_run_dir(run_dir, n=args.n, allow_dirty=args.allow_dirty))
        if args.require_judge:
            all_checks.extend(verify_judged(run_dir, args.require_judge))
    all_checks.extend(verify_repeat_set(args.run_dirs))

    table = Table(title="Campaign verification")
    table.add_column("run", style="cyan", overflow="fold")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for c in all_checks:
        if args.only_failures and c.status == OK:
            continue
        table.add_row(c.run, c.name, f"[{STYLE[c.status]}]{c.status}[/]", c.detail)
    console.print(table)

    n_fail = sum(1 for c in all_checks if c.status == FAIL)
    n_warn = sum(1 for c in all_checks if c.status == WARN)
    console.print(
        f"{len(all_checks)} checks over {len(args.run_dirs)} run(s): "
        f"[{STYLE[FAIL] if n_fail else 'green'}]{n_fail} FAIL[/], [yellow]{n_warn} WARN[/]"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
