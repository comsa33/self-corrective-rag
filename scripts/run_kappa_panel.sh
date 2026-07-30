#!/bin/bash
# Score the human-labelled panel with one judge after another.
#
# The panel is the 200 items behind round 2's human labels, so every judge here
# can be compared against the same people on the same answers. That is what
# answers "is the reported Judge column an artifact of gpt-4.1-nano" — far more
# directly, and at a fiftieth of the cost, than re-scoring all 11,250 results.
#
# Judges are passed as arguments and run in sequence, so callers control how
# many run at once. Ollama Cloud allows three concurrent requests and the full
# re-scoring sweep already holds one, which leaves room for two lanes here.
#
# Usage:
#   scripts/run_kappa_panel.sh ollama_cloud/nemotron-3-nano:30b ollama_cloud/qwen3.5:397b
set -u
cd "$(dirname "$0")/.." || exit 1

PANEL="data/results-judge/kappa-panel/panel200.jsonl"
LOG_DIR="data/results-judge/kappa-panel/logs"
mkdir -p "$LOG_DIR"
[ -f "$PANEL" ] || { echo "panel not found: $PANEL" >&2; exit 1; }
[ $# -gt 0 ] || { echo "usage: $0 <judge-model> [judge-model ...]" >&2; exit 1; }

failed=()
for judge in "$@"; do
  slug=$(printf '%s' "$judge" | tr '/:.' '---')
  log="$LOG_DIR/$slug.log"
  echo "[$(date '+%F %T')] start $judge" | tee -a "$log"
  start=$(date +%s)

  if uv run python scripts/run_llm_judge.py "$PANEL" --judge-model "$judge" >> "$log" 2>&1; then
    status=ok
  else
    status="exit=$?"
    failed+=("$judge")
  fi
  elapsed=$(( $(date +%s) - start ))

  # A truncated response is silent — the verdict still gets written, just from a
  # cut-off completion. deepseek-v4-flash truncated 2 of its first 14 verdicts
  # at the default ceiling, and a three-item probe had passed it. At 200 items a
  # handful of bad verdicts moves kappa, so this count decides whether the judge
  # belongs in the panel at all.
  trunc=$(grep -c "truncated due to exceeding" "$log")
  echo "[$(date '+%F %T')] done $judge — $status, ${elapsed}s, truncations=$trunc" | tee -a "$log"
  [ "$trunc" -gt 0 ] && echo "  DROP $judge from the panel: $trunc truncated verdicts" | tee -a "$log"
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "PANEL COMPLETE"
