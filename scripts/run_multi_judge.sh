#!/bin/bash
# Re-score every RQ1 result directory with one judge model.
#
# The paper reports LLM-as-Judge as one of three headline metrics, so the
# pipeline ranking has to survive a change of judge. Each judge runs as its own
# process over the same directories; the per-file outputs are tagged with the
# judge model, so the runs never collide and can be compared afterwards.
#
# Judging is checkpointed per item, so a directory that already finished costs
# no API calls and an interrupted run resumes where it stopped. Ollama's session
# quota resets on its own clock and will cut a long run short — rerun this
# script afterwards and it picks up the remainder.
#
# Usage:
#   scripts/run_multi_judge.sh ollama_cloud/minimax-m3
#
# Start all three:
#   for m in ollama_cloud/minimax-m3 ollama_cloud/deepseek-v4-flash ollama_cloud/glm-5.2; do
#     nohup scripts/run_multi_judge.sh "$m" > /dev/null 2>&1 & sleep 1
#   done
#
# Exits non-zero naming the directories that failed, so an incomplete sweep is
# never mistaken for a finished one.
set -u

cd "$(dirname "$0")/.." || exit 1

JUDGE="${1:-}"
[ -z "$JUDGE" ] && { echo "usage: $0 <judge-model>" >&2; exit 1; }

# Slug for log filenames only — the result files get their own tag, including a
# digest, from run_llm_judge.py.
SLUG=$(printf '%s' "$JUDGE" | tr '/:.' '---')
LOG_DIR="data/results-judge/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] [$SLUG] $*" | tee -a "$LOG_DIR/worker_$SLUG.log"; }

# Gemini and gpt-5-mini first: these are the runs behind Table 3's Judge column,
# so they are what a reviewer's question would actually be about. gpt-oss is the
# newer supporting material and can afford to finish later.
TARGETS=()
for d in data/results/paper/*rq1_*/; do [ -d "$d" ] && TARGETS+=("$d"); done
for d in data/results-openweight/final/*/; do [ -d "$d" ] && TARGETS+=("$d"); done

[ ${#TARGETS[@]} -eq 0 ] && { log "no result directories found"; exit 1; }

failed=()
log "starting — judge=$JUDGE over ${#TARGETS[@]} directories"
for d in "${TARGETS[@]}"; do
  name=$(basename "$d")
  log "dir $name"
  if uv run python scripts/run_llm_judge.py "$d" --judge-model "$JUDGE" \
      >> "$LOG_DIR/${SLUG}_${name}.log" 2>&1; then
    log "dir $name complete"
  else
    rc=$?
    log "dir $name FAILED (exit=$rc) — see $LOG_DIR/${SLUG}_${name}.log"
    failed+=("$name")
  fi
done

# A truncated response is silent: the run still reports success while the judge
# saw a cut-off prompt. glm-5.2 lost a full dataset to exactly this, so count the
# warnings rather than trusting the exit code.
trunc=$(cat "$LOG_DIR/${SLUG}_"*.log 2>/dev/null | grep -c "truncated due to exceeding max_tokens")
log "truncation warnings: $trunc"
[ "$trunc" -gt 0 ] && log "WARNING — $trunc truncated responses; check LLM_REASONING_MODELS for $JUDGE"

if [ ${#failed[@]} -gt 0 ]; then
  log "SWEEP INCOMPLETE — failed: ${failed[*]}"
  exit 1
fi

log "ALL DIRECTORIES COMPLETE"
