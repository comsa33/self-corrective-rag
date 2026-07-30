#!/bin/bash
# Run (or resume) the open-weight RQ1 sweep on Ollama Cloud for one worker.
#
# Every dataset is listed in order. Checkpoints are keyed on question id, so a
# dataset that already finished costs no API calls — the runner reads its
# checkpoint, sees every question is done, and moves on. That makes this script
# safe to run repeatedly, and safe to run again after an unclean interrupt such
# as closing a laptop mid-request.
#
# Pipelines differ by an order of magnitude in cost per question (CRAG issues
# ~40 LLM calls, Naive issues 1), so they are split across three workers to run
# concurrently within the provider's 3-request limit.
#
# Usage:
#   scripts/run_openweight.sh crag      # CRAG Replica          (slowest)
#   scripts/run_openweight.sh agentic   # Agentic (ReAct)
#   scripts/run_openweight.sh rest      # Naive + Single + Loop
#
# Start all three:
#   for w in crag agentic rest; do nohup scripts/run_openweight.sh $w & sleep 1; done
#
# Exits non-zero if any dataset failed, so a supervising process can tell an
# incomplete sweep from a finished one.
set -u

cd "$(dirname "$0")/.." || exit 1

# Honour a caller-supplied results directory; logs follow it so that results,
# checkpoints and logs never end up in different trees.
export RESULTS_DIR="${RESULTS_DIR:-data/results-openweight}"
export LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-default}"
                                      # "low" starves gpt-oss of the reasoning
                                      # it needs to complete a JSON schema
export DISABLE_LLM_CACHE="${DISABLE_LLM_CACHE:-true}"
                                      # required for latency to mean anything
export PREPROCESS_MODEL="${PREPROCESS_MODEL:-ollama_cloud/gpt-oss:120b}"
export EVALUATE_MODEL="${EVALUATE_MODEL:-ollama_cloud/gpt-oss:120b}"
export GENERATE_MODEL="${GENERATE_MODEL:-ollama_cloud/gpt-oss:120b}"
export AGENT_MODEL="${AGENT_MODEL:-ollama_cloud/gpt-oss:120b}"

LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

WORKER="${1:-}"
case "$WORKER" in
  crag)    VARIANT_ARGS=(--variants "CRAG Replica") ;;
  agentic) VARIANT_ARGS=(--variants "Agentic (ReAct)") ;;
  rest)    VARIANT_ARGS=(--variants "Naive RAG" "Single-Pass" "Loop Refinement") ;;
  *) echo "usage: $0 {crag|agentic|rest}" >&2; exit 1 ;;
esac

log() { echo "[$(date '+%F %T')] [$WORKER] $*" | tee -a "$LOG_DIR/worker_$WORKER.log"; }

failed_datasets=()

log "starting"
# FinanceBench is 150 questions in total, not a sample of a larger pool.
for spec in "2wikimultihopqa 200" "hotpotqa 200" "musique 200" "financebench 150"; do
  set -- $spec
  ds="$1"; n="$2"
  log "dataset $ds (n=$n)"
  if uv run python experiments/run.py \
      --config configs/experiment/rq1.yaml \
      --dataset "$ds" --sample "$n" \
      "${VARIANT_ARGS[@]}" >> "$LOG_DIR/${ds}_${WORKER}.log" 2>&1; then
    log "dataset $ds complete"
  else
    rc=$?
    log "dataset $ds FAILED (exit=$rc) — see $LOG_DIR/${ds}_${WORKER}.log"
    failed_datasets+=("$ds")
  fi
done

if [ ${#failed_datasets[@]} -gt 0 ]; then
  log "SWEEP INCOMPLETE — failed: ${failed_datasets[*]}"
  exit 1
fi

log "ALL DATASETS COMPLETE"
