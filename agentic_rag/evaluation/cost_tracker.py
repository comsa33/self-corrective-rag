"""API cost and latency tracker for experiment reproducibility.

Tracks per-call and aggregate statistics for:
  - LLM API costs (input/output tokens × price)
  - Embedding API costs
  - Latency per pipeline stage
  - Total experiment cost/time

Results are auto-logged for inclusion in the paper.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger
from loguru import logger

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens, as of 2024-12)
# Update these when models/pricing change.
# ---------------------------------------------------------------------------
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate USD cost for a single API call."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        # Try prefix match
        for key, val in MODEL_PRICING.items():
            if model.startswith(key):
                pricing = val
                break
    if pricing is None:
        logger.warning(f"No pricing for model '{model}', using gpt-4o-mini rates")
        pricing = MODEL_PRICING["gpt-4o-mini"]

    cost = (
        input_tokens * pricing["input"] / 1_000_000 + output_tokens * pricing["output"] / 1_000_000
    )
    return cost


# ---------------------------------------------------------------------------
# Call record
# ---------------------------------------------------------------------------
@dataclass
class APICallRecord:
    """Record of a single API call."""

    timestamp: float
    model: str
    stage: str  # e.g., "preprocess", "evaluate", "generate", "agent", "embed"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "stage": self.stage,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------
class CostTracker:
    """Track API costs and latency across an experiment run."""

    def __init__(self):
        self.records: list[APICallRecord] = []
        self._stage_timers: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(
        self,
        model: str,
        stage: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        **metadata,
    ) -> APICallRecord:
        """Record a single API call."""
        total = input_tokens + output_tokens
        cost = _estimate_cost(model, input_tokens, output_tokens)

        rec = APICallRecord(
            timestamp=time.time(),
            model=model,
            stage=stage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            latency_ms=latency_ms,
            cost_usd=cost,
            metadata=metadata,
        )
        self.records.append(rec)
        return rec

    @contextmanager
    def track_stage(self, stage: str):
        """Context manager to track latency of a pipeline stage."""
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000
        self._stage_timers[stage] = self._stage_timers.get(stage, 0.0) + elapsed

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return sum(r.total_tokens for r in self.records)

    @property
    def total_calls(self) -> int:
        """Total number of API calls."""
        return len(self.records)

    def summary(self) -> dict:
        """Generate a summary of costs and latency."""
        by_stage: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "tokens": 0, "cost": 0.0, "latency_ms": 0.0}
        )
        by_model: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})

        for r in self.records:
            by_stage[r.stage]["calls"] += 1
            by_stage[r.stage]["tokens"] += r.total_tokens
            by_stage[r.stage]["cost"] += r.cost_usd
            by_stage[r.stage]["latency_ms"] += r.latency_ms

            by_model[r.model]["calls"] += 1
            by_model[r.model]["tokens"] += r.total_tokens
            by_model[r.model]["cost"] += r.cost_usd

        return {
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "by_stage": dict(by_stage),
            "by_model": dict(by_model),
            "stage_timers_ms": dict(self._stage_timers),
        }

    def print_summary(self) -> None:
        """Print a formatted cost summary."""
        s = self.summary()
        print("\n" + "=" * 60)
        print("EXPERIMENT COST SUMMARY")
        print("=" * 60)
        print(f"Total Cost:   ${s['total_cost_usd']:.4f}")
        print(f"Total Tokens: {s['total_tokens']:,}")
        print(f"Total Calls:  {s['total_calls']}")
        print()

        print("By Stage:")
        for stage, data in s["by_stage"].items():
            print(
                f"  {stage:20s} | "
                f"calls={data['calls']:3d} | "
                f"tokens={data['tokens']:8,} | "
                f"${data['cost']:.4f} | "
                f"{data['latency_ms']:.0f}ms"
            )

        print()
        print("By Model:")
        for model, data in s["by_model"].items():
            print(
                f"  {model:30s} | "
                f"calls={data['calls']:3d} | "
                f"tokens={data['tokens']:8,} | "
                f"${data['cost']:.4f}"
            )

        if s["stage_timers_ms"]:
            print()
            print("Stage Latency:")
            for stage, ms in s["stage_timers_ms"].items():
                print(f"  {stage:20s} | {ms:.0f}ms")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Save all records and summary to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "records": [r.to_dict() for r in self.records],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Cost report saved to {path}")

    def load(self, path: Path) -> None:
        """Load records from a saved JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.records = [
            APICallRecord(
                timestamp=r["timestamp"],
                model=r["model"],
                stage=r["stage"],
                input_tokens=r.get("input_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
                total_tokens=r.get("total_tokens", 0),
                latency_ms=r.get("latency_ms", 0.0),
                cost_usd=r.get("cost_usd", 0.0),
            )
            for r in data.get("records", [])
        ]

    def reset(self) -> None:
        """Clear all records."""
        self.records.clear()
        self._stage_timers.clear()


# ---------------------------------------------------------------------------
# Measured usage (litellm callback)
# ---------------------------------------------------------------------------
# `CostTracker` above prices calls from a hand-maintained table, which is only
# ever an estimate: it does not know about cached-prompt discounts, reasoning
# tokens, or a provider's actual price list. `LiteLLMMeter` records what
# litellm itself reports for every completed call -- token breakdown,
# `response_cost`, and the model snapshot the provider answered with -- so the
# per-question cost written to a result row is a measurement, not a guess.
# (The estimate was 2-3x off the measured gpt-5-mini cost in September 2026.)
#
# litellm delivers success callbacks from a thread pool *after* the completion
# returns, while the pre-call hook runs synchronously in the caller's thread.
# The meter therefore tracks in-flight calls by id and offers `drain()` so a
# caller can wait for every call it started before reading the numbers.


@dataclass
class MeteredCall:
    """One completed LLM call as litellm reported it."""

    call_id: str
    model_spec: str  # model string as requested, e.g. "azure/gpt-5-mini"
    response_model: str  # model the provider answered with, e.g. "gpt-5-mini-2025-08-07"
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    started_at: float = 0.0
    ended_at: float = 0.0


def _usage_field(obj, name: str) -> int:
    value = getattr(obj, name, None) if obj is not None else None
    return int(value or 0)


def empty_usage() -> dict:
    """The usage block of a result row before any call has been metered."""
    return {
        "metered_calls": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "calls_without_cost": 0,
        "response_models": [],
    }


class LiteLLMMeter(CustomLogger):
    """Collect measured token counts, cost and response model per LLM call.

    Install once per process with `install()`. Aggregate a window of calls
    with `mark()` before the work and `usage_since(mark)` after it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self.calls: list[MeteredCall] = []
        self.failures: int = 0

    # -- litellm hooks -------------------------------------------------
    def log_pre_api_call(self, model, messages, kwargs) -> None:
        call_id = kwargs.get("litellm_call_id")
        if call_id:
            with self._lock:
                self._pending.add(call_id)

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        usage = getattr(response_obj, "usage", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        cost = kwargs.get("response_cost")
        started = start_time.timestamp() if hasattr(start_time, "timestamp") else 0.0
        ended = end_time.timestamp() if hasattr(end_time, "timestamp") else 0.0
        rec = MeteredCall(
            call_id=kwargs.get("litellm_call_id") or "",
            model_spec=str(kwargs.get("model") or ""),
            response_model=str(getattr(response_obj, "model", "") or ""),
            prompt_tokens=_usage_field(usage, "prompt_tokens"),
            cached_tokens=_usage_field(prompt_details, "cached_tokens"),
            completion_tokens=_usage_field(usage, "completion_tokens"),
            reasoning_tokens=_usage_field(completion_details, "reasoning_tokens"),
            cost_usd=float(cost) if cost is not None else None,
            latency_ms=(ended - started) * 1000 if started and ended else 0.0,
            started_at=started,
            ended_at=ended,
        )
        with self._lock:
            self.calls.append(rec)
            self._pending.discard(rec.call_id)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        with self._lock:
            self.failures += 1
            self._pending.discard(kwargs.get("litellm_call_id") or "")

    # -- lifecycle -----------------------------------------------------
    def install(self) -> LiteLLMMeter:
        """Register with litellm, replacing any meter installed earlier."""
        import litellm

        litellm.callbacks = [
            cb for cb in (litellm.callbacks or []) if not isinstance(cb, LiteLLMMeter)
        ] + [self]
        return self

    def drain(self, timeout: float = 10.0) -> bool:
        """Wait until every call that started has been reported.

        Returns False if some callback never arrived within `timeout`; the
        numbers read afterwards are then a lower bound.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if not self._pending:
                    return True
            if time.monotonic() >= deadline:
                with self._lock:
                    n = len(self._pending)
                logger.warning(f"LiteLLMMeter: {n} call(s) never reported within {timeout}s")
                return False
            time.sleep(0.01)

    # -- aggregation ---------------------------------------------------
    def mark(self) -> int:
        """Position in the call log; pass to `usage_since` to read a window."""
        with self._lock:
            return len(self.calls)

    def usage_since(self, mark: int) -> dict:
        """Aggregate every call recorded after `mark`."""
        with self._lock:
            window = self.calls[mark:]
        return self._aggregate(window)

    def usage_total(self) -> dict:
        with self._lock:
            window = list(self.calls)
        return self._aggregate(window)

    @property
    def observed_models(self) -> set[str]:
        with self._lock:
            return {c.response_model for c in self.calls if c.response_model}

    @staticmethod
    def _aggregate(window: list[MeteredCall]) -> dict:
        out = empty_usage()
        for c in window:
            out["metered_calls"] += 1
            out["prompt_tokens"] += c.prompt_tokens
            out["cached_tokens"] += c.cached_tokens
            out["completion_tokens"] += c.completion_tokens
            out["reasoning_tokens"] += c.reasoning_tokens
            if c.cost_usd is None:
                out["calls_without_cost"] += 1
            else:
                out["cost_usd"] += c.cost_usd
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
        out["response_models"] = sorted({c.response_model for c in window if c.response_model})
        return out
