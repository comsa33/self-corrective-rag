"""Run manifest: the conditions a result directory was produced under.

The March 2026 paper runs recorded `top_k`, `seed` and the threshold, and
nothing else. When one cell had to be re-run months later there was no way to
show it had been run under the same model snapshot, package versions, request
parameters or code, so every partial re-run was thrown away and the campaign
re-done from scratch. `RunManifest` writes everything needed for that proof
into `manifest.json` next to the results, and `preflight` refuses to start a
run whose provider answers with a snapshot other than the declared one.

What is recorded, and why each item is there:

  models[*].spec / resolved       the model string as configured, and as sent
  models[*].observed_response_model
                                  `response.model` of a real reply, i.e. the
                                  snapshot that actually answered (the Azure
                                  and native OpenAI deployments of gpt-5-mini
                                  both report `gpt-5-mini-2025-08-07`)
  models[*].sent_params           the request parameters litellm forwards
                                  after `drop_params`, reconstructed with
                                  `get_optional_params`. This is the record of
                                  whether `temperature=0` was honoured
                                  (gpt-5-mini silently drops it) and whether
                                  `reasoning_effort` reached the provider
  packages / git                  prompt formats change with dspy, parameter
                                  handling with litellm, everything with code
  cache                           latency is meaningless from a warm cache
  seed / retrieval / agent        the settings that decide the numbers
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from agentic_rag.config.settings import PROJECT_ROOT, make_lm, settings
from agentic_rag.evaluation.cost_tracker import LiteLLMMeter

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1

MODEL_SLOTS = ("preprocess", "evaluate", "generate", "agent")

# Fields every successful result row must carry for the paper's tables.
# `verify_campaign.py` refuses a run that is missing any of them; extend this
# list when a table needs a value that is not yet stored (E1 item 8).
REQUIRED_ROW_FIELDS = (
    "id",
    "question",
    "reference",
    "prediction",
    "pipeline",
    "retry_count",
    "action_history",
    "evaluation_scores",
    "passages_used",
    "total_passages_retrieved",
    "llm_calls",
    "latency_seconds",
    "question_difficulty",
    "tool_score_trace",
    # measured usage (LiteLLMMeter)
    "metered_calls",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cost_usd",
    "response_models",
)

# Request parameters that carry credentials or routing and are not part of the
# model's behaviour. They are neither reconstructed nor written to disk.
_PRIVATE_LM_KWARGS = ("api_key", "api_base", "num_retries")

_TRACKED_PACKAGES = ("dspy", "litellm", "openai", "pydantic", "sentence-transformers")


class ManifestMismatchError(RuntimeError):
    """The provider answered with a snapshot other than the declared one."""


class ModelSlotRecord(BaseModel):
    """One of the four model slots, as configured and as observed."""

    slot: str
    spec: str
    resolved: str
    provider: str | None = None
    api_version: str | None = None
    reasoning_model: bool = False
    dspy_kwargs: dict = Field(default_factory=dict)
    sent_params: dict = Field(default_factory=dict)
    expected_response_model: str | None = None
    observed_response_model: str | None = None


class RunManifest(BaseModel):
    manifest_version: int = MANIFEST_VERSION
    run_id: str
    created_at: str
    finished_at: str | None = None
    command: list[str]
    config_path: str | None = None
    experiment: str
    dataset: str
    n: int
    sample_size: int | None = None
    variants: list[str]
    seed: int
    git: dict
    packages: dict
    cache: dict
    model_defaults: dict
    retrieval: dict
    evaluation: dict
    agent: dict
    models: list[ModelSlotRecord]
    preflight: dict = Field(default_factory=lambda: {"status": "pending"})
    observed_response_models: list[str] = Field(default_factory=list)
    usage_total: dict | None = None

    # -- persistence ---------------------------------------------------
    def save(self, run_dir: Path) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / MANIFEST_FILENAME
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, run_dir: Path) -> RunManifest:
        path = Path(run_dir) / MANIFEST_FILENAME
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    # -- checks -------------------------------------------------------
    def run_preflight(self, run_dir: Path, meter: LiteLLMMeter | None = None) -> None:
        """Ask each distinct model for one reply and compare the snapshot.

        The manifest is written before the check so that a mismatch leaves
        evidence on disk. Raises `ManifestMismatchError` on the first slot
        whose reply names a snapshot other than the declared one.
        """
        meter = meter or LiteLLMMeter().install()
        observed: dict[str, str | None] = {}
        for rec in self.models:
            if rec.spec not in observed:
                observed[rec.spec] = _observe_response_model(rec.spec, self.run_id, meter)
            rec.observed_response_model = observed[rec.spec]

        mismatches = [
            f"{rec.slot}: declared {rec.expected_response_model!r}, "
            f"got {rec.observed_response_model!r}"
            for rec in self.models
            if rec.expected_response_model
            and rec.observed_response_model != rec.expected_response_model
        ]
        unpinned = sorted({rec.spec for rec in self.models if not rec.expected_response_model})
        self.preflight = {
            "status": "mismatch" if mismatches else "ok",
            "checked_at": _now(),
            "mismatches": mismatches,
            "unpinned": unpinned,
        }
        self.save(run_dir)

        if mismatches:
            raise ManifestMismatchError(
                "Refusing to start: model snapshot differs from LLM_EXPECTED_RESPONSE_MODELS -- "
                + "; ".join(mismatches)
            )
        if unpinned:
            logger.warning(
                f"Model snapshot unpinned for {unpinned}: the run records what it observes "
                f"but cannot abort on a change. Set LLM_EXPECTED_RESPONSE_MODELS to pin it."
            )
        logger.info(f"Preflight OK: {[(r.slot, r.observed_response_model) for r in self.models]}")

    def finish_run(self, run_dir: Path, meter: LiteLLMMeter | None) -> None:
        """Record what the whole run observed, then rewrite the manifest."""
        self.finished_at = _now()
        if meter is not None:
            meter.drain()
            self.observed_response_models = sorted(meter.observed_models)
            self.usage_total = meter.usage_total()
        self.save(run_dir)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def build_manifest(
    *,
    run_id: str,
    experiment: str,
    dataset: str,
    n: int,
    variants: list[str],
    config_path: str | None = None,
    sample_size: int | None = None,
) -> RunManifest:
    """Snapshot the process-wide settings and environment for one run."""
    return RunManifest(
        run_id=run_id,
        created_at=_now(),
        command=list(sys.argv),
        config_path=config_path,
        experiment=experiment,
        dataset=dataset,
        n=n,
        sample_size=sample_size,
        variants=list(variants),
        seed=settings.experiment.seed,
        git=git_state(),
        packages=package_versions(),
        cache={
            "llm_cache_disabled": settings.disable_llm_cache,
        },
        model_defaults={
            "temperature": settings.model.temperature,
            "max_tokens": settings.model.max_tokens,
            "reasoning_effort": settings.model.reasoning_effort,
            "num_retries": settings.model.num_retries,
            "embedding_model": settings.model.embedding_model,
        },
        retrieval={
            "top_k": settings.retrieval.top_k,
            "text_top_k": settings.retrieval.text_top_k,
            "query_method": settings.retrieval.query_method,
            "hybrid_weight": settings.retrieval.hybrid_weight,
            "max_passages": settings.retrieval.max_passages,
        },
        evaluation={
            "quality_threshold": settings.evaluation.quality_threshold,
            "max_retry": settings.evaluation.max_retry_count,
        },
        agent={
            "enabled_tools": settings.agent.enabled_tools,
            "max_iterations": settings.agent.max_iterations,
        },
        models=[model_slot_record(slot) for slot in MODEL_SLOTS],
    )


def model_slot_record(slot: str) -> ModelSlotRecord:
    """Describe one model slot, including the parameters litellm would send."""
    spec = getattr(settings.model, f"{slot}_model")
    lm = make_lm(spec)
    dspy_kwargs = {k: v for k, v in lm.kwargs.items() if k not in _PRIVATE_LM_KWARGS}
    provider = _provider_of(lm.model)
    return ModelSlotRecord(
        slot=slot,
        spec=spec,
        resolved=lm.model,
        provider=provider,
        api_version=os.environ.get("AZURE_API_VERSION") if provider == "azure" else None,
        reasoning_model=settings.model.is_reasoning_model(spec),
        dspy_kwargs=dspy_kwargs,
        sent_params=sent_params(lm.model, dspy_kwargs),
        expected_response_model=settings.model.expected_response_model(spec),
    )


def sent_params(model: str, lm_kwargs: dict) -> dict:
    """Reconstruct what litellm forwards to the provider for `model`.

    Uses litellm's own `get_optional_params` with `drop_params=True`, the
    setting `make_lm` runs under, so a parameter the provider does not accept
    disappears here exactly as it does on the wire.
    """
    import litellm
    from litellm.utils import get_optional_params

    name, provider, _, _ = litellm.get_llm_provider(model)
    accepted = set(inspect.signature(get_optional_params).parameters)
    kwargs = {k: v for k, v in lm_kwargs.items() if k in accepted and k not in _PRIVATE_LM_KWARGS}
    params = get_optional_params(
        model=name, custom_llm_provider=provider, drop_params=True, **kwargs
    )
    return {k: v for k, v in params.items() if v not in (None, {}, [])}


def git_state(root: Path = PROJECT_ROOT) -> dict:
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        ).stdout.rstrip("\n")

    # Porcelain lines are "XY path"; the status letters may be a space, so
    # the leading whitespace of the first line must survive parsing.
    status = _git("status", "--porcelain", "--untracked-files=no")
    modified = [line[2:].strip() for line in status.splitlines() if line.strip()]
    return {
        "commit": _git("rev-parse", "HEAD") or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(modified),
        "modified_files": modified,
    }


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _provider_of(model: str) -> str | None:
    import litellm

    try:
        return litellm.get_llm_provider(model)[1]
    except Exception:  # unknown model string: recorded as-is, provider unknown
        return None


def _observe_response_model(spec: str, run_id: str, meter: LiteLLMMeter) -> str | None:
    """Make one real call with `spec` and return the snapshot it answered with.

    The prompt carries the run id so it can never be served from a cache,
    which would report no response at all.
    """
    lm = make_lm(spec)
    mark = meter.mark()
    lm(f"Reply with the single word OK. (run {run_id})")
    meter.drain()
    calls = meter.calls[mark:]
    if not calls:
        logger.warning(f"Preflight for {spec}: no response was metered")
        return None
    return calls[-1].response_model or None


def load_manifest_or_none(run_dir: Path) -> RunManifest | None:
    path = Path(run_dir) / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        return RunManifest.load(run_dir)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Unreadable manifest at {path}: {e}")
        return None
