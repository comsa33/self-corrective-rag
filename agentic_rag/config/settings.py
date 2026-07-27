"""Global experiment settings for Self-Corrective RAG.

All hyperparameters and configuration values are centralized here
for reproducibility and easy ablation study control.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indices"
RESULTS_DIR = DATA_DIR / "results"

# Model-name prefix routing a request to Ollama Cloud (see make_lm).
OLLAMA_CLOUD_PREFIX = "ollama_cloud/"

# Floor on max_tokens for reasoning models, which spend part of the budget on
# hidden reasoning. DSPy enforces this same floor for the gpt-5 family.
REASONING_MIN_MAX_TOKENS = 16000


class ModelSettings(BaseSettings):
    """LLM and embedding model configuration."""

    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM models
    preprocess_model: str = Field("gpt-4o-mini", alias="PREPROCESS_MODEL")
    evaluate_model: str = Field("gpt-4o-mini", alias="EVALUATE_MODEL")
    generate_model: str = Field("gpt-4o", alias="GENERATE_MODEL")
    agent_model: str = Field("gpt-4o", alias="AGENT_MODEL")

    # Embedding
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimension: int = 1536

    # LLM parameters
    temperature: float = 0.0
    max_tokens: int = Field(4096, alias="LLM_MAX_TOKENS")
    # Reasoning budget for reasoning-capable models. "default" leaves the
    # provider's own setting alone, which some models need in order to emit a
    # complete structured response.
    reasoning_effort: str = Field("low", alias="LLM_REASONING_EFFORT")
    num_retries: int = Field(3, alias="LLM_NUM_RETRIES")


class RetrievalSettings(BaseSettings):
    """Retrieval hyperparameters."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    # Search parameters
    top_k: int = 50
    text_top_k: int = 40
    query_method: str = "rrf"  # rrf | vector_only | text_only | combined
    hybrid_weight: float = 0.48  # vector search weight in RRF

    # Passage accumulation
    max_passages: int = 30  # FIFO eviction above this


class EvaluationSettings(BaseSettings):
    """4D quality evaluation hyperparameters."""

    model_config = SettingsConfigDict(env_prefix="EVAL_")

    # Quality threshold
    quality_threshold: int = 40  # score >= threshold → "output"

    # Max retry
    max_retry_count: int = 3

    # 4D score ranges (for reference / validation)
    relevance_max: int = 30
    coverage_max: int = 25
    specificity_max: int = 25
    sufficiency_max: int = 20
    total_max: int = 100

    # Topic categories (13 categories from the original system)
    topic_categories: list[str] = [
        "일반",
        "DSL 함수",
        "시나리오",
        "API",
        "아키텍처",
        "DevOps",
        "데이터",
        "보안",
        "성능",
        "UI/UX",
        "통합",
        "트러블슈팅",
        "기타",
    ]


class ExperimentSettings(BaseSettings):
    """Experiment control parameters."""

    model_config = SettingsConfigDict(env_prefix="EXPERIMENT_")

    seed: int = Field(42, alias="EXPERIMENT_SEED")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Datasets
    datasets: list[str] = ["popqa", "hotpotqa", "natural_questions", "financebench"]

    # Ablation flags — toggle individual contributions on/off
    enable_iteration: bool = True  # C1: iterative loop
    enable_accumulation: bool = True  # C1: passage accumulation
    enable_4d_evaluation: bool = True  # C2: 4D quality assessment
    enable_refinement: bool = True  # C3: targeted query refinement
    enable_agent_routing: bool = True  # C4: 3-way agent routing
    enable_dspy: bool = True  # C5: DSPy pipeline (vs manual prompt)
    enable_agentic_refinement: bool = False  # C1: ReAct-based agentic refinement (opt-in)


class AgentSettings(BaseSettings):
    """ReAct agent hyperparameters for agentic retrieval refinement."""

    model_config = SettingsConfigDict(env_prefix="AGENT_")

    max_iterations: int = 5  # ReAct reasoning-action iterations

    # Tool-level ablation: None = all tools enabled
    # Valid names: search, decompose, structure, terminology, evaluate, inspect
    enabled_tools: list[str] | None = None


class Settings(BaseSettings):
    """Root settings aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    ollama_api_key: str = Field("", alias="OLLAMA_API_KEY")
    ollama_api_base: str = Field("https://ollama.com/v1", alias="OLLAMA_API_BASE")

    # Disable the LLM response cache. Required whenever latency is measured:
    # at temperature=0 a rerun returns cached completions in milliseconds,
    # which deflates latency without changing any answer.
    disable_llm_cache: bool = Field(False, alias="DISABLE_LLM_CACHE")

    # Sub-settings
    model: ModelSettings = Field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    index_dir: Path = INDEX_DIR
    results_dir: Path = Field(RESULTS_DIR, alias="RESULTS_DIR")


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------
settings = Settings()


def make_lm(model: str, **kwargs):
    """Create a dspy.LM with default retry/temperature from settings.

    Centralizes LM creation so rate-limit retry settings are consistent.
    """
    import dspy
    import litellm

    # gpt-5 reasoning models don't support temperature; drop unsupported params
    litellm.drop_params = True

    # max_tokens must be passed explicitly. Without it the provider default
    # applies, and a provider with a small default truncates the response
    # mid-structure — which surfaces as unparsable tool calls rather than as
    # an obvious error.
    defaults = {
        "temperature": settings.model.temperature,
        "num_retries": settings.model.num_retries,
        "max_tokens": settings.model.max_tokens,
    }
    # Reasoning models: hold effort at "low" so every model is compared at a
    # similar compute budget. Left at its default, gpt-oss spends thousands of
    # reasoning tokens on a single preprocessing call and truncates mid-answer.
    model_lower = model.lower()
    is_reasoning_model = "gpt-5" in model_lower or "gpt-oss" in model_lower
    if is_reasoning_model:
        if "reasoning_effort" not in kwargs and settings.model.reasoning_effort != "default":
            defaults["reasoning_effort"] = settings.model.reasoning_effort
        # Reasoning tokens are drawn from the same budget as the answer, so a
        # limit sized for ordinary completions truncates mid-structure. DSPy
        # also rejects anything under 16000 for the gpt-5 family outright.
        defaults["max_tokens"] = max(settings.model.max_tokens, REASONING_MIN_MAX_TOKENS)

    # Ollama Cloud is reached through its OpenAI-compatible endpoint so the
    # API key travels as a bearer token. Written as "ollama_cloud/<model>"
    # to stay distinct from litellm's own local-Ollama provider.
    if model.startswith(OLLAMA_CLOUD_PREFIX):
        defaults.setdefault("api_base", settings.ollama_api_base)
        defaults.setdefault("api_key", settings.ollama_api_key)
        model = f"openai/{model[len(OLLAMA_CLOUD_PREFIX) :]}"

    defaults.update(kwargs)
    return dspy.LM(model, **defaults)
