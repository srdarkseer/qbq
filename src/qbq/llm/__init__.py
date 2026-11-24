"""Live LLM providers (Gemini, OpenAI-compatible) over httpx, and their QbQ adapters."""

from qbq.llm.models import ProviderPatchModel, ProviderRankingModel, graph_secrets
from qbq.llm.providers import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_OPENAI_BASE_URL,
    OPENAI_BASE_URL,
    PROVIDERS,
    GeminiProvider,
    LLMError,
    OpenAICompatibleProvider,
    Usage,
    build_provider,
    scrub,
    scrub_messages,
)

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "GEMINI_OPENAI_BASE_URL",
    "OPENAI_BASE_URL",
    "PROVIDERS",
    "GeminiProvider",
    "LLMError",
    "OpenAICompatibleProvider",
    "ProviderPatchModel",
    "ProviderRankingModel",
    "Usage",
    "build_provider",
    "graph_secrets",
    "scrub",
    "scrub_messages",
]
