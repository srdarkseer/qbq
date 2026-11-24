"""Zero-extra-dependency LLM chat providers over ``httpx``.

Two wire formats cover most hosted and local models:

* ``OpenAICompatibleProvider`` sends ``POST {base_url}/chat/completions`` with
  ``Authorization: Bearer <key>``. That works for OpenAI, vLLM, Ollama
  (``http://localhost:11434/v1``), LM Studio, OpenRouter, and similar servers.
* ``GeminiProvider`` sends the same request to Google's OpenAI-compatible Gemini
  endpoint (``https://generativelanguage.googleapis.com/v1beta/openai``), with
  ``gemini-2.5-flash`` as the default model.

Structured output is enforced with ``response_format={"type": "json_object"}``
when ``json_mode`` is set. Requests use ``temperature=0``. Transient failures
(429, 5xx, transport errors) are retried with bounded exponential backoff.

Secret hygiene: the API key exists only in the request header. It never
appears in ``repr``, in exception messages (the key, and any other registered
secret, is scrubbed), in usage records, or in anything QbQ serializes.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
REDACTED = "[REDACTED]"

# Tests (and embedders) may route every provider through a custom transport, e.g.
# ``httpx.MockTransport``. ``None`` means real network I/O.
default_transport: httpx.BaseTransport | None = None


class LLMError(RuntimeError):
    """A provider call failed. The message never contains secrets."""


def scrub(text: str, secrets: Iterable[str]) -> str:
    for secret in sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return text


def scrub_messages(messages: Sequence[Mapping[str, str]], secrets: Iterable[str]) -> list[dict[str, str]]:
    secrets = list(secrets)
    return [{k: scrub(str(v), secrets) for k, v in m.items()} for m in messages]


@dataclass
class Usage:
    calls: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(repr=False)
class OpenAICompatibleProvider:
    """Chat completions over the OpenAI wire format."""

    model: str = DEFAULT_OPENAI_MODEL
    base_url: str = OPENAI_BASE_URL
    api_key: str | None = field(default=None, repr=False)
    timeout_s: float = 60.0
    max_retries: int = 2
    backoff_s: float = 1.0
    temperature: float = 0.0
    name: str = "openai"
    extra_headers: dict[str, str] = field(default_factory=dict)
    transport: httpx.BaseTransport | None = field(default=None, repr=False)
    usage: Usage = field(default_factory=Usage)
    sleep: Any = field(default=time.sleep, repr=False)

    def __repr__(self) -> str:
        key = "set" if self.api_key else "none"
        return f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r}, api_key={key})"

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages: Sequence[Mapping[str, str]], json_mode: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model, "messages": [dict(m) for m in messages], "temperature": self.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _safe(self, text: str) -> str:
        return scrub(text, [self.api_key or ""])

    def chat(self, messages: Sequence[Mapping[str, str]], *, json_mode: bool = False) -> str:
        """Send one chat request and return the assistant message content."""
        payload = self._payload(messages, json_mode)
        transport = self.transport or default_transport
        attempt = 0
        with httpx.Client(timeout=self.timeout_s, transport=transport) as client:
            while True:
                attempt += 1
                self.usage.calls += 1
                try:
                    response = client.post(self.endpoint, json=payload, headers=self._headers())
                except httpx.HTTPError as exc:
                    error = f"transport error: {type(exc).__name__}"
                    retryable = True
                else:
                    if response.status_code == 200:
                        return self._content(response)
                    error = f"HTTP {response.status_code}: {self._safe(response.text[:300])}"
                    retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable or attempt > self.max_retries:
                    self.usage.failures += 1
                    raise LLMError(f"{self.name} {self.model}: {error}")
                self.usage.retries += 1
                self.sleep(self.backoff_s * 2 ** (attempt - 1))

    def _content(self, response: httpx.Response) -> str:
        try:
            data = response.json()
            usage = data.get("usage") or {}
            self.usage.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.usage.completion_tokens += int(usage.get("completion_tokens") or 0)
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self.usage.failures += 1
            raise LLMError(f"{self.name} {self.model}: malformed response ({type(exc).__name__})") from None
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not isinstance(content, str):
            raise LLMError(f"{self.name} {self.model}: empty response")
        return content


@dataclass(repr=False)
class GeminiProvider(OpenAICompatibleProvider):
    """Google Gemini through its OpenAI-compatible endpoint (JSON mode supported)."""

    model: str = DEFAULT_GEMINI_MODEL
    base_url: str = GEMINI_OPENAI_BASE_URL
    name: str = "gemini"


PROVIDERS = ("none", "gemini", "openai")


def _is_local(url: str) -> bool:
    host = httpx.URL(url).host
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".local")


def build_provider(
    provider: str,
    *,
    model: str | None = None,
    api_key_env: str = "GEMINI_API_KEY",
    base_url: str | None = None,
    env: Mapping[str, str] | None = None,
) -> OpenAICompatibleProvider | None:
    """Construct a provider from CLI-style settings. Keys are read from the environment only.

    * ``gemini`` reads the key from ``api_key_env`` (default ``GEMINI_API_KEY``).
    * ``openai`` reads it from ``api_key_env``, falling back to ``OPENAI_API_KEY``
      when the default ``GEMINI_API_KEY`` was left unchanged. The base URL comes
      from ``base_url`` or ``OPENAI_BASE_URL``. Local servers (localhost) need no key.
    """
    env = os.environ if env is None else env
    if provider == "none":
        return None
    if provider == "gemini":
        key = env.get(api_key_env)
        if not key:
            raise LLMError(f"--llm-provider gemini needs an API key in ${api_key_env}")
        return GeminiProvider(model=model or DEFAULT_GEMINI_MODEL, api_key=key,
                              base_url=base_url or GEMINI_OPENAI_BASE_URL)
    if provider == "openai":
        url = base_url or env.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
        key = env.get(api_key_env) or (env.get("OPENAI_API_KEY") if api_key_env == "GEMINI_API_KEY" else None)
        if not key and not _is_local(url):
            raise LLMError(f"--llm-provider openai needs an API key in ${api_key_env} or $OPENAI_API_KEY")
        return OpenAICompatibleProvider(model=model or DEFAULT_OPENAI_MODEL, api_key=key, base_url=url)
    raise LLMError(f"unknown --llm-provider {provider!r}; choose from {', '.join(PROVIDERS)}")
