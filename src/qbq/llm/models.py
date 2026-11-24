"""Adapters that plug a chat provider into QbQ's model protocols.

* ``ProviderRankingModel`` implements ``qbq.search.policy.RankingModel``. It asks
  for a JSON ranking in JSON mode, for ``LLMGuidedPolicy``.
* ``ProviderPatchModel`` implements ``qbq.remediate.patcher.PatchModel``. It asks
  for a unified diff, for ``LLMPatcher``.

Both scrub every registered secret (persona credentials from the graph, the
provider's API key) out of the outgoing messages before they leave the process.
The prompts are built credential-free already, so this is defence in depth.
Model output is only ever a *proposal*. Rankings cannot bring back pruned
tests, and every patch must pass the deterministic fix gate.
"""

from __future__ import annotations

from collections.abc import Iterable

from qbq.graph import StateGraph
from qbq.llm.providers import OpenAICompatibleProvider, scrub_messages
from qbq.remediate.patcher import PatchPrompt
from qbq.search.policy import RankingPrompt

PATCH_FORMAT_REMINDER = (
    "Return only a unified diff for the single file named in `file`, inside one ```diff fenced "
    "block. Use the exact context lines from `ast_slice` (without the line-number prefix)."
)


def graph_secrets(graph: StateGraph | None) -> list[str]:
    """Every credential the graph knows about (persona bearer tokens)."""
    if graph is None:
        return []
    return [a.auth_token for a in graph.actors if a.auth_token]


class _ProviderModel:
    def __init__(self, provider: OpenAICompatibleProvider, *, secrets: Iterable[str] = ()) -> None:
        self.provider = provider
        self.secrets = [*secrets, provider.api_key or ""]
        self.sent: list[list[dict[str, str]]] = []  # scrubbed transcripts, for audit and tests

    def _send(self, messages: list[dict[str, str]], *, json_mode: bool) -> str:
        clean = scrub_messages(messages, self.secrets)
        self.sent.append(clean)
        return self.provider.chat(clean, json_mode=json_mode)


class ProviderRankingModel(_ProviderModel):
    def complete(self, prompt: RankingPrompt) -> str:
        return self._send(prompt.messages(), json_mode=True)


class ProviderPatchModel(_ProviderModel):
    def complete(self, prompt: PatchPrompt) -> str:
        messages = prompt.messages()
        messages[0] = {**messages[0], "content": messages[0]["content"] + " " + PATCH_FORMAT_REMINDER}
        return self._send(messages, json_mode=False)
