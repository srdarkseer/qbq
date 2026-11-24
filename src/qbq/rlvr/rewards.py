"""Verifiable rewards: prompts, action parsing, and the programmatic reward rules.

No LLM ever assigns a reward. Every reward comes from the deterministic sandbox:

=====================  ======  ==============================================================
Outcome                Reward  Rule
=====================  ======  ==============================================================
``true_violation``     +1.0    the claimed trace breaks the invariant 3/3 on the target build
                               *and* does not on the reference (patched) build
``patch_verified``     +1.0    the patch passes the closed-loop gate (exploit 0/3, no
                               functional regression)
``holds``               0.0    a well-formed test that respects the invariant (valid negative)
``false_positive``     -1.0    the "violation" also reproduces on the reference build: it is
                               inherent to the oracle, not a defect (unverified claim)
``flaky``              -1.0    reproduced on some runs only (unverified claim)
``malformed``          -1.0    unparseable JSON, or a trace that does not fit the graph
``patch_rejected``     -1.0    the patch fails to apply or build, leaves the exploit, or
                               regresses a functional QA edge
=====================  ======  ==============================================================

The reference build is the environment's ``patched`` variant. Rewards therefore
come from a differential, deterministic comparison and never read the planted-bug
manifest. The manifest appears only as an analysis label in the exported rows.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from qbq.graph import InvariantAssertion, StateGraph
from qbq.gym.base import Environment
from qbq.search.combinators import classify_invariant
from qbq.search.policy import graph_slice
from qbq.verifier.replay import DeterministicVerifier, Verdict
from qbq.verifier.trace import (
    ActionStep,
    ActionTrace,
    CandidateFinding,
    TraceValidationError,
    validate_candidate,
)

REFERENCE_VARIANT = "patched"


class Outcome(StrEnum):
    TRUE_VIOLATION = "true_violation"
    HOLDS = "holds"
    FALSE_POSITIVE = "false_positive"
    FLAKY = "flaky"
    MALFORMED = "malformed"
    PATCH_VERIFIED = "patch_verified"
    PATCH_REJECTED = "patch_rejected"


REWARDS: dict[Outcome, float] = {
    Outcome.TRUE_VIOLATION: 1.0,
    Outcome.PATCH_VERIFIED: 1.0,
    Outcome.HOLDS: 0.0,
    Outcome.FALSE_POSITIVE: -1.0,
    Outcome.FLAKY: -1.0,
    Outcome.MALFORMED: -1.0,
    Outcome.PATCH_REJECTED: -1.0,
}


def outcome_for(target: Verdict, reference: Verdict | None) -> Outcome:
    if target is Verdict.INVALID:
        return Outcome.MALFORMED
    if target is Verdict.FLAKY:
        return Outcome.FLAKY
    if target is Verdict.CONFIRMED:
        if reference in {Verdict.CONFIRMED, Verdict.FLAKY}:
            return Outcome.FALSE_POSITIVE
        return Outcome.TRUE_VIOLATION
    return Outcome.HOLDS


# --------------------------------------------------------------------------- prompts & actions

TEST_SYSTEM_PROMPT = (
    "You are a QA and security test planner operating on a typed Behavioral State Graph. "
    "Given a graph slice and one target invariant, propose ONE replayable action trace that "
    "you believe violates the invariant on this build. Reference only edges, personas and "
    "resource ids that appear in the slice. Respond with a single JSON object matching "
    "`respond_with` and nothing else."
)

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["invariant_id", "hypothesis", "steps"],
    "properties": {
        "invariant_id": {"type": "string"},
        "hypothesis": {"type": "string"},
        "steps": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "required": ["edge_id", "actor_id"],
                "properties": {
                    "edge_id": {"type": "string"}, "actor_id": {"type": "string"},
                    "path_params": {"type": "object"}, "body": {"type": ["object", "null"]},
                    "concurrency": {"type": "integer", "minimum": 1, "maximum": 256},
                },
            },
        },
    },
}


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_test_prompt(graph: StateGraph, invariant: InvariantAssertion, edge_id: str) -> list[dict[str, str]]:
    """Chat-format prompt for one (invariant, edge) frame. Contains no credentials."""
    user = {
        "graph_slice": graph_slice(graph, edge_id),
        "invariant": {
            "id": invariant.id, "category": invariant.category.value, "severity": invariant.severity.value,
            "description": invariant.description, "oracle": classify_invariant(invariant).value,
            "checks": [c.model_dump(mode="json") for c in invariant.checks],
        },
        "respond_with": ACTION_SCHEMA,
    }
    return [{"role": "system", "content": TEST_SYSTEM_PROMPT}, {"role": "user", "content": _dumps(user)}]


def prompt_id(messages: Sequence[Mapping[str, str]]) -> str:
    return hashlib.sha256(_dumps(list(messages)).encode()).hexdigest()[:16]


def action_json(candidate: CandidateFinding) -> str:
    return _dumps({
        "invariant_id": candidate.invariant_id,
        "hypothesis": candidate.hypothesis,
        "steps": [s.model_dump(mode="json", exclude_defaults=True) for s in candidate.trace.steps],
    })


class ActionError(ValueError):
    """The completion is not a well-formed action for this graph."""


def completion_text(completion: Any) -> str:
    """Accept a plain string or TRL's conversational ``[{"role": ..., "content": ...}]``."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion and isinstance(completion[-1], Mapping):
        return str(completion[-1].get("content", ""))
    return str(completion)


def parse_action(text: str, graph: StateGraph, invariant_id: str) -> CandidateFinding:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ActionError("no JSON object in completion")
    try:
        data = json.loads(match.group(0))
        if data.get("invariant_id", invariant_id) != invariant_id:
            raise ActionError(f"action targets {data.get('invariant_id')!r}, prompt asked for {invariant_id!r}")
        steps = tuple(ActionStep.model_validate(s) for s in data["steps"])
        candidate = CandidateFinding(
            id="rlvr." + hashlib.sha256(text.encode()).hexdigest()[:12], invariant_id=invariant_id,
            hypothesis=str(data.get("hypothesis") or "model proposal"), trace=ActionTrace(steps=steps),
        )
        validate_candidate(graph, candidate)
    except ActionError:
        raise
    except (ValueError, KeyError, TypeError, ValidationError, TraceValidationError) as exc:
        raise ActionError(f"{type(exc).__name__}: {exc}") from exc
    return candidate


# --------------------------------------------------------------------------- reward engine


class RewardEngine:
    """Scores proposed actions on one environment, caching verdicts by (build, trace)."""

    def __init__(self, env: Environment, *, runs: int = 3) -> None:
        self.env = env
        self.runs = runs
        self._verdicts: dict[tuple[str, str, str], Verdict] = {}
        self._verifiers: dict[str, DeterministicVerifier] = {}

    def _verifier(self, variant: str) -> DeterministicVerifier:
        if variant not in self._verifiers:
            active = next(v.active_bugs for v in self.env.variants() if v.name == variant)
            self._verifiers[variant] = DeterministicVerifier(
                self.env.graph, self.env.app_factory(active), self.env.snapshot, runs=self.runs)
        return self._verifiers[variant]

    async def averdict(self, variant: str, candidate: CandidateFinding) -> Verdict:
        key = (variant, candidate.invariant_id, candidate.trace.model_dump_json())
        if key not in self._verdicts:
            self._verdicts[key] = (await self._verifier(variant).averify(candidate)).verdict
        return self._verdicts[key]

    async def ascore_action(
        self, text: str, variant: str, invariant_id: str
    ) -> tuple[float, Outcome, Verdict | None, Verdict | None]:
        try:
            candidate = parse_action(text, self.env.graph, invariant_id)
        except ActionError:
            return REWARDS[Outcome.MALFORMED], Outcome.MALFORMED, None, None
        target = await self.averdict(variant, candidate)
        reference = None
        if target is Verdict.CONFIRMED:
            reference = await self.averdict(REFERENCE_VARIANT, candidate)
        outcome = outcome_for(target, reference)
        return REWARDS[outcome], outcome, target, reference


class VerifierReward:
    """A TRL-compatible reward function: ``reward(prompts, completions, **columns) -> list[float]``.

    It reads the ``env``, ``variant`` and ``invariant_id`` dataset columns, parses
    each completion as an action, and scores it in the deterministic sandbox.
    Pass an instance as ``reward_funcs`` to ``trl.GRPOTrainer``. TRL is not a
    QbQ dependency.
    """

    __name__ = "qbq_verifier_reward"

    def __init__(self, environments: Mapping[str, Environment] | None = None, *, runs: int = 3) -> None:
        self.runs = runs
        self._envs = dict(environments or {})
        self._engines: dict[str, RewardEngine] = {}

    def engine(self, env_name: str) -> RewardEngine:
        if env_name not in self._engines:
            if env_name not in self._envs:
                from qbq.rlvr.recorder import load_app_environment

                self._envs[env_name] = load_app_environment(env_name)
            self._engines[env_name] = RewardEngine(self._envs[env_name], runs=self.runs)
        return self._engines[env_name]

    def __call__(
        self, prompts: Sequence[Any], completions: Sequence[Any], *,
        env: Sequence[str], variant: Sequence[str], invariant_id: Sequence[str], **_: Any,
    ) -> list[float]:
        engines = {e: self.engine(e) for e in env}  # resolve (and compile) outside the event loop

        async def score_all() -> list[float]:
            out = []
            for completion, e, v, inv in zip(completions, env, variant, invariant_id, strict=True):
                reward, *_ = await engines[e].ascore_action(completion_text(completion), v, inv)
                out.append(reward)
            return out

        return asyncio.run(score_all())
