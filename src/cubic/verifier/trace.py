"""Action traces: structured graph mutations replayed by the verifier.

Agents (and, in Milestone 1, a scripted baseline) never emit raw bash. They emit
sequences of steps that reference *edges* and *actors* of the State Graph, e.g.
"replay ``edge.withdraw`` as ``user_b`` with ``user_a``'s wallet id, 20x in parallel".
"""

from __future__ import annotations

import json
import shlex
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cubic.graph import StateGraph, StatusCodeCheck
from cubic.graph.ids import Identifier


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ActionStep(_Frozen):
    edge_id: Identifier
    actor_id: Identifier
    path_params: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] | None = None
    concurrency: int = Field(
        default=1, ge=1, le=256,
        description="Fire this many identical requests released by a single barrier.",
    )


class ActionTrace(_Frozen):
    steps: tuple[ActionStep, ...] = Field(min_length=1)


class CandidateFinding(_Frozen):
    """A hypothesis that ``trace`` violates invariant ``invariant_id``."""

    id: Identifier
    invariant_id: Identifier
    hypothesis: str = Field(min_length=1)
    trace: ActionTrace


class TraceValidationError(ValueError):
    """The candidate cannot be replayed against this graph."""


def validate_candidate(graph: StateGraph, candidate: CandidateFinding) -> None:
    try:
        invariant = graph.invariant(candidate.invariant_id)
        for step in candidate.trace.steps:
            graph.edge(step.edge_id).render_path(step.path_params)
            graph.actor(step.actor_id)
    except KeyError as exc:
        raise TraceValidationError(str(exc.args[0] if exc.args else exc)) from exc
    n_steps = len(candidate.trace.steps)
    for check in invariant.checks:
        if isinstance(check, StatusCodeCheck) and check.step >= n_steps:
            raise TraceValidationError(
                f"invariant {invariant.id!r} inspects step {check.step} but trace has {n_steps} steps"
            )


def render_curl(graph: StateGraph, trace: ActionTrace, base_url: str) -> str:
    """Render a trace as a standalone, replayable shell script of ``curl`` calls."""
    lines = ["#!/usr/bin/env bash", "set -u"]
    for index, step in enumerate(trace.steps):
        edge = graph.edge(step.edge_id)
        actor = graph.actor(step.actor_id)
        parts = ["curl", "-sS", "-X", edge.method.value]
        for name, value in actor.auth_headers().items():
            parts += ["-H", f"{name}: {value}"]
        if step.body is not None:
            parts += ["-H", "Content-Type: application/json", "--data", json.dumps(step.body, sort_keys=True)]
        parts.append(base_url.rstrip("/") + edge.render_path(step.path_params))
        cmd = shlex.join(parts)
        lines.append(f"# step {index}: {edge.id} as {actor.id}")
        if step.concurrency == 1:
            lines.append(cmd)
        else:
            lines.append(f"for _ in $(seq {step.concurrency}); do {cmd} & done; wait")
    return "\n".join(lines) + "\n"
