"""Deterministic sandbox verifier: the zero-hallucination gate.

For every candidate the verifier builds a *fresh* application instance (a forked,
freshly seeded database), replays the trace in-process over ASGI, evaluates the
invariant deterministically, and repeats. A finding is CONFIRMED only if the
invariant breaks on every run; partial reproductions are FLAKY and discarded.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel, Field

from qbq.graph import StateGraph
from qbq.verifier.checks import CheckOutcome, Exchange, evaluate_invariant
from qbq.verifier.snapshot import DbDiff, DbSnapshot, diff_snapshots
from qbq.verifier.trace import (
    ActionStep,
    ActionTrace,
    CandidateFinding,
    TraceValidationError,
    render_curl,
    validate_candidate,
)

AppFactory = Callable[[], Any]
Snapshotter = Callable[[Any], DbSnapshot]

SANDBOX_BASE_URL = "http://qbq-sandbox"


class Verdict(StrEnum):
    CONFIRMED = "confirmed"
    NOT_REPRODUCED = "not_reproduced"
    FLAKY = "flaky"
    INVALID = "invalid"


class RunRecord(BaseModel):
    violated: bool
    outcomes: list[CheckOutcome]


class VerificationResult(BaseModel):
    candidate_id: str
    invariant_id: str
    verdict: Verdict
    runs: list[RunRecord] = Field(default_factory=list)
    error: str | None = None
    evidence_exchanges: list[Exchange] = Field(default_factory=list)
    evidence_db_diff: DbDiff | None = None
    reproduction: str | None = None
    duration_s: float = 0.0

    @property
    def confirmed(self) -> bool:
        return self.verdict is Verdict.CONFIRMED


async def replay_trace(app: Any, graph: StateGraph, trace: ActionTrace) -> list[list[Exchange]]:
    """Execute a trace against an ASGI app; returns exchanges grouped per step."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=SANDBOX_BASE_URL) as client:
        results: list[list[Exchange]] = []
        for index, step in enumerate(trace.steps):
            results.append(await _run_step(client, graph, index, step))
        return results


async def _run_step(
    client: httpx.AsyncClient, graph: StateGraph, index: int, step: ActionStep
) -> list[Exchange]:
    edge = graph.edge(step.edge_id)
    actor = graph.actor(step.actor_id)
    path = edge.render_path(step.path_params)

    async def send() -> Exchange:
        response = await client.request(
            edge.method.value, path, headers=actor.auth_headers(), json=step.body
        )
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text
        return Exchange(
            step=index, edge_id=edge.id, actor_id=actor.id, method=edge.method.value,
            path=path, request_body=step.body, status_code=response.status_code,
            response_body=body,
        )

    if step.concurrency == 1:
        return [await send()]

    barrier = asyncio.Barrier(step.concurrency)

    async def fire() -> Exchange:
        await barrier.wait()  # release every request in the same scheduler tick
        return await send()

    return list(await asyncio.gather(*(fire() for _ in range(step.concurrency))))


class DeterministicVerifier:
    def __init__(
        self,
        graph: StateGraph,
        app_factory: AppFactory,
        snapshot: Snapshotter,
        *,
        runs: int = 3,
    ) -> None:
        if runs < 1:
            raise ValueError("runs must be >= 1")
        self.graph = graph
        self.app_factory = app_factory
        self.snapshot = snapshot
        self.runs = runs

    async def averify(self, candidate: CandidateFinding) -> VerificationResult:
        started = time.perf_counter()
        result = VerificationResult(
            candidate_id=candidate.id, invariant_id=candidate.invariant_id,
            verdict=Verdict.NOT_REPRODUCED,
        )
        try:
            validate_candidate(self.graph, candidate)
        except TraceValidationError as exc:
            result.verdict, result.error = Verdict.INVALID, str(exc)
            return result

        invariant = self.graph.invariant(candidate.invariant_id)
        for _ in range(self.runs):
            app = self.app_factory()  # fresh, seeded state for every run
            before = self.snapshot(app)
            exchanges = await replay_trace(app, self.graph, candidate.trace)
            after = self.snapshot(app)
            outcomes = evaluate_invariant(invariant, exchanges, after)
            violated = not all(o.holds for o in outcomes)
            result.runs.append(RunRecord(violated=violated, outcomes=outcomes))
            if violated and result.evidence_db_diff is None:
                result.evidence_exchanges = [ex for step in exchanges for ex in step]
                result.evidence_db_diff = diff_snapshots(before, after)

        hits = sum(r.violated for r in result.runs)
        if hits == self.runs:
            result.verdict = Verdict.CONFIRMED
            result.reproduction = render_curl(self.graph, candidate.trace, SANDBOX_BASE_URL)
        elif hits:
            result.verdict = Verdict.FLAKY
            # unreproducible findings are discarded: drop evidence so nothing leaks into reports
            result.evidence_exchanges, result.evidence_db_diff = [], None
        result.duration_s = time.perf_counter() - started
        return result

    def verify(self, candidate: CandidateFinding) -> VerificationResult:
        return asyncio.run(self.averify(candidate))
