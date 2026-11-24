"""Data model for the closed remediation loop: finding -> patch -> verified fix -> regression test."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from qbq.graph import InvariantAssertion, StateGraph
from qbq.search.combinators import Oracle, classify_invariant, probe_step
from qbq.verifier.replay import VerificationResult
from qbq.verifier.trace import CandidateFinding


class SourceRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    file: str = Field(description="Display path from compile provenance, e.g. qbq/gym/fintech/app.py")
    function: str
    line: int


class ConfirmedFinding(BaseModel):
    """A 3-of-3 confirmed invariant violation, located in the source."""

    model_config = ConfigDict(frozen=True)

    id: str
    variant: str
    invariant: InvariantAssertion
    oracle: Oracle
    candidate: CandidateFinding
    edge_id: str
    source: SourceRef | None
    reproduction: str = Field(repr=False)

    @classmethod
    def from_verification(
        cls, graph: StateGraph, candidate: CandidateFinding, result: VerificationResult, *, variant: str = ""
    ) -> ConfirmedFinding:
        if not result.confirmed:
            raise ValueError(f"{candidate.id}: verdict {result.verdict.value}, not confirmed")
        invariant = graph.invariant(candidate.invariant_id)
        steps = candidate.trace.steps
        edge = graph.edge(steps[min(probe_step(invariant), len(steps) - 1)].edge_id)
        src = edge.metadata.get("source") or {}
        source = (
            SourceRef(file=src["file"], function=src["function"], line=int(src["line"]))
            if src.get("file") and src.get("function") and src.get("line") else None
        )
        return cls(
            id=candidate.id, variant=variant, invariant=invariant, oracle=classify_invariant(invariant),
            candidate=candidate, edge_id=edge.id, source=source, reproduction=result.reproduction or "",
        )


class PatchCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    finding_id: str
    patcher: str
    rule: str
    file: str
    description: str
    diff: str
    patched_source: str = Field(repr=False, exclude=True)


class FixVerdict(StrEnum):
    VERIFIED = "verified"
    EXPLOIT_REPRODUCES = "exploit_reproduces"
    FUNCTIONAL_REGRESSION = "functional_regression"
    PATCH_INVALID = "patch_invalid"


class FunctionalCheck(BaseModel):
    test_id: str
    invariant_id: str
    before: str
    after: str

    @property
    def regressed(self) -> bool:
        return self.before == "not_reproduced" and self.after != "not_reproduced"


class FixVerification(BaseModel):
    finding_id: str
    patcher: str
    rule: str
    verdict: FixVerdict
    runs: int
    exploit_violations: int = 0
    exploit_verdict: str | None = None
    functional: list[FunctionalCheck] = Field(default_factory=list)
    error: str | None = None
    duration_s: float = 0.0

    @property
    def verified(self) -> bool:
        return self.verdict is FixVerdict.VERIFIED

    @property
    def regressions(self) -> list[str]:
        return [c.test_id for c in self.functional if c.regressed]
