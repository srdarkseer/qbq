"""End-to-end closed loop: compile -> search -> verify -> patch -> verify fix -> export test.

``aremediate`` runs the whole Milestone 1-4 pipeline against one build of a Bug Gym
environment:

1. **compile** the app from source plus runtime, including persona grounding;
2. **search** the compiled graph for a prioritized test plan;
3. **verify** every planned test 3-of-3 on the target build and keep the
   confirmed findings (one per violated invariant, the first in plan order);
4. **patch** each finding: candidates from every patcher are tried in order
   until one passes the verification gate (exploit 0/``runs`` *and* no
   functional QA regression);
5. **export** a standalone pytest regression test per finding, then check it.
   It must fail on the unpatched build and pass on the verified patched code
   under the same build toggles.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cubic.compiler import acompile_app
from cubic.gym.base import AppSource, Environment
from cubic.remediate.exporter import export_regression_test, render_regression_test
from cubic.remediate.models import ConfirmedFinding, FixVerification, PatchCandidate
from cubic.remediate.patcher import Patcher, PatchError, RulePatcher, load_context
from cubic.remediate.verify_fix import _verdicts, averify_fix, patched_module
from cubic.search import Oracle, SearchPolicy, search
from cubic.verifier.replay import DeterministicVerifier


class RegressionCheck(BaseModel):
    file: str
    fails_on_unpatched: bool
    passes_on_patched: bool | None = None

    @property
    def valid(self) -> bool:
        return self.fails_on_unpatched and self.passes_on_patched is not False


class FindingRemediation(BaseModel):
    finding_id: str
    invariant_id: str
    oracle: Oracle
    edge_id: str
    source: str | None
    attempts: list[FixVerification] = Field(default_factory=list)
    patches: list[PatchCandidate] = Field(default_factory=list)
    verified_patch: PatchCandidate | None = None
    regression_test: RegressionCheck | None = None
    error: str | None = None

    @property
    def verified(self) -> bool:
        return self.verified_patch is not None


class RemediationReport(BaseModel):
    environment: str
    variant: str
    active_bugs: list[str]
    runs: int
    graph: dict[str, int]
    plan_tests: int
    invariants_covered: int
    findings: list[FindingRemediation]
    timings_s: dict[str, float] = Field(default_factory=dict)

    @property
    def verified(self) -> int:
        return sum(f.verified for f in self.findings)

    @property
    def patch_pass_rate(self) -> float | None:
        return self.verified / len(self.findings) if self.findings else None


def _variant(env: Environment, name: str) -> frozenset[str]:
    for variant in env.variants():
        if variant.name == name:
            return variant.active_bugs
    raise KeyError(f"unknown variant {name!r}; available: {[v.name for v in env.variants()]}")


def _run_exported(source: str, factory: Any, args: tuple) -> bool:
    """Execute a rendered regression test in-process; True if every parametrized run passes."""
    namespace: dict[str, Any] = {"__name__": "cubic_regression_check"}
    exec(compile(source, "<regression>", "exec"), namespace)  # noqa: S102 - our own generated code
    namespace["_create_app"], namespace["APP_FACTORY_ARGS"] = factory, args
    test = next(v for k, v in namespace.items() if k.startswith("test_regression_"))
    try:
        for run in range(3):
            test(run)
    except AssertionError:
        return False
    return True


async def aremediate(
    env: Environment,
    *,
    variant: str = "vulnerable",
    runs: int = 3,
    patchers: Sequence[Patcher] | None = None,
    policy: SearchPolicy | None = None,
    out_dir: Path | str | None = None,
) -> RemediationReport:
    if env.app_source is None:
        raise ValueError(f"environment {env.name!r} has no app_source to patch")
    source: AppSource = env.app_source
    active = _variant(env, variant)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    module = importlib.import_module(source.module)
    factory = getattr(module, source.factory)
    personas = [a.model_copy(update={"observed_state": {}}) for a in env.graph.actors]
    compiled = await acompile_app(lambda: source.build(factory), personas=personas, app_name=env.graph.app_name)
    graph = compiled.graph
    t1 = time.perf_counter()
    timings["compile"] = t1 - t0

    plan = search(graph, policy)
    functional = [t.to_candidate() for t in plan.tests if t.oracle is Oracle.SUCCESS]
    t2 = time.perf_counter()
    timings["search"] = t2 - t1

    def build() -> Any:
        return source.build(factory, active)

    verifier = DeterministicVerifier(graph, build, env.snapshot, runs=runs)
    findings: list[ConfirmedFinding] = []
    seen: set[str] = set()
    for candidate in plan.candidates():
        result = await verifier.averify(candidate)
        if result.confirmed and candidate.invariant_id not in seen:
            seen.add(candidate.invariant_id)
            findings.append(ConfirmedFinding.from_verification(graph, candidate, result, variant=variant))
    t3 = time.perf_counter()
    timings["verify"] = t3 - t2

    baseline = await _verdicts(graph, build, env.snapshot, functional, runs)
    out: list[FindingRemediation] = []
    for finding in findings:
        record = FindingRemediation(
            finding_id=finding.id, invariant_id=finding.invariant.id, oracle=finding.oracle,
            edge_id=finding.edge_id,
            source=f"{finding.source.file}:{finding.source.line}" if finding.source else None,
        )
        try:
            ctx = load_context(finding, source)
            for patcher in patchers or (RulePatcher(),):
                record.patches += patcher.propose(finding, ctx)
        except PatchError as exc:
            record.error = str(exc)
        for patch in record.patches:
            attempt = await averify_fix(
                finding, patch, graph=graph, app_source=source, active_bugs=active,
                original_factory=build, original_snapshot=env.snapshot,
                functional_tests=functional, runs=runs, baseline=baseline,
            )
            record.attempts.append(attempt)
            if attempt.verified:
                record.verified_patch = patch
                break
        out.append(record)
    t4 = time.perf_counter()
    timings["patch_and_verify"] = t4 - t3

    for finding, record in zip(findings, out, strict=True):
        rendered = render_regression_test(finding, graph, source)
        name = (export_regression_test(finding, graph, source, out_dir).name if out_dir is not None
                else f"<memory>/{finding.invariant.id}")
        # the exported test drives its own event loop, so it runs in a worker thread
        args = source.factory_args(active)
        unpatched_ok = await asyncio.to_thread(_run_exported, rendered, factory, args)
        check = RegressionCheck(file=name, fails_on_unpatched=not unpatched_ok)
        if record.verified_patch is not None:
            with patched_module(record.verified_patch, source) as fixed:
                check.passes_on_patched = await asyncio.to_thread(
                    _run_exported, rendered, getattr(fixed, source.factory), args)
        record.regression_test = check
    t5 = time.perf_counter()
    timings["export"] = t5 - t4
    timings["total"] = t5 - t0

    return RemediationReport(
        environment=env.name, variant=variant, active_bugs=sorted(active), runs=runs,
        graph={"nodes": len(graph.nodes), "edges": len(graph.edges), "invariants": len(graph.invariants)},
        plan_tests=len(plan.tests), invariants_covered=sum(1 for c in plan.coverage.values() if c.tests),
        findings=out, timings_s=timings,
    )


def remediate(env: Environment, **kwargs: Any) -> RemediationReport:
    return asyncio.run(aremediate(env, **kwargs))
