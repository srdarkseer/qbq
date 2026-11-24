"""Eval harness: score candidate findings against Bug Gym ground truth.

Definitions (computed over every build variant of an environment):

* **Recall@1** - for every (variant, active bug) pair, take the *first* submitted
  candidate targeting that bug's invariant. The bug counts as found only if that
  candidate is CONFIRMED by the deterministic verifier.
* **False Positive Rate** - over every (variant, candidate) pair whose invariant is
  *not* broken by any active bug in that variant: ``FP / (FP + TN)`` where FP means
  the verifier CONFIRMED a violation anyway.

FLAKY and INVALID verdicts are never findings; they count as misses/negatives.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from pydantic import BaseModel, Field

from cubic.gym.base import Environment, Variant
from cubic.verifier.replay import DeterministicVerifier, Verdict, VerificationResult
from cubic.verifier.trace import CandidateFinding


class CaseResult(BaseModel):
    variant: str
    candidate_id: str
    invariant_id: str
    expected_violation: bool
    verdict: Verdict


class Metrics(BaseModel):
    recall_at_1: float | None
    false_positive_rate: float | None
    bugs_total: int
    bugs_found_at_1: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    flaky: int
    invalid: int


class EvalReport(BaseModel):
    environment: str
    runs_per_candidate: int
    variants: list[str]
    metrics: Metrics
    per_variant: dict[str, Metrics]
    cases: list[CaseResult]
    findings: list[VerificationResult] = Field(
        default_factory=list, description="Confirmed findings with evidence and reproduction."
    )
    wall_clock_s: float = 0.0


def compute_metrics(
    cases: Sequence[CaseResult], ground_truth: Sequence[tuple[str, str]]
) -> Metrics:
    """``ground_truth`` lists (variant, invariant_id) pairs that must be found."""
    tp = sum(c.expected_violation and c.verdict is Verdict.CONFIRMED for c in cases)
    fp = sum(not c.expected_violation and c.verdict is Verdict.CONFIRMED for c in cases)
    tn = sum(not c.expected_violation and c.verdict is not Verdict.CONFIRMED for c in cases)
    fn = sum(c.expected_violation and c.verdict is not Verdict.CONFIRMED for c in cases)

    first: dict[tuple[str, str], CaseResult] = {}
    for case in cases:
        first.setdefault((case.variant, case.invariant_id), case)
    found = sum(
        1 for key in ground_truth if key in first and first[key].verdict is Verdict.CONFIRMED
    )
    return Metrics(
        recall_at_1=found / len(ground_truth) if ground_truth else None,
        false_positive_rate=fp / (fp + tn) if fp + tn else None,
        bugs_total=len(ground_truth),
        bugs_found_at_1=found,
        true_positives=tp, false_positives=fp, true_negatives=tn, false_negatives=fn,
        flaky=sum(c.verdict is Verdict.FLAKY for c in cases),
        invalid=sum(c.verdict is Verdict.INVALID for c in cases),
    )


async def arun_eval(
    env: Environment,
    candidates: Sequence[CandidateFinding] | None = None,
    *,
    runs: int = 3,
    variants: Sequence[Variant] | None = None,
) -> EvalReport:
    started = time.perf_counter()
    candidates = tuple(env.baseline_candidates() if candidates is None else candidates)
    variants = tuple(env.variants() if variants is None else variants)

    cases: list[CaseResult] = []
    findings: list[VerificationResult] = []
    ground_truth: list[tuple[str, str]] = []
    per_variant: dict[str, Metrics] = {}

    for variant in variants:
        positives = {env.bug(b).invariant_id for b in variant.active_bugs}
        variant_truth = sorted((variant.name, inv) for inv in positives)
        verifier = DeterministicVerifier(
            env.graph, env.app_factory(variant.active_bugs), env.snapshot, runs=runs
        )
        variant_cases: list[CaseResult] = []
        for candidate in candidates:
            result = await verifier.averify(candidate)
            variant_cases.append(CaseResult(
                variant=variant.name, candidate_id=candidate.id,
                invariant_id=candidate.invariant_id,
                expected_violation=candidate.invariant_id in positives,
                verdict=result.verdict,
            ))
            if result.confirmed:
                findings.append(result)
        per_variant[variant.name] = compute_metrics(variant_cases, variant_truth)
        cases += variant_cases
        ground_truth += variant_truth

    return EvalReport(
        environment=env.name,
        runs_per_candidate=runs,
        variants=[v.name for v in variants],
        metrics=compute_metrics(cases, ground_truth),
        per_variant=per_variant,
        cases=cases,
        findings=findings,
        wall_clock_s=time.perf_counter() - started,
    )


def run_eval(
    env: Environment,
    candidates: Sequence[CandidateFinding] | None = None,
    *,
    runs: int = 3,
    variants: Sequence[Variant] | None = None,
) -> EvalReport:
    return asyncio.run(arun_eval(env, candidates, runs=runs, variants=variants))


def format_report(report: EvalReport) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    m = report.metrics
    lines = [
        f"Cubic eval: environment={report.environment} variants={len(report.variants)} "
        f"runs/candidate={report.runs_per_candidate}",
        "",
        f"{'variant':<28} {'recall@1':>9} {'FPR':>7} {'found':>7} {'FP':>4} {'flaky':>6}",
    ]
    for name, vm in report.per_variant.items():
        lines.append(
            f"{name:<28} {pct(vm.recall_at_1):>9} {pct(vm.false_positive_rate):>7} "
            f"{vm.bugs_found_at_1:>3}/{vm.bugs_total:<3} {vm.false_positives:>4} {vm.flaky:>6}"
        )
    lines += [
        "",
        f"Recall@1             {pct(m.recall_at_1)}  ({m.bugs_found_at_1}/{m.bugs_total} bugs)",
        f"False Positive Rate  {pct(m.false_positive_rate)}  "
        f"({m.false_positives}/{m.false_positives + m.true_negatives} negatives)",
        f"Confirmed findings   {len(report.findings)}   flaky={m.flaky} invalid={m.invalid}",
        f"Wall clock           {report.wall_clock_s:.2f}s",
    ]
    return "\n".join(lines)
