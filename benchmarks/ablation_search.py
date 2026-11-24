"""Search ablation on the fintech testbed's compiled State Graph.

    uv run python benchmarks/ablation_search.py [--runs 3] [--seeds 200] [--out benchmarks/results/search_ablation.json]

Two ablations, both scored against Bug Gym ground truth over every build variant:

1. **False-positive filtering.** Every proposal is reported as a finding
   (*unfiltered*), compared with only the proposals the 3-of-3 replay verifier
   confirms (*verifier-gated*). This is measured for the guided search plan
   and for the full combinator pool, which has no policy pruning.
2. **Search guidance.** Uniform random edge sampling over the full pool,
   compared with State-Graph-guided prioritized search. The random baseline
   gets the same test budget as the guided plan, averaged over ``--seeds``
   orderings. Metrics are Recall@1 and candidate efficiency: how many tests
   must be verified, in submission order, before every active bug has a
   confirmed finding.

Every pool test is verified once per variant. The orderings are then scored
from that verdict cache. The verifier is deterministic, so the snapshot is
byte-identical across runs. Wall-clock time is printed but not recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from cubic import __version__
from cubic.eval import CaseResult, compute_metrics
from cubic.gym import load_environment
from cubic.gym.base import Environment, Variant
from cubic.search import ProposedTest, generate_pool, search, uniform_edge_sample
from cubic.verifier import DeterministicVerifier, Verdict

Verdicts = dict[tuple[str, str], Verdict]  # (variant, test id) -> verdict


async def _verify_all(
    env: Environment, tests: Sequence[ProposedTest], variants: Sequence[Variant], runs: int
) -> Verdicts:
    out: Verdicts = {}
    for variant in variants:
        verifier = DeterministicVerifier(env.graph, env.app_factory(variant.active_bugs), env.snapshot, runs=runs)
        for test in tests:
            out[(variant.name, test.id)] = (await verifier.averify(test.to_candidate())).verdict
    return out


def _positives(env: Environment, variant: Variant) -> set[str]:
    return {env.bug(b).invariant_id for b in variant.active_bugs}


def _cases(env: Environment, order: Sequence[ProposedTest], variants: Sequence[Variant],
           verdicts: Verdicts) -> list[CaseResult]:
    return [
        CaseResult(variant=v.name, candidate_id=t.id, invariant_id=t.invariant_id,
                   expected_violation=t.invariant_id in _positives(env, v),
                   verdict=verdicts[(v.name, t.id)])
        for v in variants for t in order
    ]


def _truth(env: Environment, variants: Sequence[Variant]) -> list[tuple[str, str]]:
    return sorted((v.name, inv) for v in variants for inv in _positives(env, v))


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def fp_filtering(env: Environment, order: Sequence[ProposedTest], variants: Sequence[Variant],
                 verdicts: Verdicts) -> dict:
    """Unfiltered (every proposal reported) vs. verifier-gated (only CONFIRMED reported)."""
    cases = _cases(env, order, variants, verdicts)
    positives = sum(c.expected_violation for c in cases)
    negatives = len(cases) - positives

    def summarize(reported: list[CaseResult]) -> dict:
        tp = sum(c.expected_violation for c in reported)
        fp = len(reported) - tp
        return {
            "reported_findings": len(reported),
            "true_positives": tp,
            "false_positives": fp,
            "precision": _ratio(tp, len(reported)),
            "false_positive_rate": _ratio(fp, negatives),
            "false_discovery_rate": _ratio(fp, len(reported)),
        }

    gated = [c for c in cases if c.verdict is Verdict.CONFIRMED]
    return {
        "proposals": len(order),
        "variant_proposal_pairs": len(cases),
        "pairs_targeting_broken_invariant": positives,
        "pairs_targeting_intact_invariant": negatives,
        "unfiltered": summarize(cases),
        "verifier_gated": summarize(gated) | {
            "filtered_out": len(cases) - len(gated),
            "flaky_discarded": sum(c.verdict is Verdict.FLAKY for c in cases),
        },
    }


def tests_to_full_recall(env: Environment, order: Sequence[ProposedTest], variant: Variant,
                         verdicts: Verdicts) -> int | None:
    """Tests verified, in order, until every active bug has a confirmed finding."""
    missing = _positives(env, variant)
    if not missing:
        return 0
    for index, test in enumerate(order, start=1):
        if test.invariant_id in missing and verdicts[(variant.name, test.id)] is Verdict.CONFIRMED:
            missing.discard(test.invariant_id)
            if not missing:
                return index
    return None


def efficiency(env: Environment, order: Sequence[ProposedTest], variants: Sequence[Variant],
               verdicts: Verdicts, budget: int) -> dict:
    cases = _cases(env, order[:budget], variants, verdicts)
    m = compute_metrics(cases, _truth(env, variants))
    to_full = {v.name: tests_to_full_recall(env, order, v, verdicts)
               for v in variants if v.active_bugs}
    return {
        "budget": budget,
        "recall_at_1": m.recall_at_1,
        "bugs_found_at_1": m.bugs_found_at_1,
        "bugs_total": m.bugs_total,
        "false_positive_rate_gated": m.false_positive_rate,
        "false_positives_gated": m.false_positives,
        "tests_to_full_recall": to_full,
    }


def _dist(values: Sequence[float]) -> dict:
    return {
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def run(runs: int = 3, seeds: int = 200, variant_names: Sequence[str] | None = None) -> dict:
    env = load_environment("fintech-compiled")
    graph = env.graph
    variants = tuple(v for v in env.variants() if variant_names is None or v.name in variant_names)
    pool = generate_pool(graph)
    plan = search(graph)
    guided = plan.tests
    pool_ids = {t.id for t in pool}
    if not all(t.id in pool_ids for t in guided):
        raise SystemExit("guided plan contains tests outside the combinator pool")

    verdicts = asyncio.run(_verify_all(env, pool, variants, runs))
    budget = len(guided)

    random_runs = [
        efficiency(env, uniform_edge_sample(pool, seed), variants, verdicts, budget) for seed in range(seeds)
    ]
    full = [r["tests_to_full_recall"]["vulnerable"] for r in random_runs
            if r["tests_to_full_recall"].get("vulnerable") is not None]
    recalls = [r["recall_at_1"] for r in random_runs]
    fprs = [r["false_positive_rate_gated"] for r in random_runs if r["false_positive_rate_gated"] is not None]
    guided_eff = efficiency(env, guided, variants, verdicts, budget)

    return {
        "benchmark": "search_ablation",
        "cubic_version": __version__,
        "environment": env.name,
        "variants": [v.name for v in variants],
        "runs_per_candidate": runs,
        "graph": {"nodes": len(graph.nodes), "edges": len(graph.edges), "actors": len(graph.actors),
                  "invariants": len(graph.invariants)},
        "pool": {
            "size": len(pool),
            "by_combinator": dict(sorted(Counter(t.combinator.value for t in pool).items())),
        },
        "guided_plan": {
            "policy": plan.policy,
            "config": plan.config,
            "tests": [t.id for t in guided],
            "invariants_covered": sum(1 for c in plan.coverage.values() if c.tests),
            "coverage_ratio": round(plan.coverage_ratio, 6),
            "uncovered": plan.uncovered(),
            "stats": plan.stats.model_dump(),
        },
        "fp_filtering": {
            "guided_plan": fp_filtering(env, guided, variants, verdicts),
            "full_pool_no_policy": fp_filtering(env, pool, variants, verdicts),
        },
        "search_efficiency": {
            "graph_guided": guided_eff,
            "uniform_random_edge": {
                "seeds": seeds,
                "budget": budget,
                "recall_at_1": _dist(recalls),
                "runs_with_full_recall_at_1": sum(r == 1.0 for r in recalls),
                "false_positive_rate_gated": _dist(fprs) if fprs else None,
                "runs_with_any_false_positive": sum(r["false_positives_gated"] > 0 for r in random_runs),
                "tests_to_full_recall_vulnerable": _dist(full) if full else None,
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3, help="Replays per candidate (default: 3)")
    parser.add_argument("--seeds", type=int, default=200, help="Random-baseline orderings (default: 200)")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "search_ablation.json")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    result = run(args.runs, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")

    fp, eff = result["fp_filtering"], result["search_efficiency"]
    g, r = eff["graph_guided"], eff["uniform_random_edge"]
    print(
        f"pool {result['pool']['size']} tests; guided plan {len(result['guided_plan']['tests'])} tests\n"
        f"FP filtering (guided):  unfiltered FDR {fp['guided_plan']['unfiltered']['false_discovery_rate']} "
        f"-> gated FPR {fp['guided_plan']['verifier_gated']['false_positive_rate']}\n"
        f"FP filtering (pool):    unfiltered FDR {fp['full_pool_no_policy']['unfiltered']['false_discovery_rate']} "
        f"-> gated FPR {fp['full_pool_no_policy']['verifier_gated']['false_positive_rate']}\n"
        f"Recall@1 guided {g['recall_at_1']} vs random mean {r['recall_at_1']['mean']} "
        f"(budget {g['budget']}, {r['seeds']} seeds)\n"
        f"tests to full recall (vulnerable): guided {g['tests_to_full_recall'].get('vulnerable')} "
        f"vs random mean {(r['tests_to_full_recall_vulnerable'] or {}).get('mean')}\n"
        f"written to {args.out} in {time.perf_counter() - started:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
