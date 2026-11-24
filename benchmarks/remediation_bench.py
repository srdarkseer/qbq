"""Closed-loop remediation benchmark across both Bug Gym apps (fintech and SaaS).

    uv run python benchmarks/remediation_bench.py [--runs 3] [--out benchmarks/results/remediation_bench.json]

For each app it records:

* **Detection.** Recall@1 and FPR of the autonomous plan (compile -> search ->
  3-of-3 verifier) over all six builds, with and without persona grounding.
* **Remediation.** For each build with planted bugs (each single-bug build and
  ``vulnerable``), the share of confirmed findings with a patch that passes the
  closed-loop gate (``patch_pass_rate``), how often the first candidate patch
  passes, which rules fixed what, and whether each exported regression test is
  valid (fails on the unpatched build, passes on the patched code).
* **Latency.** Wall-clock time of the full ``cubic remediate`` loop on
  ``vulnerable``, per stage.

Everything except the ``latency_s`` block is deterministic. That block depends on
the machine, which is recorded with it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

from cubic import __version__
from cubic.eval import arun_eval, autonomous_environment
from cubic.gym import load_environment
from cubic.gym.base import Environment
from cubic.remediate import RemediationReport, aremediate
from cubic.search import HeuristicGraphPolicy

APPS = {"fintech": "fintech-compiled", "saas": "saas-compiled"}


def _detection(report) -> dict:
    m = report.metrics
    return {
        "recall_at_1": m.recall_at_1, "bugs_found_at_1": m.bugs_found_at_1, "bugs_total": m.bugs_total,
        "false_positive_rate": m.false_positive_rate, "false_positives": m.false_positives,
        "negatives": m.false_positives + m.true_negatives, "flaky": m.flaky,
    }


def _remediation(report: RemediationReport) -> dict:
    findings = report.findings
    return {
        "findings": len(findings),
        "verified_patches": report.verified,
        "patch_pass_rate": report.patch_pass_rate,
        "first_attempt_pass": sum(bool(f.attempts) and f.attempts[0].verified for f in findings),
        "patch_attempts": sum(len(f.attempts) for f in findings),
        "regression_tests_valid": sum(bool(f.regression_test and f.regression_test.valid) for f in findings),
        "fixes": {
            f.invariant_id: {
                "oracle": f.oracle.value,
                "rule": f"{f.verified_patch.patcher}:{f.verified_patch.rule}" if f.verified_patch else None,
                "source": f.source,
                "exploit_violations_after": f.attempts[-1].exploit_violations if f.attempts else None,
                "functional_checks": len(f.attempts[-1].functional) if f.attempts else 0,
                "regressions": f.attempts[-1].regressions if f.attempts else [],
                "diff_lines": sum(
                    1 for line in (f.verified_patch.diff if f.verified_patch else "").splitlines()
                    if line[:1] in {"+", "-"} and not line.startswith(("+++", "---"))
                ),
            }
            for f in findings
        },
    }


async def bench_app(env: Environment, runs: int) -> tuple[dict, dict]:
    grounded = await arun_eval(autonomous_environment(env), runs=runs)
    ungrounded = await arun_eval(
        autonomous_environment(env, policy=HeuristicGraphPolicy(use_observed_state=False)), runs=runs)

    remediation: dict[str, dict] = {}
    rule_counts: Counter[str] = Counter()
    latency: dict[str, float] = {}
    for variant in env.variants():
        if not variant.active_bugs:
            continue
        report = await aremediate(env, variant=variant.name, runs=runs)
        remediation[variant.name] = _remediation(report)
        rule_counts.update(fix["rule"] for fix in remediation[variant.name]["fixes"].values() if fix["rule"])
        if variant.name == "vulnerable":
            latency = {k: round(v, 4) for k, v in report.timings_s.items()}
            plan = {"plan_tests": report.plan_tests, "invariants_covered": report.invariants_covered,
                    "graph": report.graph}

    totals = {
        "findings": sum(r["findings"] for r in remediation.values()),
        "verified_patches": sum(r["verified_patches"] for r in remediation.values()),
        "regression_tests_valid": sum(r["regression_tests_valid"] for r in remediation.values()),
    }
    totals["patch_pass_rate"] = (totals["verified_patches"] / totals["findings"]) if totals["findings"] else None
    return {
        "environment": env.name,
        **plan,
        "detection": {"grounded": _detection(grounded), "ungrounded_m3_policy": _detection(ungrounded)},
        "remediation": {"totals": totals, "rules": dict(sorted(rule_counts.items())), "by_variant": remediation},
    }, latency


def run(runs: int = 3, apps_to_run: tuple[str, ...] = tuple(APPS)) -> dict:
    started = time.perf_counter()
    apps, latency = {}, {}
    for app in apps_to_run:
        env = load_environment(APPS[app])  # compiles synchronously, outside the event loop
        apps[app], latency[app] = asyncio.run(bench_app(env, runs))
    findings = sum(a["remediation"]["totals"]["findings"] for a in apps.values())
    verified = sum(a["remediation"]["totals"]["verified_patches"] for a in apps.values())
    return {
        "benchmark": "remediation",
        "cubic_version": __version__,
        "runs_per_candidate": runs,
        "summary": {
            "apps": len(apps),
            "recall_at_1": {k: a["detection"]["grounded"]["recall_at_1"] for k, a in apps.items()},
            "false_positive_rate": {k: a["detection"]["grounded"]["false_positive_rate"] for k, a in apps.items()},
            "findings_remediated": findings,
            "verified_patches": verified,
            "patch_pass_rate": verified / findings if findings else None,
        },
        "apps": apps,
        "latency_s": {
            "platform": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "remediate_vulnerable": latency,
            "benchmark_total": round(time.perf_counter() - started, 2),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "remediation_bench.json")
    args = parser.parse_args(argv)
    result = run(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    s = result["summary"]
    lat = result["latency_s"]["remediate_vulnerable"]
    print(
        f"Recall@1 {s['recall_at_1']}  FPR {s['false_positive_rate']}\n"
        f"patches verified {s['verified_patches']}/{s['findings_remediated']} ({s['patch_pass_rate']:.0%})\n"
        f"end-to-end remediate (vulnerable): "
        + ", ".join(f"{app} {t['total']:.2f}s" for app, t in lat.items())
        + f"\nwritten to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
