"""Command-line entry point for Cubic."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from cubic import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cubic", description="QbQ (Cubic) QA & security engine")
    parser.add_argument("--version", action="version", version=f"cubic {__version__}")
    sub = parser.add_subparsers(dest="command")

    ev = sub.add_parser("eval", help="Run the deterministic eval harness against a Bug Gym environment")
    ev.add_argument("--env", default="fintech",
                    help="Bug Gym environment: fintech, fintech-compiled, fintech-autonomous, saas-compiled, "
                         "saas-autonomous (default: fintech)")
    ev.add_argument("--runs", type=int, default=3, help="Replays per candidate (default: 3)")
    ev.add_argument("--json", type=Path, metavar="PATH", help="Write the full JSON report to PATH")
    ev.add_argument("--min-recall", type=float, default=None, help="Fail if Recall@1 is below this")
    ev.add_argument("--max-fpr", type=float, default=None, help="Fail if the FPR is above this")

    co = sub.add_parser("compile", help="Compile an ASGI app (source + runtime) into a State Graph")
    co.add_argument("target", help="pkg.module[:attr] or path/to/app.py[:attr] (an app or app factory)")
    co.add_argument("--out", type=Path, metavar="PATH", help="Write graph JSON here (default: stdout)")
    co.add_argument("--personas", type=Path, metavar="PATH",
                    help="JSON list of ActorPersona objects, or a StateGraph JSON to take actors from")
    co.add_argument("--report", type=Path, metavar="PATH", help="Write the compile report JSON here")
    co.add_argument("--app-name", default=None, help="Override the graph's app_name")
    co.add_argument("--no-invariants", action="store_true", help="Skip invariant synthesis")
    co.add_argument("--no-probe", action="store_true", help="Skip unauthenticated runtime probes")
    co.add_argument("--no-ground", action="store_true",
                    help="Skip persona grounding (authenticated reads of each persona's own resources)")

    se = sub.add_parser("search", help="Synthesize a prioritized test plan from an environment's State Graph")
    se.add_argument("--env", default="fintech-compiled", help="Bug Gym environment (default: fintech-compiled)")
    se.add_argument("--graph", type=Path, metavar="PATH", help="Search a StateGraph JSON file instead")
    se.add_argument("--per-invariant", type=int, default=1, help="Tests per invariant (default: 1)")
    se.add_argument("--beam-width", type=int, default=3, help="Tests kept per frame (default: 3)")
    se.add_argument("--max-tests", type=int, default=None, help="Global test budget")
    se.add_argument("--json", type=Path, metavar="PATH", help="Write the full search result JSON to PATH")

    re_ = sub.add_parser(
        "remediate",
        help="Closed loop: compile, search, verify findings, synthesize + verify patches, export regression tests",
    )
    re_.add_argument("--env", default="fintech",
                     help="Bug Gym app: fintech or saas (or any environment with an app source)")
    re_.add_argument("--variant", default="vulnerable", help="Build to remediate (default: vulnerable)")
    re_.add_argument("--runs", type=int, default=3, help="Replays per exploit / fix check (default: 3)")
    re_.add_argument("--out-dir", type=Path, default=Path("regression_tests"),
                     help="Where to write test_regression_<invariant>.py files (default: regression_tests)")
    re_.add_argument("--patch-dir", type=Path, metavar="DIR", help="Also write each verified patch as a .patch file")
    re_.add_argument("--json", type=Path, metavar="PATH", help="Write the full remediation report JSON to PATH")
    re_.add_argument("--require-verified", action="store_true",
                     help="Exit non-zero unless every finding has a verified patch")

    rl = sub.add_parser("export-rlvr", help="Export verifiable-reward trajectories (JSONL) for GRPO/RLVR training")
    rl.add_argument("--out", type=Path, default=Path("datasets/cubic_rlvr.jsonl"),
                    help="Full dataset path (default: datasets/cubic_rlvr.jsonl)")
    rl.add_argument("--train", default="fintech", help="Comma-separated train apps (default: fintech)")
    rl.add_argument("--eval", dest="eval_apps", default="saas",
                    help="Comma-separated held-out eval apps (default: saas)")
    rl.add_argument("--runs", type=int, default=3, help="Replays per verification (default: 3)")
    rl.add_argument("--variants", default=None, help="Comma-separated build variants (default: all)")
    rl.add_argument("--no-patches", action="store_true", help="Skip patch-verification trajectories")
    rl.add_argument("--sample-out", type=Path, metavar="PATH", help="Also write a stratified sample JSONL")
    rl.add_argument("--sample-per-stratum", type=int, default=2)
    rl.add_argument("--stats", type=Path, metavar="PATH", help="Write dataset statistics JSON")

    gr = sub.add_parser("graph", help="Print an environment's Behavioral State Graph as JSON")
    gr.add_argument("--env", default="fintech")

    sub.add_parser("schema", help="Print the State Graph JSON Schema")
    return parser


def _cmd_eval(args: argparse.Namespace) -> int:
    from cubic.eval import format_report, run_eval
    from cubic.gym import load_environment

    report = run_eval(load_environment(args.env), runs=args.runs)
    print(format_report(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(report.model_dump_json(indent=2))
        print(f"\nReport written to {args.json}")

    m, failures = report.metrics, []
    if args.min_recall is not None and (m.recall_at_1 or 0.0) < args.min_recall:
        failures.append(f"Recall@1 {m.recall_at_1} < {args.min_recall}")
    if args.max_fpr is not None and (m.false_positive_rate or 0.0) > args.max_fpr:
        failures.append(f"FPR {m.false_positive_rate} > {args.max_fpr}")
    for failure in failures:
        print(f"GATE FAILED: {failure}", file=sys.stderr)
    return 1 if failures else 0


def _cmd_search(args: argparse.Namespace) -> int:
    from cubic.graph import StateGraph
    from cubic.gym import load_environment
    from cubic.search import SearchConfig, search

    graph = (StateGraph.from_json(args.graph.read_text()) if args.graph
             else load_environment(args.env).graph)
    config = SearchConfig(beam_width=args.beam_width, per_invariant=args.per_invariant,
                          max_tests=args.max_tests)
    result = search(graph, config=config)
    for e in result.emitted:
        print(f"{e.rank:>3}  {e.priority:6.2f}  {e.test.combinator.value:<19} {e.test.invariant_id:<34} "
              f"{e.test.hypothesis}")
    covered = sum(1 for c in result.coverage.values() if c.tests)
    print(f"\n{len(result.tests)} tests covering {covered}/{len(result.coverage)} invariants "
          f"(policy={result.policy}, {result.stats.tests_generated} generated, "
          f"{result.stats.tests_pruned} pruned)")
    for inv_id, reason in result.uncovered().items():
        print(f"  uncovered {inv_id}: {reason}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(result.model_dump_json(indent=2) + "\n")
        print(f"\nSearch result written to {args.json}")
    return 0


def _cmd_remediate(args: argparse.Namespace) -> int:
    import re

    from cubic.gym import ENVIRONMENTS, load_environment
    from cubic.remediate import remediate

    name = args.env if args.env in ENVIRONMENTS else f"{args.env}-compiled"
    env = load_environment(name)
    if env.app_source is None and f"{args.env}-compiled" in ENVIRONMENTS:
        env = load_environment(f"{args.env}-compiled")
    report = remediate(env, variant=args.variant, runs=args.runs, out_dir=args.out_dir)

    print(f"Cubic remediate: environment={report.environment} variant={report.variant} "
          f"runs={report.runs} plan={report.plan_tests} tests "
          f"({report.invariants_covered}/{report.graph['invariants']} invariants)\n")
    print(f"{'invariant':<36} {'oracle':<20} {'patch':<26} {'verdict':<22} regression test")
    for f in report.findings:
        last = f.attempts[-1] if f.attempts else None
        rule = f"{f.verified_patch.patcher}:{f.verified_patch.rule}" if f.verified_patch else "-"
        verdict = last.verdict.value if last else (f.error or "no patch proposed")
        test = f.regression_test
        test_note = (f"{test.file} ({'valid' if test.valid else 'INVALID'})" if test else "-")
        print(f"{f.invariant_id:<36} {f.oracle.value:<20} {rule:<26} {verdict:<22} {test_note}")
    rate = report.patch_pass_rate
    print(f"\n{report.verified}/{len(report.findings)} findings have a verified patch"
          + (f" ({rate:.0%})" if rate is not None else "")
          + f"; regression tests in {args.out_dir}/; total {report.timings_s['total']:.2f}s")

    if args.patch_dir:
        args.patch_dir.mkdir(parents=True, exist_ok=True)
        for f in report.findings:
            if f.verified_patch:
                slug = re.sub(r"[^A-Za-z0-9]+", "_", f.invariant_id).strip("_")
                (args.patch_dir / f"{slug}.patch").write_text(f.verified_patch.diff)
        print(f"Verified patches written to {args.patch_dir}/")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(report.model_dump_json(indent=2) + "\n")
        print(f"Report written to {args.json}")
    if args.require_verified and report.verified < len(report.findings):
        print("GATE FAILED: some findings have no verified patch", file=sys.stderr)
        return 1
    return 0


def _cmd_export_rlvr(args: argparse.Namespace) -> int:
    from cubic.rlvr import build_dataset, dataset_stats, stratified_sample, write_jsonl

    def apps(text: str) -> tuple[str, ...]:
        return tuple(a.strip() for a in text.split(",") if a.strip())

    splits = {"train": apps(args.train), "eval": apps(args.eval_apps)}
    variants = apps(args.variants) if args.variants else None
    records = build_dataset(splits, runs=args.runs, variants=variants, include_patches=not args.no_patches)
    digest = write_jsonl(records, args.out)
    stats = dataset_stats(records, sha256=digest, runs=args.runs)
    print(f"{len(records)} trajectories -> {args.out} (sha256 {digest[:12]})")
    for split, summary in stats["splits"].items():
        print(f"  {split:<5} {','.join(summary['envs']):<16} rows={summary['rows']:<5} "
              f"prompts={summary['prompts']:<4} rewards={summary['reward_histogram']} "
              f"mean={summary['mean_reward']}")
    if args.sample_out:
        sample = stratified_sample(records, args.sample_per_stratum)
        stats["sample"] = {"path": args.sample_out.as_posix(), "rows": len(sample),
                           "per_stratum": args.sample_per_stratum,
                           "sha256": write_jsonl(sample, args.sample_out)}
        print(f"  sample: {len(sample)} rows -> {args.sample_out}")
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"  stats -> {args.stats}")
    return 0


def _load_personas(path: Path) -> list:
    from cubic.graph import ActorPersona

    data = json.loads(path.read_text())
    items = data.get("actors", []) if isinstance(data, dict) else data
    return [ActorPersona.model_validate(item) for item in items]


def _cmd_compile(args: argparse.Namespace) -> int:
    from cubic.compiler import TargetError, compile_app, load_target

    try:
        factory = load_target(args.target)
    except TargetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    personas = _load_personas(args.personas) if args.personas else []
    result = compile_app(
        factory, app_name=args.app_name, personas=personas,
        synthesize=not args.no_invariants, probe=not args.no_probe, ground=not args.no_ground,
    )
    r = result.report
    graph_json = result.graph.to_json()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(graph_json + "\n")
    else:
        print(graph_json)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(r.model_dump_json(indent=2) + "\n")
    summary = sys.stdout if args.out else sys.stderr
    print(
        f"compiled {r.app_name}: {r.routes_compiled}/{r.routes_total} routes "
        f"({r.route_coverage:.0%} coverage), {r.nodes} nodes, {r.edges} edges, {r.invariants} invariants, "
        f"{r.grounded_records} grounded records\n"
        f"  latency {r.timings_s['total'] * 1000:.1f} ms | source {r.source_bytes} B -> graph "
        f"{r.graph_bytes} B (structural {r.structural_graph_bytes} B)",
        file=summary,
    )
    if r.dynamic_only_routes:
        print(f"  dynamic-only routes: {', '.join(r.dynamic_only_routes)}", file=summary)
    if args.out:
        print(f"  graph written to {args.out}", file=summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "eval":
        return _cmd_eval(args)
    if args.command == "compile":
        return _cmd_compile(args)
    if args.command == "search":
        return _cmd_search(args)
    if args.command == "remediate":
        return _cmd_remediate(args)
    if args.command == "export-rlvr":
        return _cmd_export_rlvr(args)
    if args.command == "graph":
        from cubic.gym import load_environment

        print(load_environment(args.env).graph.to_json())
        return 0
    if args.command == "schema":
        from cubic.graph import StateGraph

        print(json.dumps(StateGraph.json_schema(), indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
