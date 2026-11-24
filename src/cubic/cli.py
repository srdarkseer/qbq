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
    ev.add_argument("--env", default="fintech", help="Bug Gym environment name (default: fintech)")
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
        synthesize=not args.no_invariants, probe=not args.no_probe,
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
        f"({r.route_coverage:.0%} coverage), {r.nodes} nodes, {r.edges} edges, {r.invariants} invariants\n"
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
