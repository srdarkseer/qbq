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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "eval":
        return _cmd_eval(args)
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
