"""Compiler latency and compression benchmark on the fintech testbed.

    uv run python benchmarks/compile_bench.py [--iterations 25] [--out benchmarks/results/compile_fintech.json]

Records per-phase wall-clock latency (median / p95 / min over N compiles) and graph
compression (source bytes vs compiled graph bytes) for later paper evaluation.
Byte counts are deterministic; latencies depend on the machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

from cubic import __version__
from cubic.compiler import compare_graphs, compile_app
from cubic.gym.fintech import create_app
from cubic.gym.fintech.graph import build_graph


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def run(iterations: int) -> dict:
    personas = build_graph().actors
    results = [compile_app(lambda: create_app(), personas=personas) for _ in range(iterations)]
    first = results[0]
    if any(r.graph.to_json() != first.graph.to_json() for r in results):
        raise SystemExit("non-deterministic compile output")
    phases = first.report.timings_s.keys()
    latency = {
        phase: {
            "median_ms": statistics.median(r.report.timings_s[phase] for r in results) * 1000,
            "p95_ms": _pct([r.report.timings_s[phase] for r in results], 0.95) * 1000,
            "min_ms": min(r.report.timings_s[phase] for r in results) * 1000,
        }
        for phase in phases
    }
    rep = first.report
    fidelity = compare_graphs(build_graph(), first.graph)
    return {
        "benchmark": "compile_fintech",
        "cubic_version": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "iterations": iterations,
        "routes": {"total": rep.routes_total, "compiled": rep.routes_compiled, "coverage": rep.route_coverage},
        "graph": {"nodes": rep.nodes, "edges": rep.edges, "invariants": rep.invariants},
        "bytes": {
            "source": rep.source_bytes,
            "openapi": rep.openapi_bytes,
            "compiled_graph": rep.graph_bytes,
            "structural_graph": rep.structural_graph_bytes,
        },
        "compression": {
            "source_over_graph": rep.source_bytes / rep.graph_bytes,
            "source_over_structural_graph": rep.source_bytes / rep.structural_graph_bytes,
            "source_plus_openapi_over_structural_graph":
                (rep.source_bytes + rep.openapi_bytes) / rep.structural_graph_bytes,
        },
        "fidelity": {
            "structural_fidelity": fidelity.structural_fidelity,
            "route_coverage": fidelity.route_coverage,
            "role_agreement": fidelity.role_agreement,
        },
        "latency": latency,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "compile_fintech.json")
    args = parser.parse_args(argv)
    result = run(args.iterations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    total = result["latency"]["total"]
    print(f"compile latency median {total['median_ms']:.1f} ms (p95 {total['p95_ms']:.1f} ms) "
          f"over {args.iterations} runs; source {result['bytes']['source']} B -> graph "
          f"{result['bytes']['compiled_graph']} B; written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
