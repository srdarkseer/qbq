"""Milestone 2 end-to-end: compile the fintech testbed and drive the M1 verifier with it."""

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.routing import APIRoute

from qbq.cli import main
from qbq.compiler import compare_graphs, compile_app
from qbq.eval import run_eval
from qbq.graph import StateGraph
from qbq.gym import load_environment
from qbq.gym.fintech import BUGS, create_app
from qbq.gym.fintech.compiled import BUG_TO_SYNTHESIZED_INVARIANT, compiled_graph
from qbq.gym.fintech.graph import build_graph
from qbq.verifier import DeterministicVerifier, Verdict

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = build_graph().actors


@pytest.fixture(scope="module")
def compiled():
    return compile_app(lambda: create_app(), personas=PERSONAS)


# -- fidelity -----------------------------------------------------------------------

def test_compiled_graph_covers_every_live_route(compiled):
    live = {
        (m, r.path) for r in create_app().routes
        if isinstance(r, APIRoute) and r.include_in_schema for m in r.methods
    }
    assert {(e.method.value, e.path) for e in compiled.graph.edges} == live
    assert compiled.report.route_coverage == 1.0
    assert compiled.report.static_matched == compiled.report.static_routes == len(live)
    assert compiled.report.dynamic_only_routes == compiled.report.static_unmatched_routes == []


def test_structural_fidelity_matches_hand_written_graph(compiled):
    fidelity = compare_graphs(build_graph(), compiled.graph)
    assert fidelity.route_coverage == 1.0 and fidelity.missing_routes == [] == fidelity.extra_routes
    assert fidelity.auth_agreement == fidelity.path_param_agreement == 1.0
    assert fidelity.payload_presence_agreement == fidelity.lifecycle_agreement == 1.0
    assert fidelity.structural_fidelity == 1.0
    # Roles are policy: the code lets ADMIN call wallet/loan endpoints the reference reserves for USER.
    assert all("compiled ['ADMIN', 'USER']" in m for m in fidelity.role_mismatches)


def test_fidelity_detects_regressions(compiled):
    graph = compiled.graph
    broken_edges = tuple(
        e.model_copy(update={"auth_required": False, "allowed_roles": frozenset(), "required_headers": frozenset()})
        if e.id == "edge.withdraw" else e
        for e in graph.edges if e.id != "edge.health"
    )
    broken = graph.model_copy(update={"edges": broken_edges, "invariants": ()})
    fidelity = compare_graphs(build_graph(), broken)
    assert fidelity.missing_routes == ["GET /health"]
    assert fidelity.auth_agreement < 1.0 and fidelity.structural_fidelity < 1.0


# -- verifier integration ------------------------------------------------------------

def test_every_bug_maps_to_a_synthesized_invariant():
    graph = compiled_graph()
    assert set(BUG_TO_SYNTHESIZED_INVARIANT) == {b.id for b in BUGS}
    for invariant_id in BUG_TO_SYNTHESIZED_INVARIANT.values():
        graph.invariant(invariant_id)


@pytest.fixture(scope="module")
def compiled_report():
    return run_eval(load_environment("fintech-compiled"), runs=3)


def test_m1_verifier_runs_cleanly_on_compiled_graph(compiled_report):
    m = compiled_report.metrics
    assert m.recall_at_1 == 1.0 and m.false_positive_rate == 0.0
    assert m.invalid == 0 and m.flaky == 0
    assert m.bugs_total == 2 * len(BUGS)
    assert m.true_negatives > run_eval(load_environment("fintech"), runs=1).metrics.true_negatives


def test_compiled_findings_carry_replayable_evidence(compiled_report):
    edge_ids = {e.id for e in compiled_graph().edges}
    for finding in compiled_report.findings:
        assert finding.reproduction and finding.evidence_db_diff is not None
        assert {ex.edge_id for ex in finding.evidence_exchanges} <= edge_ids


def test_compiled_eval_matches_hand_written_eval(compiled_report):
    hand = run_eval(load_environment("fintech"), runs=3)

    def confirmed(report):
        return {(c.variant, c.candidate_id) for c in report.cases if c.verdict is Verdict.CONFIRMED}

    assert confirmed(compiled_report) == confirmed(hand)


def test_synthesized_invariants_hold_on_patched_build(compiled):
    """Each FPR control exercising a synthesized invariant holds on the patched app."""
    env = load_environment("fintech-compiled")
    verifier = DeterministicVerifier(env.graph, env.app_factory(()), env.snapshot, runs=1)
    for candidate in env.baseline_candidates():
        assert verifier.verify(candidate).verdict is Verdict.NOT_REPRODUCED, candidate.id


# -- latency & compression ------------------------------------------------------------

def test_compile_latency_and_compression_metrics(compiled):
    r = compiled.report
    assert r.timings_s["total"] < 2.0  # generous CI budget; typical ~30-80 ms
    assert set(r.timings_s) == {"dynamic", "static", "merge_and_synthesis", "grounding", "total"}
    assert r.grounded_records == 3  # one wallet per funded or unfunded user persona
    assert r.graph_bytes == len(compiled.graph.to_json(indent=None).encode())
    assert r.source_bytes == len((ROOT / "src/qbq/gym/fintech/app.py").read_bytes())
    assert r.structural_graph_bytes < r.source_bytes  # structure compresses source
    assert r.compression_ratio == pytest.approx(r.source_bytes / r.graph_bytes)


def test_benchmark_script(tmp_path):
    spec = importlib.util.spec_from_file_location("compile_bench", ROOT / "benchmarks/compile_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    out = tmp_path / "bench.json"
    assert bench.main(["--iterations", "2", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["routes"]["coverage"] == 1.0 and data["fidelity"]["structural_fidelity"] == 1.0
    assert data["bytes"]["source"] > 0 and data["latency"]["total"]["median_ms"] > 0


# -- CLI --------------------------------------------------------------------------------

def _personas_file(tmp_path, as_graph: bool) -> Path:
    path = tmp_path / ("graph.json" if as_graph else "personas.json")
    if as_graph:
        path.write_text(build_graph().to_json())
    else:
        path.write_text(json.dumps([a.model_dump(mode="json") for a in PERSONAS]))
    return path


@pytest.mark.parametrize("as_graph", [False, True])
def test_cli_compile_writes_byte_identical_graph(tmp_path, capsys, compiled, as_graph):
    out, report = tmp_path / "out" / "graph.json", tmp_path / "report.json"
    code = main(["compile", "qbq.gym.fintech.app:create_app", "--out", str(out),
                 "--personas", str(_personas_file(tmp_path, as_graph)), "--report", str(report)])
    assert code == 0
    assert out.read_text() == compiled.graph.to_json() + "\n"
    graph = StateGraph.from_json(out.read_text())
    assert len(graph.actors) == len(PERSONAS) and len(graph.invariants) == 23
    assert json.loads(report.read_text())["route_coverage"] == 1.0
    stdout = capsys.readouterr().out
    assert "7/7 routes (100% coverage)" in stdout and "latency" in stdout


def test_cli_compile_file_target_to_stdout(capsys):
    assert main(["compile", "src/qbq/gym/fintech/app.py", "--no-invariants", "--no-probe"]) == 0
    captured = capsys.readouterr()
    graph = StateGraph.from_json(captured.out)
    assert graph.invariants == () and graph.actors == ()
    assert graph.edge("edge.health").metadata["evidence"] == ["openapi", "static"]
    assert "compiled qbq-fintech-testbed" in captured.err


def test_cli_compile_bad_target(capsys):
    assert main(["compile", "nope/missing.py"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_cli_eval_on_compiled_environment(capsys):
    assert main(["eval", "--env", "fintech-compiled", "--runs", "1",
                 "--min-recall", "1.0", "--max-fpr", "0.005"]) == 0
    assert "environment=fintech-compiled" in capsys.readouterr().out
