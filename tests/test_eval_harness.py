import json

import pytest

from cubic.cli import main
from cubic.eval import CaseResult, compute_metrics, format_report, run_eval
from cubic.gym import load_environment
from cubic.verifier import Verdict

ENV = load_environment("fintech")


def _case(variant, cand, inv, expected, verdict):
    return CaseResult(variant=variant, candidate_id=cand, invariant_id=inv,
                      expected_violation=expected, verdict=verdict)


def test_compute_metrics_definitions():
    cases = [
        _case("v", "c1", "inv.a", True, Verdict.NOT_REPRODUCED),  # rank-1 for inv.a misses
        _case("v", "c2", "inv.a", True, Verdict.CONFIRMED),       # later hit: not Recall@1
        _case("v", "c3", "inv.b", True, Verdict.CONFIRMED),
        _case("v", "c4", "inv.x", False, Verdict.CONFIRMED),      # false positive
        _case("v", "c5", "inv.y", False, Verdict.NOT_REPRODUCED),
        _case("v", "c6", "inv.z", False, Verdict.FLAKY),          # discarded -> negative
        _case("v", "c7", "inv.w", False, Verdict.INVALID),
    ]
    m = compute_metrics(cases, [("v", "inv.a"), ("v", "inv.b"), ("v", "inv.c")])
    assert m.recall_at_1 == pytest.approx(1 / 3)  # inv.c had no candidate at all
    assert m.bugs_found_at_1 == 1 and m.bugs_total == 3
    assert (m.true_positives, m.false_positives, m.true_negatives, m.false_negatives) == (2, 1, 3, 1)
    assert m.false_positive_rate == pytest.approx(1 / 4)
    assert (m.flaky, m.invalid) == (1, 1)


def test_compute_metrics_empty():
    m = compute_metrics([], [])
    assert m.recall_at_1 is None and m.false_positive_rate is None


@pytest.fixture(scope="module")
def baseline_report():
    return run_eval(ENV, runs=3)


def test_baseline_meets_milestone_targets(baseline_report):
    m = baseline_report.metrics
    assert m.recall_at_1 == 1.0
    assert m.false_positive_rate == 0.0 < 0.005
    assert m.bugs_total == 2 * len(ENV.bugs)  # each bug: its single variant + "vulnerable"
    assert m.flaky == 0 and m.invalid == 0
    assert baseline_report.per_variant["patched"].recall_at_1 is None
    assert baseline_report.per_variant["patched"].false_positives == 0
    assert len(baseline_report.findings) == m.bugs_total
    assert all(f.reproduction and f.evidence_db_diff is not None for f in baseline_report.findings)


def test_eval_is_deterministic(baseline_report):
    again = run_eval(ENV, runs=3)
    assert again.cases == baseline_report.cases
    assert again.metrics == baseline_report.metrics


def test_recall_at_1_penalizes_wrong_first_guess():
    probes = {c.id: c for c in ENV.baseline_candidates()}
    ordered = [probes["probe.sequential_withdrawals"], probes["probe.withdraw_concurrency_spike"]]
    report = run_eval(ENV, ordered, runs=1, variants=[v for v in ENV.variants() if v.name == "only:race_withdraw"])
    assert report.metrics.recall_at_1 == 0.0
    assert report.metrics.true_positives == 1  # the later candidate still verified


def test_report_serializes_and_formats(baseline_report):
    payload = json.loads(baseline_report.model_dump_json())
    assert payload["metrics"]["recall_at_1"] == 1.0
    text = format_report(baseline_report)
    assert "Recall@1             100.0%" in text and "False Positive Rate  0.0%" in text


def test_cli_eval_writes_json_and_passes_gates(tmp_path, capsys):
    out = tmp_path / "reports" / "eval.json"
    assert main(["eval", "--runs", "1", "--json", str(out), "--min-recall", "1.0", "--max-fpr", "0.005"]) == 0
    assert json.loads(out.read_text())["metrics"]["false_positive_rate"] == 0.0
    assert "Recall@1" in capsys.readouterr().out


def test_cli_eval_gate_failure(capsys):
    assert main(["eval", "--runs", "1", "--max-fpr", "-1"]) == 1
    assert "GATE FAILED" in capsys.readouterr().err


def test_cli_graph_and_schema(capsys):
    assert main(["graph"]) == 0
    assert json.loads(capsys.readouterr().out)["app_name"] == "cubic-fintech-testbed"
    assert main(["schema"]) == 0
    assert "StateNode" in json.loads(capsys.readouterr().out)["$defs"]


def test_unknown_environment():
    with pytest.raises(KeyError, match="unknown environment"):
        load_environment("nope")
