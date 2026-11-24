import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest

from cubic.cli import main
from cubic.eval import autonomous_environment, plan_for, run_eval
from cubic.gym import load_environment
from cubic.gym.fintech import manifest
from cubic.search import LLMGuidedPolicy, RankingPrompt, SearchConfig
from cubic.verifier import Verdict

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "benchmarks/results/search_ablation.json"
PLANTED = {
    "idor_wallet_read": "inv.tenant.read_wallet",
    "race_withdraw": "inv.ledger.conservation",
    "loan_state_skip": "inv.lifecycle.disburse_loan",
    "negative_transfer": "inv.input.transfer.amount",
}


@pytest.fixture(scope="module")
def env():
    return load_environment("fintech-autonomous")


@pytest.fixture(scope="module")
def report(env):
    return run_eval(env, runs=3)


def test_autonomous_env_uses_compiled_graph_and_search_plan(env):
    compiled = load_environment("fintech-compiled")
    assert env.name == "fintech-autonomous"
    assert env.graph == compiled.graph and env.bugs == compiled.bugs
    plan = env.baseline_candidates()
    assert plan and all(c.id.startswith("auto:") for c in plan)
    assert not {c.id for c in plan} & {c.id for c in compiled.baseline_candidates()}


def test_search_never_reads_hand_written_probes(monkeypatch):
    def forbidden():
        raise AssertionError("autonomous search must not use the probe manifest")

    monkeypatch.setattr(manifest, "baseline_candidates", forbidden)
    compiled = dataclasses.replace(load_environment("fintech-compiled"), baseline_candidates=forbidden)
    plan = autonomous_environment(compiled).baseline_candidates()
    assert len(plan) >= len(PLANTED)


def test_autonomous_eval_finds_all_planted_bugs_with_zero_fpr(report, env):
    m = report.metrics
    assert m.recall_at_1 == 1.0
    assert m.bugs_found_at_1 == m.bugs_total == 8  # 4 single-bug builds + 4 in "vulnerable"
    assert m.false_positives == 0 and m.false_positive_rate == 0.0
    assert m.flaky == 0 and m.invalid == 0
    assert len(report.variants) == 6
    assert report.per_variant["patched"].false_positives == 0
    for variant in env.variants():
        assert report.per_variant[variant.name].bugs_found_at_1 == len(variant.active_bugs)


def test_each_planted_bug_is_found_by_its_attack_shape(report):
    confirmed = {(c.variant, c.invariant_id): c.candidate_id
                 for c in report.cases if c.verdict is Verdict.CONFIRMED}
    shapes = {
        "idor_wallet_read": "auto:cross_persona:",
        "race_withdraw": "auto:parallel_contention:",
        "loan_state_skip": "auto:lifecycle_order:",
        "negative_transfer": "auto:boundary_perturb:",
    }
    for bug, invariant in PLANTED.items():
        assert confirmed[(f"only:{bug}", invariant)].startswith(shapes[bug])
        assert confirmed[("vulnerable", invariant)].startswith(shapes[bug])
    assert len(confirmed) == 8
    for finding in report.findings:
        assert finding.reproduction and finding.evidence_db_diff is not None


def test_cli_autonomous_eval_gate(tmp_path, capsys):
    out = tmp_path / "auto.json"
    code = main(["eval", "--env", "fintech-autonomous", "--min-recall", "1.0", "--max-fpr", "0.0",
                 "--json", str(out)])
    assert code == 0
    assert "8/8 bugs" in capsys.readouterr().out
    assert json.loads(out.read_text())["environment"] == "fintech-autonomous"


def test_cli_search_prints_plan_and_coverage(tmp_path, capsys):
    out = tmp_path / "plan.json"
    assert main(["search", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "parallel_contention" in text and "uncovered inv.functional.disburse_loan" in text
    data = json.loads(out.read_text())
    assert data["policy"] == "heuristic" and len(data["emitted"]) == data["stats"]["tests_emitted"]
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(load_environment("fintech-compiled").graph.to_json())
    assert main(["search", "--graph", str(graph_file), "--max-tests", "3"]) == 0
    assert "3 tests covering 3/23 invariants" in capsys.readouterr().out


class _ReverseModel:
    def complete(self, prompt: RankingPrompt) -> str:
        keys = [c["key"] for c in prompt.candidates][::-1]
        return json.dumps({"ranking": [{"key": k} for k in keys]})


def test_llm_guided_plan_cannot_reintroduce_false_positives():
    base = load_environment("fintech-compiled")
    env = autonomous_environment(base, policy=LLMGuidedPolicy(_ReverseModel()),
                                 config=SearchConfig(per_invariant=2), name="fintech-llm")
    patched = [v for v in env.variants() if v.name == "patched"]
    report = run_eval(env, runs=1, variants=patched)
    assert report.metrics.false_positives == 0


# --------------------------------------------------------------------------- ablation benchmark


def _bench():
    spec = importlib.util.spec_from_file_location("ablation_search", ROOT / "benchmarks/ablation_search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ablation_benchmark_small_run():
    result = _bench().run(runs=1, seeds=10, variant_names=("patched", "vulnerable"))
    assert result["variants"] == ["patched", "vulnerable"]
    guided = result["fp_filtering"]["guided_plan"]
    assert guided["unfiltered"]["false_positives"] > 0
    assert guided["verifier_gated"]["false_positives"] == 0
    assert result["fp_filtering"]["full_pool_no_policy"]["verifier_gated"]["false_positives"] > 0
    eff = result["search_efficiency"]
    assert eff["graph_guided"]["recall_at_1"] == 1.0
    assert eff["uniform_random_edge"]["recall_at_1"]["mean"] < 1.0
    assert eff["uniform_random_edge"]["seeds"] == 10


def test_committed_ablation_snapshot_is_consistent():
    data = json.loads(SNAPSHOT.read_text())
    assert data["benchmark"] == "search_ablation" and data["runs_per_candidate"] == 3
    assert len(data["variants"]) == 6
    guided = data["search_efficiency"]["graph_guided"]
    random = data["search_efficiency"]["uniform_random_edge"]
    assert guided["recall_at_1"] == 1.0 and guided["bugs_found_at_1"] == guided["bugs_total"] == 8
    assert guided["false_positive_rate_gated"] == 0.0
    assert random["recall_at_1"]["mean"] < guided["recall_at_1"]
    assert random["tests_to_full_recall_vulnerable"]["mean"] > guided["tests_to_full_recall"]["vulnerable"]
    gated = data["fp_filtering"]["guided_plan"]["verifier_gated"]
    assert gated["false_positives"] == 0 and gated["true_positives"] == 8
    assert data["guided_plan"]["tests"] == [t.id for t in plan_for(load_environment("fintech-compiled")).tests]
