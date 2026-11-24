import asyncio
import json
from pathlib import Path

import pytest

from cubic.cli import main
from cubic.gym import load_environment
from cubic.remediate.patcher import load_context
from cubic.remediate.models import ConfirmedFinding
from cubic.rlvr import (
    REWARDS,
    LeakageError,
    Outcome,
    VerifierReward,
    action_json,
    arecord_environment,
    build_dataset,
    check_leakage,
    corrupt,
    jsonl_lines,
    outcome_for,
    parse_action,
    stratified_sample,
    synthetic_bad_patches,
)
from cubic.rlvr.rewards import ActionError
from cubic.search import search
from cubic.verifier import DeterministicVerifier, Verdict

ROOT = Path(__file__).resolve().parents[1]
FINTECH = load_environment("fintech-compiled")


@pytest.fixture(scope="module")
def records():
    return asyncio.run(arecord_environment("fintech", "train", env=FINTECH, runs=1,
                                           variants=("patched", "vulnerable")))


def test_reward_rules_are_programmatic():
    assert outcome_for(Verdict.CONFIRMED, Verdict.NOT_REPRODUCED) is Outcome.TRUE_VIOLATION
    assert outcome_for(Verdict.CONFIRMED, Verdict.CONFIRMED) is Outcome.FALSE_POSITIVE
    assert outcome_for(Verdict.NOT_REPRODUCED, None) is Outcome.HOLDS
    assert outcome_for(Verdict.FLAKY, None) is Outcome.FLAKY
    assert outcome_for(Verdict.INVALID, None) is Outcome.MALFORMED
    assert {REWARDS[o] for o in Outcome} == {1.0, 0.0, -1.0}
    assert REWARDS[Outcome.HOLDS] == 0.0 and REWARDS[Outcome.PATCH_VERIFIED] == 1.0


def test_parse_action_round_trip_and_rejections():
    candidate = search(FINTECH.graph).candidates()[0]
    text = action_json(candidate)
    parsed = parse_action(f"Sure:\n```json\n{text}\n```", FINTECH.graph, candidate.invariant_id)
    assert parsed.trace == candidate.trace
    with pytest.raises(ActionError):
        parse_action(text, FINTECH.graph, "inv.some_other")
    for kind in ("unknown_edge", "unknown_actor", "missing_path_param", "truncated_json"):
        with pytest.raises(ActionError):
            parse_action(corrupt(text, kind), FINTECH.graph, candidate.invariant_id)


def test_records_cover_tests_malformed_and_patches(records):
    kinds = {(r.task, r.source) for r in records}
    assert kinds == {("test", "pool"), ("test", "malformed"), ("patch", "rule_patch"), ("patch", "synthetic_patch")}
    for r in records:
        assert r.reward == REWARDS[r.outcome]
        assert "tok_" not in json.dumps(r.prompt)  # personas never leak credentials into prompts
    assert all(r.outcome is Outcome.MALFORMED for r in records if r.source == "malformed")
    assert all(r.outcome is Outcome.PATCH_VERIFIED for r in records if r.source == "rule_patch")
    assert all(r.outcome is Outcome.PATCH_REJECTED for r in records if r.source == "synthetic_patch")


def test_differential_rewards_match_ground_truth(records):
    pool = [r for r in records if r.source == "pool"]
    for r in pool:
        if r.outcome is Outcome.TRUE_VIOLATION:
            assert r.labels["ground_truth_violation"] and r.variant == "vulnerable"
    # oracle-ambiguous claims (e.g. an admin reading a wallet) reproduce on the reference build too
    fp = [r for r in pool if r.outcome is Outcome.FALSE_POSITIVE]
    assert fp and all(r.labels["reference_verdict"] == "confirmed" for r in fp)
    assert all(r.labels["policy_pruned"] for r in fp if r.variant == "patched")
    planted = {r.invariant_id for r in pool if r.outcome is Outcome.TRUE_VIOLATION}
    assert planted == {b.invariant_id for b in FINTECH.bugs}


def test_verifier_reward_is_trl_compatible(records):
    sample = [r for r in records if r.task == "test"][:: 25]
    reward = VerifierReward({"fintech": FINTECH}, runs=1)
    rows = [r.to_row() for r in sample]
    scores = reward(
        [row["prompt"] for row in rows], [row["completion"] for row in rows],
        env=[row["env"] for row in rows], variant=[row["variant"] for row in rows],
        invariant_id=[row["invariant_id"] for row in rows], split=[row["split"] for row in rows],
    )
    assert scores == [r.reward for r in sample]
    assert reward([[]], ["not json"], env=["fintech"], variant=["vulnerable"],
                  invariant_id=["inv.tenant.read_wallet"]) == [-1.0]


def test_synthetic_bad_patches_are_rejected_by_the_gate():
    plan = search(FINTECH.graph)
    active = frozenset({"idor_wallet_read"})
    verifier = DeterministicVerifier(FINTECH.graph, FINTECH.app_factory(active), FINTECH.snapshot, runs=1)
    candidate = next(c for c in plan.candidates() if c.invariant_id == "inv.tenant.read_wallet")
    finding = ConfirmedFinding.from_verification(FINTECH.graph, candidate, verifier.verify(candidate))
    rules = [p.rule for p in synthetic_bad_patches(finding, load_context(finding, FINTECH.app_source))]
    assert rules == ["cosmetic", "disable_endpoint"]


def test_jsonl_rows_are_trl_conversational_and_sorted(records):
    lines = jsonl_lines(records)
    row = json.loads(lines[0])
    assert row["prompt"][0]["role"] == "system" and row["completion"][0]["role"] == "assistant"
    assert {"env", "variant", "invariant_id", "reward", "split"} <= row.keys()
    assert lines == jsonl_lines(records)  # stable serialization


def test_leakage_is_rejected(records):
    assert check_leakage(records)["shared_prompts"] == 0
    leaked = [records[0].model_copy(update={"split": "eval"}), *records]
    with pytest.raises(LeakageError):
        check_leakage(leaked)
    with pytest.raises(LeakageError):
        build_dataset({"train": ("fintech",), "eval": ("fintech",)})


def test_stratified_sample_covers_every_outcome(records):
    sample = stratified_sample(records, 1)
    assert {r.outcome for r in sample} == {r.outcome for r in records}
    assert len(sample) == len({(r.split, r.task, r.source, r.outcome) for r in records})


def test_cli_export_rlvr_small(tmp_path, capsys):
    out, sample, stats = tmp_path / "d.jsonl", tmp_path / "s.jsonl", tmp_path / "stats.json"
    code = main(["export-rlvr", "--out", str(out), "--runs", "1", "--variants", "patched,vulnerable",
                 "--no-patches", "--sample-out", str(sample), "--stats", str(stats)])
    assert code == 0
    data = json.loads(stats.read_text())
    assert data["leakage"]["envs_by_split"] == {"eval": ["agent_gateway", "saas"], "train": ["fintech"]}
    assert data["rows"] == len(out.read_text().splitlines())
    assert data["sample"]["rows"] == len(sample.read_text().splitlines())
    assert "trajectories" in capsys.readouterr().out


def test_committed_rlvr_snapshot_and_sample_are_consistent():
    stats = json.loads((ROOT / "benchmarks/results/rlvr_stats.json").read_text())
    sample_path = ROOT / stats["sample"]["path"]
    rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
    assert len(rows) == stats["sample"]["rows"]
    assert stats["runs_per_verification"] == 3 and stats["leakage"]["shared_prompts"] == 0
    assert set(stats["splits"]) == {"train", "eval"}
    assert {row["split"] for row in rows} == {"train", "eval"}
    assert {row["outcome"] for row in rows} >= {"true_violation", "holds", "false_positive", "malformed",
                                                 "patch_verified", "patch_rejected"}
    assert sum(s["rows"] for s in stats["splits"].values()) == stats["rows"]
    for row in rows:
        assert row["reward"] == REWARDS[Outcome(row["outcome"])]
