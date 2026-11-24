import itertools

import pytest
from fastapi import FastAPI

from qbq.graph import (
    ActorPersona,
    InvariantAssertion,
    LedgerConservationCheck,
    NodeKind,
    StateGraph,
    StateGuardCheck,
    StateNode,
    StatusCodeCheck,
    TransitionEdge,
)
from qbq.gym import load_environment
from qbq.verifier import (
    ActionStep,
    ActionTrace,
    CandidateFinding,
    DbSnapshot,
    DeterministicVerifier,
    Exchange,
    Verdict,
    diff_snapshots,
    evaluate_check,
    render_curl,
)

ENV = load_environment("fintech")
PROBES = {c.id: c for c in ENV.baseline_candidates()}


def _verifier(*bugs: str, runs: int = 3) -> DeterministicVerifier:
    return DeterministicVerifier(ENV.graph, ENV.app_factory(bugs), ENV.snapshot, runs=runs)


def _ex(status: int) -> Exchange:
    return Exchange(step=0, edge_id="e", actor_id="a", method="GET", path="/", status_code=status)


# --- check evaluation --------------------------------------------------------

def test_status_code_check():
    check = StatusCodeCheck(step=0, allowed=frozenset({403, 404}))
    assert evaluate_check(check, [[_ex(404), _ex(403)]], DbSnapshot()).holds
    outcome = evaluate_check(check, [[_ex(404), _ex(200)]], DbSnapshot())
    assert not outcome.holds and "1/2" in outcome.detail


def test_ledger_conservation_check():
    wallets = {"w1": {"balance": 50}, "w2": {"balance": -10}}
    ledger = ({"wallet_id": "w1", "amount": 100}, {"wallet_id": "w1", "amount": -50},
              {"wallet_id": "w2", "amount": -10})
    snap = DbSnapshot(entities={"wallets": wallets}, ledger=ledger)
    assert evaluate_check(LedgerConservationCheck(account_ids=frozenset({"w1"})), [], snap).holds
    neg = evaluate_check(LedgerConservationCheck(), [], snap)
    assert not neg.holds and "negative balance" in neg.detail
    assert evaluate_check(LedgerConservationCheck(allow_negative=True), [], snap).holds
    drift = DbSnapshot(entities={"wallets": {"w1": {"balance": 60}}}, ledger=ledger[:2])
    assert "!= ledger sum 50" in evaluate_check(LedgerConservationCheck(), [], drift).detail
    missing = evaluate_check(LedgerConservationCheck(account_ids=frozenset({"ghost"})), [], snap)
    assert not missing.holds


def test_state_guard_check():
    check = StateGuardCheck(entity="loans", entity_id="l1", when_field="status", when_value="DISBURSED",
                            require_entity="users", require_id="u1", require_field="kyc",
                            require_value="APPROVED")

    def snap(status: str, kyc: str) -> DbSnapshot:
        return DbSnapshot(entities={"loans": {"l1": {"status": status}}, "users": {"u1": {"kyc": kyc}}})

    assert evaluate_check(check, [], snap("APPLIED", "PENDING")).holds  # not triggered
    assert evaluate_check(check, [], snap("DISBURSED", "APPROVED")).holds
    assert not evaluate_check(check, [], snap("DISBURSED", "PENDING")).holds
    assert evaluate_check(check, [], DbSnapshot()).holds


def test_diff_snapshots():
    before = DbSnapshot(entities={"w": {"a": {"b": 1, "x": 0}, "gone": {}}}, ledger=({"n": 1},))
    after = DbSnapshot(entities={"w": {"a": {"b": 2, "x": 0}, "new": {}}}, ledger=({"n": 1}, {"n": 2}))
    diff = diff_snapshots(before, after)
    assert [(c.row_id, c.field, c.before, c.after) for c in diff.changed] == [("a", "b", 1, 2)]
    assert diff.added_rows == {"w": ["new"]} and diff.removed_rows == {"w": ["gone"]}
    assert diff.ledger_appended == [{"n": 2}]
    assert diff_snapshots(before, before).is_empty


# --- end-to-end verification against the testbed -----------------------------

@pytest.mark.parametrize(
    ("probe_id", "bug"),
    [
        ("probe.cross_account_wallet_read", "idor_wallet_read"),
        ("probe.withdraw_concurrency_spike", "race_withdraw"),
        ("probe.loan_skip_approval", "loan_state_skip"),
        ("probe.negative_transfer", "negative_transfer"),
    ],
)
def test_planted_bug_confirmed_only_when_present(probe_id, bug):
    probe = PROBES[probe_id]
    confirmed = _verifier(bug).verify(probe)
    assert confirmed.verdict is Verdict.CONFIRMED
    assert [r.violated for r in confirmed.runs] == [True, True, True]
    assert confirmed.evidence_exchanges and confirmed.reproduction.startswith("#!/usr/bin/env bash")
    assert _verifier().verify(probe).verdict is Verdict.NOT_REPRODUCED


def test_race_evidence_shows_double_spend():
    result = _verifier("race_withdraw").verify(PROBES["probe.withdraw_concurrency_spike"])
    successes = [ex for ex in result.evidence_exchanges if ex.status_code == 200]
    assert len(successes) > 1  # more than one 6000 withdrawal from a 10000 balance
    withdrawals = [e for e in result.evidence_db_diff.ledger_appended if e["kind"] == "withdrawal"]
    assert len(withdrawals) == len(successes)
    assert "balance" in {c.field for c in result.evidence_db_diff.changed}

    patched = _verifier().verify(PROBES["probe.withdraw_concurrency_spike"])
    assert patched.verdict is Verdict.NOT_REPRODUCED


def test_patched_concurrency_spike_allows_exactly_one_withdrawal():
    import asyncio

    from qbq.verifier import replay_trace

    app = ENV.app_factory(())()
    exchanges = asyncio.run(replay_trace(app, ENV.graph, PROBES["probe.withdraw_concurrency_spike"].trace))
    assert sorted(ex.status_code for ex in exchanges[0]) == [200] + [400] * 9
    assert app.state.store.wallets["w_alice"]["balance"] == 4_000


def test_invalid_candidates_are_rejected_without_replay():
    bad_edge = CandidateFinding(
        id="bad.edge", invariant_id="inv.wallet_read_isolation", hypothesis="h",
        trace=ActionTrace(steps=(ActionStep(edge_id="edge.ghost", actor_id="user_a"),)),
    )
    bad_params = CandidateFinding(
        id="bad.params", invariant_id="inv.wallet_read_isolation", hypothesis="h",
        trace=ActionTrace(steps=(ActionStep(edge_id="edge.wallet_read", actor_id="user_a"),)),
    )
    bad_step = CandidateFinding(
        id="bad.step", invariant_id="inv.loan_happy_path", hypothesis="h",
        trace=ActionTrace(steps=(ActionStep(edge_id="edge.loan_apply", actor_id="user_a",
                                            body={"amount": 1}),)),
    )
    bad_inv = bad_params.model_copy(update={"invariant_id": "inv.ghost"})
    for candidate in (bad_edge, bad_params, bad_step, bad_inv):
        result = _verifier().verify(candidate)
        assert result.verdict is Verdict.INVALID and result.error and not result.runs


def test_runs_must_be_positive():
    with pytest.raises(ValueError):
        _verifier(runs=0)


def _flaky_world():
    graph = StateGraph(
        app_name="flaky",
        nodes=(StateNode(id="n", kind=NodeKind.API_ENDPOINT, label="n"),),
        edges=(TransitionEdge(id="edge.coin", source="n", target="n", method="GET",
                              path="/coin", auth_required=False),),
        actors=(ActorPersona(id="anon", display_name="anon", role="ANONYMOUS"),),
        invariants=(InvariantAssertion(id="inv.ok", category="functional", description="always 200",
                                       checks=(StatusCodeCheck(step=0, allowed=frozenset({200})),)),),
    )
    counter = itertools.count()

    def factory() -> FastAPI:
        broken = next(counter) % 2 == 0  # every other sandbox misbehaves
        app = FastAPI()

        @app.get("/coin")
        async def coin():
            from fastapi import HTTPException

            if broken:
                raise HTTPException(500)
            return {"ok": True}

        return app

    candidate = CandidateFinding(
        id="c", invariant_id="inv.ok", hypothesis="h",
        trace=ActionTrace(steps=(ActionStep(edge_id="edge.coin", actor_id="anon"),)),
    )
    return graph, factory, candidate


def test_unreproducible_findings_are_flaky_and_evidence_discarded():
    graph, factory, candidate = _flaky_world()
    result = DeterministicVerifier(graph, factory, lambda app: DbSnapshot(), runs=3).verify(candidate)
    assert [r.violated for r in result.runs] == [True, False, True]
    assert result.verdict is Verdict.FLAKY
    assert not result.confirmed and result.evidence_exchanges == [] and result.reproduction is None


def test_render_curl():
    script = render_curl(ENV.graph, PROBES["probe.withdraw_concurrency_spike"].trace, "http://localhost:8000")
    assert "Authorization: Bearer tok_alice" in script
    assert "http://localhost:8000/wallets/w_alice/withdraw" in script
    assert "$(seq 10)" in script and script.rstrip().endswith("wait")
    single = render_curl(ENV.graph, PROBES["probe.anonymous_wallet_read"].trace, "http://h/")
    assert "Authorization" not in single and "http://h/wallets/w_alice" in single
