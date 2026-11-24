import asyncio

import httpx
import pytest

from qbq.eval import autonomous_environment, run_eval
from qbq.graph import NodeKind
from qbq.gym import load_environment
from qbq.gym.saas import BUG_IDS, create_app, snapshot
from qbq.gym.saas.environment import BUGS, PERSONAS
from qbq.search import HeuristicGraphPolicy, search
from qbq.verifier import Verdict

ALPHA = {"Authorization": "Bearer tok_alpha"}
BETA = {"Authorization": "Bearer tok_beta"}


def _call(app, method, path, headers=None, json=None, n=1):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            return await asyncio.gather(*(c.request(method, path, headers=headers, json=json) for _ in range(n)))
    return asyncio.run(go())


def _ledger_ok(app):
    snap = snapshot(app)
    return all(
        w["seats_available"] == sum(e["amount"] for e in snap.ledger if e["workspace_id"] == wid)
        and w["seats_available"] >= 0
        for wid, w in snap.entities["workspaces"].items()
    )


def test_seed_reconciles_and_bug_ids_are_validated():
    assert _ledger_ok(create_app())
    assert {b.id for b in BUGS} == BUG_IDS
    with pytest.raises(ValueError):
        create_app({"nope"})


@pytest.mark.parametrize("bug", sorted(BUG_IDS))
def test_each_defect_is_off_when_patched_and_on_when_toggled(bug):
    def exploit(app):
        if bug == "cross_tenant_workspace_read":
            return _call(app, "GET", "/workspaces/ws_alpha", BETA)[0].status_code == 200
        if bug == "seat_limit_race":
            _call(app, "POST", "/workspaces/ws_beta/members", BETA, {"email": "x@y"}, n=10)
            return not _ledger_ok(app)
        if bug == "tier_state_skip":
            return _call(app, "POST", "/subscriptions/sub_beta/upgrade", BETA, {"tier": "PRO"})[0].status_code == 200
        return _call(app, "POST", "/workspaces/ws_beta/seats", BETA, {"quantity": -2})[0].status_code == 200

    assert not exploit(create_app())
    assert exploit(create_app({bug}))


def test_happy_paths_and_guards_on_patched_build():
    app = create_app()
    assert _call(app, "POST", "/workspaces/ws_beta/members", BETA, {"email": "a@b"})[0].status_code == 201
    assert _call(app, "POST", "/workspaces/ws_alpha/members", ALPHA, {"email": "a@b"})[0].status_code == 409
    assert _call(app, "POST", "/subscriptions/sub_beta/activate", BETA)[0].status_code == 200
    assert _call(app, "POST", "/subscriptions/sub_beta/upgrade", BETA, {"tier": "PRO"})[0].status_code == 200
    assert _call(app, "POST", "/subscriptions/sub_alpha/activate", ALPHA)[0].status_code == 409
    assert _call(app, "DELETE", "/api_keys/key_alpha", BETA)[0].status_code == 403
    assert _call(app, "GET", "/workspaces/ws_beta")[0].status_code == 401
    assert _ledger_ok(app)


# --------------------------------------------------------------------------- compile + search + verify


@pytest.fixture(scope="module")
def env():
    return load_environment("saas-autonomous")


def test_compiler_extracts_saas_structure_with_no_app_rules(env):
    g = env.graph
    assert len(g.edges) == 10 and {a.id for a in g.actors} == {p.id for p in PERSONAS}
    for bug in BUGS:
        g.invariant(bug.invariant_id)  # every ground-truth invariant was synthesized
    lifecycle = {n.id for n in g.nodes if n.kind is NodeKind.ENTITY_STATE}
    assert lifecycle == {"subscription.active", "subscription.trialing"}
    assert g.edge("edge.upgrade_subscription").metadata["guards"]["state_requires"] == "ACTIVE"
    assert g.edge("edge.activate_subscription").metadata["guards"]["preconditions"] == [
        {"field": "billing_status", "status": 409, "value": "VERIFIED"}]
    conservation = g.invariant("inv.ledger.conservation").checks[0]
    assert (conservation.entity, conservation.balance_field, conservation.account_field) == (
        "workspaces", "seats_available", "workspace_id")
    assert g.actor("beta_owner").observed("workspaces", "ws_beta")["seats_available"] == 5


def test_grounded_search_picks_state_ready_personas(env):
    first = {c.invariant_id: c.tests[0] for c in search(env.graph).coverage.values() if c.tests}
    assert first["inv.ledger.conservation"] == (
        "auto:parallel_contention:inv.ledger.conservation:add_member:beta_owner.x16")
    assert first["inv.lifecycle.upgrade_subscription"].endswith("beta_owner.sub_beta")
    assert first["inv.functional.activate_subscription"].endswith("beta_owner.nominal")
    assert first["inv.functional.upgrade_subscription"].endswith("alpha_owner.nominal")


def test_saas_autonomous_eval_full_recall_zero_fpr(env):
    report = run_eval(env, runs=3)
    m = report.metrics
    assert m.recall_at_1 == 1.0 and m.bugs_found_at_1 == m.bugs_total == 8
    assert m.false_positives == 0 and m.false_positive_rate == 0.0
    assert m.flaky == 0 and m.invalid == 0


def test_ungrounded_policy_regresses_on_saas():
    """Milestone 3 tie-breaking picks Alpha (no free seats, already ACTIVE): misses and FPs."""
    base = load_environment("saas-compiled")
    env = autonomous_environment(base, policy=HeuristicGraphPolicy(use_observed_state=False))
    report = run_eval(env, runs=1)
    assert report.metrics.recall_at_1 < 1.0
    assert report.metrics.false_positives > 0
    fp = {c.candidate_id for c in report.cases if c.verdict is Verdict.CONFIRMED and not c.expected_violation}
    assert fp == {"auto:lifecycle_order:inv.lifecycle.upgrade_subscription:upgrade_subscription:alpha_owner.sub_alpha"}
