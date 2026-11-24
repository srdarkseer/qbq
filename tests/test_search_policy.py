import json

import pytest

from qbq.gym.fintech.compiled import compiled_graph
from qbq.gym.fintech.graph import build_graph
from qbq.search import (
    CombinatorKind,
    CoverageStatus,
    GraphSearch,
    HeuristicGraphPolicy,
    LLMGuidedPolicy,
    RankingModel,
    RankingPrompt,
    SearchConfig,
    SearchFrame,
    SearchPolicy,
    build_frames,
    expand,
    generate_pool,
    parse_ranking,
    search,
    uniform_edge_sample,
)
from qbq.verifier import validate_candidate

G = compiled_graph()
H = HeuristicGraphPolicy()


def _frame(inv_id, edge_id, kind):
    return SearchFrame(node_id=G.edge(edge_id).source, invariant_id=inv_id, edge_id=edge_id, combinator=kind)


def _ranked(inv_id, edge_id, kind, policy=H):
    frame = _frame(inv_id, edge_id, kind)
    return policy.rank(G, frame, expand(G, kind, G.edge(edge_id), G.invariant(inv_id)))


@pytest.fixture(scope="module")
def plan():
    return search(G)


def test_policies_satisfy_protocol():
    assert isinstance(H, SearchPolicy)
    assert isinstance(LLMGuidedPolicy(_Echo()), SearchPolicy)


def test_frames_cover_every_invariant_edge_and_combinator():
    frames = build_frames(G)
    assert {f.invariant_id for f in frames} == {i.id for i in G.invariants}
    assert len({f.key for f in frames}) == len(frames)


def test_uncovered_invariants_rank_above_covered():
    frame = _frame("inv.tenant.withdraw", "edge.withdraw", CombinatorKind.CROSS_PERSONA)
    assert H.score_frame(G, frame, {}) > H.score_frame(G, frame, {"inv.tenant.withdraw": 1})


def test_static_guards_shape_frame_priority():
    # a schema field with no lower bound outranks one with a declared bound
    unbounded = _frame("inv.input.transfer.amount", "edge.transfer", CombinatorKind.BOUNDARY_PERTURB)
    bounded = _frame("inv.input.withdraw.amount", "edge.withdraw", CombinatorKind.BOUNDARY_PERTURB)
    assert H.score_frame(G, unbounded, {}) > H.score_frame(G, bounded, {})
    # contention on a path-bound ledger account outranks a body-addressed one
    withdraw = _frame("inv.ledger.conservation", "edge.withdraw", CombinatorKind.PARALLEL_CONTENTION)
    transfer = _frame("inv.ledger.conservation", "edge.transfer", CombinatorKind.PARALLEL_CONTENTION)
    assert H.score_frame(G, withdraw, {}) > H.score_frame(G, transfer, {})
    # an auth guard the compiler already saw hold is lower priority than an ownership gap
    auth = _frame("inv.auth.withdraw", "edge.withdraw", CombinatorKind.CROSS_PERSONA)
    tenant = _frame("inv.tenant.withdraw", "edge.withdraw", CombinatorKind.CROSS_PERSONA)
    assert H.score_frame(G, tenant, {}) > H.score_frame(G, auth, {})


def test_role_permitted_intruder_is_pruned():
    ranked = _ranked("inv.tenant.read_wallet", "edge.read_wallet", CombinatorKind.CROSS_PERSONA)
    pruned = {s.test.features["intruder"] for s in ranked if s.pruned}
    live = [s for s in ranked if not s.pruned]
    assert pruned == {"merchant_admin"}
    assert all(s.test.features["intruder_role"] == s.test.features["owner_role"] for s in live)
    # a lower-privilege intruder on a role-guarded edge is a valid vertical check
    approve = _ranked("inv.tenant.approve_loan", "edge.approve_loan", CombinatorKind.CROSS_PERSONA)
    assert approve and all(s.pruned is None for s in approve)


def test_success_probes_with_unmet_preconditions_are_pruned():
    cases = {
        ("inv.functional.disburse_loan", "edge.disburse_loan"): "lifecycle precondition",
        ("inv.functional.approve_loan", "edge.approve_loan"): "data preconditions",
    }
    for (inv, edge), reason in cases.items():
        ranked = _ranked(inv, edge, CombinatorKind.BOUNDARY_PERTURB)
        assert ranked and all(s.pruned and reason in s.pruned for s in ranked), inv
    apply = _ranked("inv.functional.apply_loan", "edge.apply_loan", CombinatorKind.BOUNDARY_PERTURB)
    assert {s.test.features["actor"] for s in apply if s.pruned} == {"merchant_admin"}


def test_ledger_success_probes_follow_observed_balances():
    grounded = _ranked("inv.functional.withdraw", "edge.withdraw", CombinatorKind.BOUNDARY_PERTURB)
    live = {s.test.features["actor"] for s in grounded if not s.pruned}
    assert live == {"user_a", "user_b"}  # funded wallets
    carol = [s for s in grounded if s.test.features["actor"] == "user_c"]
    assert carol and all("w_carol.balance=0 cannot cover" in s.pruned for s in carol)
    # without grounding, balances are unknown and every ledger success probe is pruned
    ungrounded = _ranked("inv.functional.withdraw", "edge.withdraw", CombinatorKind.BOUNDARY_PERTURB,
                         HeuristicGraphPolicy(use_observed_state=False))
    assert ungrounded and all("account balances" in s.pruned for s in ungrounded)


def test_contention_prunes_unfunded_accounts():
    ranked = _ranked("inv.ledger.conservation", "edge.withdraw", CombinatorKind.PARALLEL_CONTENTION)
    assert {s.test.features["actor"] for s in ranked if s.pruned} == {"user_c"}
    assert ranked[0].test.features["actor"] in {"user_a", "user_b"}
    assert "funded account" in ranked[0].rationale


def test_input_probes_only_perturb_referenced_fields():
    ranked = _ranked("inv.input.transfer.amount", "edge.transfer", CombinatorKind.BOUNDARY_PERTURB)
    assert {s.test.features["field"] for s in ranked if not s.pruned} == {"amount"}
    assert ranked[0].test.features["case"] == "negative"


def test_lifecycle_prefers_owner(plan):
    first = plan.coverage["inv.lifecycle.disburse_loan"].tests[0]
    assert first.endswith("user_c.loan_carol")


def test_search_plan_targets_planted_attack_shapes_first(plan):
    first = {cov.invariant_id: cov.tests[0] for cov in plan.coverage.values() if cov.tests}
    assert first["inv.tenant.read_wallet"].startswith("auto:cross_persona:inv.tenant.read_wallet:read_wallet:")
    assert first["inv.ledger.conservation"].startswith(
        "auto:parallel_contention:inv.ledger.conservation:withdraw:")
    assert first["inv.input.transfer.amount"].endswith("amount.negative")
    assert first["inv.lifecycle.disburse_loan"].startswith("auto:lifecycle_order:")


def test_search_is_deterministic_and_replayable(plan):
    again = search(G)
    assert [t.id for t in again.tests] == [t.id for t in plan.tests]
    assert [e.rank for e in plan.emitted] == list(range(len(plan.emitted)))
    for candidate in plan.candidates():
        validate_candidate(G, candidate)


def test_search_reports_coverage_with_reasons(plan):
    assert set(plan.coverage) == {i.id for i in G.invariants}
    covered = [c for c in plan.coverage.values() if c.status is CoverageStatus.COVERED]
    assert len(covered) == len(plan.tests)  # per_invariant=1
    uncovered = plan.uncovered()
    assert "inv.functional.disburse_loan" in uncovered
    assert uncovered["inv.lifecycle.approve_loan"] == "combinators produced no replayable trace"
    assert all(uncovered.values())
    assert plan.stats.tests_pruned > 0 and plan.stats.frames_expanded <= plan.stats.frames_total


def test_search_config_quota_and_budget():
    two = GraphSearch(G, config=SearchConfig(per_invariant=2)).run()
    per_inv = {}
    for t in two.tests:
        per_inv[t.invariant_id] = per_inv.get(t.invariant_id, 0) + 1
    assert max(per_inv.values()) == 2
    capped = search(G, config=SearchConfig(max_tests=5))
    assert len(capped.tests) == 5
    assert [t.id for t in capped.tests] == [t.id for t in search(G).tests][:5]
    assert any(c.reason == "search budget exhausted" for c in capped.coverage.values())
    with pytest.raises(ValueError):
        SearchConfig(beam_width=0)


def test_search_runs_on_reference_graph():
    ref = build_graph()
    result = search(ref)
    assert result.coverage["inv.loan_requires_kyc"].status is CoverageStatus.COVERED
    assert result.coverage["inv.wallet_read_isolation"].status is CoverageStatus.COVERED
    for candidate in result.candidates():
        validate_candidate(ref, candidate)


def test_uniform_edge_sample_is_a_seeded_permutation():
    pool = generate_pool(G)
    a, b = uniform_edge_sample(pool, 7), uniform_edge_sample(pool, 7)
    assert [t.id for t in a] == [t.id for t in b]
    assert sorted(t.id for t in a) == sorted(t.id for t in pool)
    assert [t.id for t in uniform_edge_sample(pool, 8)] != [t.id for t in a]


# --------------------------------------------------------------------------- LLM adapter


class _Echo:
    """Mock model: ranks candidates in reverse key order and records prompts."""

    def __init__(self, reply=None):
        self.reply = reply
        self.prompts: list[RankingPrompt] = []

    def complete(self, prompt: RankingPrompt) -> str:
        self.prompts.append(prompt)
        if self.reply is not None:
            return self.reply(prompt) if callable(self.reply) else self.reply
        keys = [c["key"] for c in prompt.candidates][::-1]
        ranking = [{"key": k, "score": 1 - i / len(keys)} for i, k in enumerate(keys)]
        return "```json\n" + json.dumps({"ranking": ranking}) + "\n```"


def test_mock_model_satisfies_ranking_protocol():
    assert isinstance(_Echo(), RankingModel)


def test_ranking_prompt_is_compact_structured_and_credential_free():
    model = _Echo()
    policy = LLMGuidedPolicy(model)
    _ranked("inv.tenant.read_wallet", "edge.read_wallet", CombinatorKind.CROSS_PERSONA, policy)
    (prompt,) = model.prompts
    text = prompt.render()
    assert "tok_" not in text and "auth_token" not in text
    assert prompt.invariant["id"] == "inv.tenant.read_wallet"
    assert prompt.invariant["oracle"] == "forbidden"
    assert prompt.graph_slice["edge"]["id"] == "edge.read_wallet"
    assert {a["id"] for a in prompt.graph_slice["actors"]} == {a.id for a in G.actors}
    assert all(c["key"].startswith("c") and c["steps"] for c in prompt.candidates)
    # guard-pruned tests are never shown to the model
    assert not any("merchant_admin" in json.dumps(c["steps"]) for c in prompt.candidates)
    assert prompt.messages()[0]["role"] == "system" and json.loads(prompt.messages()[1]["content"])


def test_llm_policy_applies_model_ranking_but_keeps_pruning():
    heuristic = _ranked("inv.tenant.read_wallet", "edge.read_wallet", CombinatorKind.CROSS_PERSONA)
    llm = _ranked("inv.tenant.read_wallet", "edge.read_wallet", CombinatorKind.CROSS_PERSONA,
                  LLMGuidedPolicy(_Echo()))
    live_h = [s.test.id for s in heuristic if not s.pruned]
    live_l = [s.test.id for s in llm if not s.pruned]
    assert live_l == live_h[::-1]  # the mock reversed the order it was given
    assert [s.test.id for s in llm if s.pruned] == [s.test.id for s in heuristic if s.pruned]


def test_llm_policy_partial_ranking_appends_heuristic_rest():
    def top_only(prompt):
        return json.dumps({"ranking": [{"key": prompt.candidates[-1]["key"]}, {"key": "bogus"}]})

    heuristic = [s for s in _ranked("inv.tenant.read_wallet", "edge.read_wallet",
                                    CombinatorKind.CROSS_PERSONA) if not s.pruned]
    llm = [s for s in _ranked("inv.tenant.read_wallet", "edge.read_wallet",
                              CombinatorKind.CROSS_PERSONA, LLMGuidedPolicy(_Echo(top_only))) if not s.pruned]
    assert llm[0].test.id == heuristic[-1].test.id
    assert [s.test.id for s in llm[1:]] == [s.test.id for s in heuristic[:-1]]


def test_llm_policy_falls_back_on_bad_model_output():
    policy = LLMGuidedPolicy(_Echo("I think c0 looks best."))
    heuristic = _ranked("inv.tenant.read_wallet", "edge.read_wallet", CombinatorKind.CROSS_PERSONA)
    llm = _ranked("inv.tenant.read_wallet", "edge.read_wallet", CombinatorKind.CROSS_PERSONA, policy)
    assert [s.test.id for s in llm] == [s.test.id for s in heuristic]
    assert policy.calls == 1 and len(policy.errors) == 1


def test_llm_guided_search_end_to_end():
    model = _Echo()
    policy = LLMGuidedPolicy(model)
    result = search(G, policy)
    heuristic = search(G)
    assert result.policy == "llm" and policy.calls == len(model.prompts) > 0
    assert set(result.coverage) == set(heuristic.coverage)
    assert {k for k, c in result.coverage.items() if c.tests} == {
        k for k, c in heuristic.coverage.items() if c.tests
    }
    assert [t.id for t in result.tests] != [t.id for t in heuristic.tests]


def test_parse_ranking_is_tolerant():
    keys = ["c0", "c1", "c2"]
    assert parse_ranking('{"ranking": ["c2", "c0"]}', keys) == [("c2", 1.0), ("c0", 0.5)]
    assert parse_ranking('noise {"ranking": [{"key": "c1", "score": 7}]} trailing', keys) == [("c1", 1.0)]
    assert parse_ranking('{"ranking": [{"key": "c1"}, {"key": "c1"}, {"key": "zz"}]}', keys) == [
        ("c1", 1.0)
    ]
    with pytest.raises(ValueError):
        parse_ranking("no json here", keys)
    with pytest.raises(ValueError):
        parse_ranking('{"order": []}', keys)
