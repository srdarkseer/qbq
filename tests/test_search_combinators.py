import pytest

from cubic.graph import (
    InvariantAssertion,
    InvariantCategory,
    LedgerConservationCheck,
    StateGuardCheck,
    StatusCodeCheck,
)
from cubic.gym.fintech.compiled import compiled_graph
from cubic.gym.fintech.graph import build_graph
from cubic.search import (
    BoundaryCase,
    CombinatorKind,
    Oracle,
    applicable_combinators,
    boundary_perturb,
    classify_invariant,
    cross_persona_check,
    expand,
    generate_pool,
    lifecycle_order_check,
    nominal_step,
    parallel_contention,
)
from cubic.search.combinators import boundary_values, initial_state_node, prerequisite_path
from cubic.verifier import validate_candidate

G = compiled_graph()


def _inv(checks, id="inv.x"):
    return InvariantAssertion(id=id, category=InvariantCategory.SECURITY, description="d", checks=checks)


@pytest.mark.parametrize(
    ("checks", "oracle"),
    [
        ((StatusCodeCheck(step=0, allowed=frozenset({401})),), Oracle.UNAUTHENTICATED),
        ((StatusCodeCheck(step=0, allowed=frozenset({403, 404})),), Oracle.FORBIDDEN),
        ((StatusCodeCheck(step=0, allowed=frozenset({400, 422})),), Oracle.REJECTED_INPUT),
        ((StatusCodeCheck(step=0, allowed=frozenset({400, 403, 409})),), Oracle.REJECTED_TRANSITION),
        ((StatusCodeCheck(step=0, allowed=frozenset({200, 201})),), Oracle.SUCCESS),
        ((LedgerConservationCheck(),), Oracle.CONSERVATION),
        ((StateGuardCheck(entity="e", entity_id="1", when_field="f", when_value=1, require_entity="r",
                          require_id="2", require_field="g", require_value=2),), Oracle.STATE_GUARD),
        ((StatusCodeCheck(step=0, allowed=frozenset({200, 500})),), Oracle.UNSUPPORTED),
        ((StatusCodeCheck(step=0, allowed=frozenset({201})),
          StatusCodeCheck(step=1, allowed=frozenset({200}))), Oracle.UNSUPPORTED),
    ],
)
def test_classify_invariant_uses_check_structure_only(checks, oracle):
    assert classify_invariant(_inv(checks)) is oracle


def test_every_compiled_invariant_has_a_combinator():
    for inv in G.invariants:
        assert applicable_combinators(inv), inv.id


def test_nominal_step_binds_owned_resources_and_valid_bodies():
    step = nominal_step(G, G.edge("edge.withdraw"), G.actor("user_a"))
    assert step.path_params == {"wallet_id": "w_alice"} and step.body == {"amount": 1}
    transfer = nominal_step(G, G.edge("edge.transfer"), G.actor("user_b"))
    assert transfer.body == {"amount": 1, "from_wallet_id": "w_bob", "to_wallet_id": "w_alice"}
    # role-guarded admin edge binds a resource the caller does not own
    approve = nominal_step(G, G.edge("edge.approve_loan"), G.actor("merchant_admin"))
    assert approve.path_params == {"loan_id": "loan_carol"}
    # an actor without a matching resource, an anonymous caller, or a wrong role cannot record
    assert nominal_step(G, G.edge("edge.withdraw"), G.actor("merchant_admin")) is None
    assert nominal_step(G, G.edge("edge.withdraw"), G.actor("anonymous")) is None
    assert nominal_step(G, G.edge("edge.approve_loan"), G.actor("user_c")) is None


def test_cross_persona_keeps_owner_resource_ids():
    inv = G.invariant("inv.tenant.withdraw")
    test = cross_persona_check(G, G.edge("edge.withdraw"), inv, G.actor("user_a"), G.actor("user_b"))
    (step,) = test.trace.steps
    assert step.actor_id == "user_b" and step.path_params == {"wallet_id": "w_alice"}
    assert test.combinator is CombinatorKind.CROSS_PERSONA and test.oracle is Oracle.FORBIDDEN
    assert test.features["owner"] == "user_a" and test.features["intruder_role"] == "USER"


def test_cross_persona_read_adds_owner_control_step():
    inv = G.invariant("inv.tenant.read_wallet")
    test = cross_persona_check(G, G.edge("edge.read_wallet"), inv, G.actor("user_a"), G.actor("user_c"))
    assert [(s.actor_id, s.path_params["wallet_id"]) for s in test.trace.steps] == [
        ("user_c", "w_alice"), ("user_a", "w_alice"),
    ]


def test_cross_persona_oracle_matching():
    edge, a, b, anon = G.edge("edge.withdraw"), G.actor("user_a"), G.actor("user_b"), G.actor("anonymous")
    auth, tenant = G.invariant("inv.auth.withdraw"), G.invariant("inv.tenant.withdraw")
    assert cross_persona_check(G, edge, auth, a, anon).trace.steps[0].actor_id == "anonymous"
    assert cross_persona_check(G, edge, auth, a, b) is None       # authenticated: not an auth test
    assert cross_persona_check(G, edge, tenant, a, anon) is None  # anonymous: not a tenant test
    assert cross_persona_check(G, edge, tenant, a, a) is None
    # an intruder that owns the bound resource is not an isolation test
    approve = G.edge("edge.approve_loan")
    assert cross_persona_check(G, approve, G.invariant("inv.tenant.approve_loan"),
                               G.actor("merchant_admin"), G.actor("user_c")) is None


def test_parallel_contention_is_barrier_synchronized():
    inv = G.invariant("inv.ledger.conservation")
    test = parallel_contention(G, G.edge("edge.withdraw"), inv, G.actor("user_a"), 16)
    first, observe = test.trace.steps
    assert first.concurrency == 16 and first.body == {"amount": 1}
    assert observe.edge_id == "edge.read_wallet" and observe.concurrency == 1
    assert test.request_count == 17
    assert parallel_contention(G, G.edge("edge.withdraw"), inv, G.actor("user_a"), 1) is None
    assert parallel_contention(G, G.edge("edge.read_wallet"), inv, G.actor("user_a"), 8) is None


def test_lifecycle_order_skips_prerequisite_edges():
    assert initial_state_node(G, "loan") == "loan.applied"
    assert prerequisite_path(G, "loan.applied", "loan.approved") == ("edge.approve_loan",)
    inv = G.invariant("inv.lifecycle.disburse_loan")
    test = lifecycle_order_check(G, G.edge("edge.disburse_loan"), inv, G.actor("user_c"))
    assert test.features["skipped_edges"] == ["edge.approve_loan"]
    assert test.features["actor_owns_resource"] is True
    assert test.trace.steps[0].path_params == {"loan_id": "loan_carol"}
    # approve_loan already starts at the seeded state: nothing to skip
    approve_inv = G.invariant("inv.lifecycle.approve_loan")
    assert lifecycle_order_check(G, G.edge("edge.approve_loan"), approve_inv, G.actor("merchant_admin")) is None


def test_boundary_values_respect_schema_bounds():
    bounded = boundary_values({"type": "integer", "exclusiveMinimum": 0, "maximum": 100}, True)
    assert bounded[BoundaryCase.NEGATIVE] == -1 and bounded[BoundaryCase.ZERO] == 0
    assert bounded[BoundaryCase.MIN_VALID] == 1 and bounded[BoundaryCase.MAX_VALID] == 100
    assert bounded[BoundaryCase.ABOVE_MAX] == 101 and BoundaryCase.MISSING in bounded
    shifted = boundary_values({"type": "integer", "minimum": 10}, False)
    assert shifted[BoundaryCase.BELOW_MIN] == 9 and BoundaryCase.MISSING not in shifted
    signed = boundary_values({"type": "integer", "minimum": -50}, False)
    assert BoundaryCase.NEGATIVE not in signed and BoundaryCase.ZERO not in signed
    unbounded = boundary_values({"type": "integer"}, True)
    assert unbounded[BoundaryCase.NEGATIVE] == -1 and unbounded[BoundaryCase.MIN_VALID] == 1


def test_boundary_perturb_targets_matching_oracles():
    edge, actor = G.edge("edge.transfer"), G.actor("user_a")
    rejected, success = G.invariant("inv.input.transfer.amount"), G.invariant("inv.functional.transfer")
    neg = boundary_perturb(G, edge, rejected, actor, "amount", BoundaryCase.NEGATIVE)
    assert neg.trace.steps[0].body["amount"] == -1
    missing = boundary_perturb(G, edge, rejected, actor, "amount", BoundaryCase.MISSING)
    assert "amount" not in missing.trace.steps[0].body
    assert boundary_perturb(G, edge, rejected, actor, "amount", BoundaryCase.MIN_VALID) is None
    assert boundary_perturb(G, edge, success, actor, "amount", BoundaryCase.NEGATIVE) is None
    ok = boundary_perturb(G, edge, success, actor, "amount", BoundaryCase.MIN_VALID)
    assert ok.trace.steps[0].body["amount"] == 1
    nominal = boundary_perturb(G, G.edge("edge.health"), G.invariant("inv.functional.health"),
                               G.actor("anonymous"), None, BoundaryCase.NOMINAL)
    assert nominal.trace.steps[0].body is None


def test_pool_is_deterministic_unique_and_replayable():
    pool = generate_pool(G)
    assert pool == generate_pool(G)
    assert len({t.id for t in pool}) == len(pool)
    assert len({(t.invariant_id, t.trace.model_dump_json()) for t in pool}) == len(pool)
    # fintech has no tool-reference edges, so the confused-deputy combinator never applies there
    assert {t.combinator for t in pool} == set(CombinatorKind) - {CombinatorKind.CROSS_TENANT_REFERENCE}
    for test in pool:
        validate_candidate(G, test.to_candidate())
        assert test.edge_id in G.invariant(test.invariant_id).edge_ids
        assert test.node_id == G.edge(test.edge_id).source


def test_pool_contains_the_four_attack_shapes():
    ids = {t.id for t in generate_pool(G)}
    assert "auto:cross_persona:inv.tenant.read_wallet:read_wallet:user_a.user_b" in ids
    assert "auto:parallel_contention:inv.ledger.conservation:withdraw:user_a.x16" in ids
    assert "auto:lifecycle_order:inv.lifecycle.disburse_loan:disburse_loan:user_c.loan_carol" in ids
    assert "auto:boundary_perturb:inv.input.transfer.amount:transfer:user_a.amount.negative" in ids


def test_expand_collapses_identical_traces():
    edge, inv = G.edge("edge.apply_loan"), G.invariant("inv.auth.apply_loan")
    tests = expand(G, CombinatorKind.CROSS_PERSONA, edge, inv)
    assert len(tests) == 1  # every owner's anonymous replay of {"amount": 1} is the same trace


def test_combinators_work_on_hand_written_reference_graph():
    ref = build_graph()
    pool = generate_pool(ref)
    kyc = [t for t in pool if t.invariant_id == "inv.loan_requires_kyc"]
    assert kyc and all(t.combinator is CombinatorKind.LIFECYCLE_ORDER for t in kyc)
    assert {t.oracle for t in kyc} == {Oracle.STATE_GUARD}
    for test in pool:
        validate_candidate(ref, test.to_candidate())
