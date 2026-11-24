import json

import pytest
from pydantic import ValidationError

from qbq.graph import (
    ActorPersona,
    HttpMethod,
    InvariantAssertion,
    InvariantCategory,
    LedgerConservationCheck,
    NodeKind,
    StateGraph,
    StateGuardCheck,
    StateNode,
    StatusCodeCheck,
    TransitionEdge,
)


def _graph(**overrides) -> StateGraph:
    fields = dict(
        app_name="mini",
        nodes=(
            StateNode(id="api.wallet", kind=NodeKind.API_ENDPOINT, label="Wallet API"),
            StateNode(
                id="wallet.active", kind=NodeKind.ENTITY_STATE, label="Wallet active",
                entity="wallet", state="ACTIVE",
            ),
        ),
        edges=(
            TransitionEdge(
                id="edge.withdraw", source="api.wallet", target="wallet.active",
                method=HttpMethod.POST, path="/wallets/{wallet_id}/withdraw",
                allowed_roles=frozenset({"USER"}),
                payload_schema={"type": "object", "properties": {"amount": {"type": "integer"}}},
            ),
        ),
        actors=(
            ActorPersona(id="anon", display_name="Unauthenticated", role="ANONYMOUS"),
            ActorPersona(
                id="user_a", display_name="User A", role="USER", auth_token="tok_a",
                owned_resources={"wallets": frozenset({"w_a"})},
            ),
        ),
        invariants=(
            InvariantAssertion(
                id="inv.cross_tenant", category=InvariantCategory.SECURITY,
                description="Other users cannot withdraw from A's wallet",
                edge_ids=("edge.withdraw",),
                checks=(StatusCodeCheck(step=0, allowed=frozenset({403, 404})),),
            ),
            InvariantAssertion(
                id="inv.ledger", category=InvariantCategory.FINANCIAL,
                description="Balances match the ledger",
                checks=(LedgerConservationCheck(),),
            ),
            InvariantAssertion(
                id="inv.guard", category=InvariantCategory.FUNCTIONAL,
                description="Loan disbursal requires KYC",
                checks=(
                    StateGuardCheck(
                        entity="loans", entity_id="l1", when_field="status", when_value="DISBURSED",
                        require_entity="users", require_id="u1", require_field="kyc",
                        require_value="APPROVED",
                    ),
                ),
            ),
        ),
    )
    fields.update(overrides)
    return StateGraph(**fields)


def test_json_round_trip_is_lossless():
    graph = _graph()
    restored = StateGraph.from_json(graph.to_json())
    assert restored == graph
    assert restored.actor("user_a").auth_token == "tok_a"
    assert isinstance(restored.invariant("inv.ledger").checks[0], LedgerConservationCheck)
    assert isinstance(restored.invariant("inv.guard").checks[0], StateGuardCheck)


def test_json_is_plain_and_versioned():
    payload = json.loads(_graph().to_json())
    assert payload["schema_version"] == "1"
    assert payload["invariants"][0]["checks"][0]["kind"] == "status_code"


def test_json_schema_export_covers_all_models():
    schema = StateGraph.json_schema()
    defs = set(schema["$defs"])
    assert {"StateNode", "TransitionEdge", "ActorPersona", "InvariantAssertion"} <= defs
    json.dumps(schema)  # must be serializable


def test_lookups_and_adjacency():
    graph = _graph()
    assert graph.edge("edge.withdraw").method is HttpMethod.POST
    assert [e.id for e in graph.outgoing("api.wallet")] == ["edge.withdraw"]
    assert [e.id for e in graph.incoming("wallet.active")] == ["edge.withdraw"]
    with pytest.raises(KeyError):
        graph.node("missing")


def test_models_are_immutable():
    graph = _graph()
    with pytest.raises(ValidationError):
        graph.app_name = "other"


def test_actor_auth_headers_and_token_not_in_repr():
    actor = _graph().actor("user_a")
    assert actor.auth_headers() == {"Authorization": "Bearer tok_a"}
    assert "tok_a" not in repr(actor)
    assert _graph().actor("anon").auth_headers() == {}


def test_render_path():
    edge = _graph().edge("edge.withdraw")
    assert edge.path_params == ("wallet_id",)
    assert edge.render_path({"wallet_id": "w_a"}) == "/wallets/w_a/withdraw"
    with pytest.raises(KeyError):
        edge.render_path({})
    with pytest.raises(KeyError):
        edge.render_path({"wallet_id": "w", "extra": "x"})


@pytest.mark.parametrize(
    "path", ["wallets", "/wallets/{1bad}", "/a/{x}/{x}", "/a/{x", "/a/x}"],
)
def test_invalid_path_templates_rejected(path):
    with pytest.raises(ValidationError):
        TransitionEdge(id="e", source="a", target="b", method="GET", path=path)


def test_unauthenticated_edge_cannot_restrict_roles():
    with pytest.raises(ValidationError, match="unauthenticated"):
        TransitionEdge(
            id="e", source="a", target="b", method="GET", path="/x",
            auth_required=False, allowed_roles=frozenset({"ADMIN"}),
        )


def test_entity_state_node_requires_entity_and_state():
    with pytest.raises(ValidationError):
        StateNode(id="n", kind=NodeKind.ENTITY_STATE, label="x", entity="order")
    with pytest.raises(ValidationError):
        StateNode(id="n", kind=NodeKind.UI_SCREEN, label="x", state="PAID")


def test_rejects_duplicate_ids():
    graph = _graph()
    with pytest.raises(ValidationError, match="duplicate node id"):
        _graph(nodes=graph.nodes + (graph.nodes[0],))


def test_rejects_dangling_edge_endpoint():
    bad = TransitionEdge(id="e2", source="api.wallet", target="ghost", method="GET", path="/x")
    with pytest.raises(ValidationError, match="unknown node 'ghost'"):
        _graph(edges=_graph().edges + (bad,))


def test_rejects_invariant_with_unknown_edge():
    inv = InvariantAssertion(
        id="inv.x", category="security", description="x", edge_ids=("edge.ghost",),
        checks=(StatusCodeCheck(step=0, allowed=frozenset({200})),),
    )
    with pytest.raises(ValidationError, match="unknown edges"):
        _graph(invariants=(inv,))


def test_invariant_validation():
    with pytest.raises(ValidationError):
        InvariantAssertion(id="i", category="security", description="x", checks=())
    with pytest.raises(ValidationError):
        StatusCodeCheck(step=0, allowed=frozenset({999}))
    with pytest.raises(ValidationError):
        StatusCodeCheck(step=-1, allowed=frozenset({200}))


def test_rejects_unknown_fields_and_bad_ids():
    with pytest.raises(ValidationError):
        StateNode(id="n", kind="ui_screen", label="x", surprise=True)
    with pytest.raises(ValidationError):
        StateNode(id="1node", kind="ui_screen", label="x")


def test_discriminated_check_parsing_from_raw_json():
    raw = {
        "id": "inv.raw", "category": "financial", "description": "d",
        "checks": [{"kind": "ledger_conservation", "account_ids": ["w1"]}],
    }
    inv = InvariantAssertion.model_validate(raw)
    assert inv.checks[0].account_ids == frozenset({"w1"})
    raw["checks"][0]["kind"] = "nope"
    with pytest.raises(ValidationError):
        InvariantAssertion.model_validate(raw)


def test_serialization_is_deterministic_for_sets():
    edge = TransitionEdge(
        id="e", source="a", target="b", method="GET", path="/x",
        allowed_roles=frozenset({"USER", "ADMIN", "MERCHANT"}),
    )
    assert json.loads(edge.model_dump_json())["allowed_roles"] == ["ADMIN", "MERCHANT", "USER"]
    check = StatusCodeCheck(step=0, allowed=frozenset({404, 403, 401}))
    assert json.loads(check.model_dump_json())["allowed"] == [401, 403, 404]
