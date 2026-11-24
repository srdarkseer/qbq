import asyncio

from qbq.compiler import compile_app
from qbq.compiler.grounding import aground_personas
from qbq.graph import ActorPersona, StateGraph
from qbq.gym.fintech import create_app
from qbq.gym.fintech.compiled import compiled_graph
from qbq.gym.fintech.graph import build_graph
from qbq.gym.saas.environment import compiled_graph as saas_graph
from qbq.search.readiness import (
    Readiness,
    ledger_readiness,
    observed_state_of,
    precondition_readiness,
)
from qbq.verifier import ActionStep

G = compiled_graph()
S = saas_graph()


def test_grounding_records_owned_resource_state():
    alice = G.actor("user_a")
    assert alice.observed("wallets", "w_alice")["balance"] == 10_000
    assert G.actor("user_c").observed("wallets", "w_carol")["balance"] == 0
    assert G.actor("anonymous").observed_state == {} and G.actor("merchant_admin").observed_state == {}


def test_grounding_only_reads_and_round_trips():
    graph = StateGraph.from_json(G.to_json())
    assert graph == G
    assert graph.actor("user_b").observed("wallets", "w_bob")["balance"] == 5_000


def test_grounding_drops_credential_like_fields():
    from fastapi import FastAPI, Header

    app = FastAPI()

    @app.get("/vaults/{vault_id}")
    async def read(vault_id: str, authorization: str = Header()) -> dict:
        return {"id": vault_id, "api_token": "s3cr3t", "password_hash": "x", "level": 3, "tags": ["a"]}

    persona = ActorPersona(id="u", display_name="U", role="USER", auth_token="t",
                           owned_resources={"vaults": frozenset({"v1"})})
    graph = compile_app(lambda: app, personas=[persona]).graph
    assert graph.actor("u").observed("vaults", "v1") == {"id": "v1", "level": 3}


def test_compile_without_grounding_leaves_personas_untouched():
    result = compile_app(lambda: create_app(), personas=build_graph().actors, ground=False)
    assert all(a.observed_state == {} for a in result.graph.actors)
    assert result.report.grounded_records == 0


def test_grounding_is_deterministic():
    again = asyncio.run(aground_personas(lambda: create_app(), G))
    assert again == G.actors


# --------------------------------------------------------------------------- readiness


def _step(edge, actor, **kw):
    return ActionStep(edge_id=edge, actor_id=actor, **kw)


def test_ledger_readiness_uses_observed_balances():
    edge = G.edge("edge.withdraw")
    ready = ledger_readiness(G, edge, _step("edge.withdraw", "user_a", path_params={"wallet_id": "w_alice"},
                                            body={"amount": 1}))
    assert ready.status is Readiness.READY
    broke = ledger_readiness(G, edge, _step("edge.withdraw", "user_c", path_params={"wallet_id": "w_carol"},
                                            body={"amount": 1}))
    assert broke.status is Readiness.NOT_READY and "w_carol" in broke.reason
    too_big = ledger_readiness(G, edge, _step("edge.withdraw", "user_b", path_params={"wallet_id": "w_bob"},
                                              body={"amount": 6_000}))
    assert too_big.status is Readiness.NOT_READY
    # a transfer only checks the account the caller owns (the source), not the destination
    transfer = _step("edge.transfer", "user_a",
                     body={"amount": 1, "from_wallet_id": "w_alice", "to_wallet_id": "w_carol"})
    assert ledger_readiness(G, G.edge("edge.transfer"), transfer).ready


def test_observed_lifecycle_state_and_preconditions():
    upgrade = S.edge("edge.upgrade_subscription")
    alpha = _step(upgrade.id, "alpha_owner", path_params={"subscription_id": "sub_alpha"})
    beta = _step(upgrade.id, "beta_owner", path_params={"subscription_id": "sub_beta"})
    assert observed_state_of(S, upgrade, alpha) == "ACTIVE"
    assert observed_state_of(S, upgrade, beta) == "TRIALING"
    activate = S.edge("edge.activate_subscription")
    assert precondition_readiness(
        S, activate, _step(activate.id, "beta_owner", path_params={"subscription_id": "sub_beta"})).ready
    pending = precondition_readiness(
        S, activate, _step(activate.id, "alpha_owner", path_params={"subscription_id": "sub_alpha"}))
    assert pending.status is Readiness.NOT_READY and "billing_status='PENDING'" in pending.reason
    # fintech exposes no KYC read endpoint: the approve precondition stays unknown
    approve = G.edge("edge.approve_loan")
    fact = precondition_readiness(G, approve, _step(approve.id, "merchant_admin",
                                                    path_params={"loan_id": "loan_carol"}))
    assert fact.status is Readiness.UNKNOWN
