import asyncio

import httpx
import pytest
from fastapi.routing import APIRoute

from cubic.graph import StateGraph
from cubic.gym import load_environment
from cubic.gym.fintech import BUG_IDS, BUGS, build_graph, create_app

ALICE = {"Authorization": "Bearer tok_alice"}
BOB = {"Authorization": "Bearer tok_bob"}
CAROL = {"Authorization": "Bearer tok_carol"}
ADMIN = {"Authorization": "Bearer tok_admin"}


class SyncASGIClient:
    """Minimal synchronous client over httpx's ASGI transport (state persists per app)."""

    def __init__(self, app):
        self.app = app

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def go() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testbed") as c:
                return await c.request(method, url, **kwargs)

        return asyncio.run(go())

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)


def client(*bugs: str) -> SyncASGIClient:
    return SyncASGIClient(create_app(bugs, io_delay_s=0))


def test_health_is_public():
    assert client().get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "tok_alice"}])
def test_authentication_required(headers):
    assert client().get("/wallets/w_alice", headers=headers).status_code == 401


def test_owner_and_admin_can_read_wallet():
    c = client()
    assert c.get("/wallets/w_alice", headers=ALICE).json()["balance"] == 10_000
    assert c.get("/wallets/w_alice", headers=ADMIN).status_code == 200


def test_idor_toggle():
    assert client().get("/wallets/w_alice", headers=BOB).status_code == 404
    assert client("idor_wallet_read").get("/wallets/w_alice", headers=BOB).status_code == 200


def test_withdraw_rules():
    c = client()
    assert c.post("/wallets/w_alice/withdraw", json={"amount": 1_000}, headers=BOB).status_code == 403
    assert c.post("/wallets/w_alice/withdraw", json={"amount": 0}, headers=ALICE).status_code == 422
    assert c.post("/wallets/w_alice/withdraw", json={"amount": 99_999}, headers=ALICE).status_code == 400
    r = c.post("/wallets/w_alice/withdraw", json={"amount": 1_000}, headers=ALICE)
    assert r.status_code == 200 and r.json()["balance"] == 9_000


def test_transfer_and_negative_amount_regression():
    c = client()
    body = {"from_wallet_id": "w_alice", "to_wallet_id": "w_bob", "amount": 2_500}
    assert c.post("/transfers", json=body, headers=ALICE).status_code == 200
    assert c.get("/wallets/w_bob", headers=BOB).json()["balance"] == 7_500
    assert c.post("/transfers", json=body, headers=BOB).status_code == 403
    assert c.post("/transfers", json={**body, "to_wallet_id": "w_alice"}, headers=ALICE).status_code == 400

    negative = {**body, "amount": -5_000}
    assert client().post("/transfers", json=negative, headers=ALICE).status_code == 422
    vuln = client("negative_transfer")
    assert vuln.post("/transfers", json=negative, headers=ALICE).status_code == 200
    assert vuln.get("/wallets/w_bob", headers=BOB).json()["balance"] == 0


def test_loan_lifecycle_and_state_skip():
    c = client()
    loan = c.post("/loans", json={"amount": 50_000}, headers=ALICE)
    assert loan.status_code == 201
    loan_id = loan.json()["id"]
    assert c.post(f"/loans/{loan_id}/disburse", headers=ALICE).status_code == 409
    assert c.post(f"/loans/{loan_id}/approve", headers=ALICE).status_code == 403
    assert c.post(f"/loans/{loan_id}/approve", headers=ADMIN).status_code == 200
    assert c.post(f"/loans/{loan_id}/disburse", headers=ALICE).json()["status"] == "DISBURSED"
    assert c.post(f"/loans/{loan_id}/disburse", headers=ALICE).status_code == 409

    # Carol's KYC is pending: approval is refused, and disbursal must be too.
    assert c.post("/loans/loan_carol/approve", headers=ADMIN).status_code == 409
    assert c.post("/loans/loan_carol/disburse", headers=CAROL).status_code == 409
    assert c.post("/loans/loan_carol/disburse", headers=BOB).status_code == 403
    assert client("loan_state_skip").post("/loans/loan_carol/disburse", headers=CAROL).status_code == 200


def test_unknown_bug_toggle_rejected():
    with pytest.raises(ValueError, match="unknown bug"):
        create_app({"sql_injection"})


def test_each_app_instance_is_isolated():
    a, b = create_app(), create_app()
    a.state.store.wallets["w_alice"]["balance"] = 0
    assert b.state.store.wallets["w_alice"]["balance"] == 10_000


def test_seed_state_reconciles_with_ledger():
    snap = create_app().state.store.snapshot()
    for wid, wallet in snap.entities["wallets"].items():
        assert wallet["balance"] == sum(e["amount"] for e in snap.ledger if e["wallet_id"] == wid)


def test_graph_matches_application_routes():
    graph = build_graph()
    routes = {
        (method, route.path)
        for route in create_app().routes if isinstance(route, APIRoute)
        for method in route.methods
    }
    for edge in graph.edges:
        assert (edge.method.value, edge.path) in routes, edge.id


def test_graph_actors_match_seeded_tokens():
    from cubic.gym.fintech.app import SEED_TOKENS

    tokens = {a.auth_token for a in build_graph().actors if a.auth_token}
    assert tokens == set(SEED_TOKENS)


def test_graph_round_trips_and_manifest_is_consistent():
    graph = build_graph()
    assert StateGraph.from_json(graph.to_json()) == graph
    assert {b.id for b in BUGS} == BUG_IDS
    env = load_environment("fintech")
    names = [v.name for v in env.variants()]
    assert names[0] == "patched" and names[-1] == "vulnerable" and len(names) == len(BUGS) + 2
