import textwrap

from cubic.compiler import analyze_files, analyze_sources, extract_candidates
from cubic.graph import HttpMethod, StateGraph

ROUTER_SRC = textwrap.dedent('''
    from typing import Annotated
    from fastapi import APIRouter, Depends, Header, HTTPException
    from pydantic import BaseModel, Field

    router = APIRouter()

    def get_user(authorization: str = Header(None)):
        if authorization is None:
            raise HTTPException(status_code=401)
        return {"id": "u1", "role": "MERCHANT"}

    def require_admin(user=Depends(get_user)):
        return user

    Viewer = Annotated[dict, Depends(get_user)]

    class OrderIn(BaseModel):
        sku: str
        quantity: int = Field(ge=1, le=10)
        note: str | None = None

    def load_order(order_id, user):
        order = DB[order_id]
        if order["owner_id"] != user["id"]:
            raise HTTPException(403)
        return order

    @router.get("/orders/{order_id}")
    def read_order(order_id: str, user: Viewer):
        return load_order(order_id, user)

    @router.post("/orders", status_code=201)
    def create_order(body: OrderIn, user: Annotated[dict, Depends(get_user)]):
        return {"id": "o1", "status": "PENDING"}

    @router.post("/orders/{order_id}/ship")
    async def ship_order(order_id: str, user=Depends(get_user)):
        if user["role"] != "MERCHANT":
            raise HTTPException(403, "merchants only")
        order = load_order(order_id, user)
        if "skip_check" not in FLAGS and order["status"] != "PAID":
            raise HTTPException(409)
        order["status"] = "SHIPPED"
        order["shipped_at"] = 1
        return order

    @router.get("/public/ping")
    def ping():
        return {"status": "ok"}

    @router.delete("/orders/{order_id}")
    def ship_order(order_id: str, user: Viewer):  # duplicate function name
        raise HTTPException(status_code=405)
''')


def analyze():
    return analyze_sources([("svc/orders.py", ROUTER_SRC)])


def route(analysis, method, path):
    return next(r for r in analysis.routes if r.method is method and r.path == path)


def test_extracts_all_decorated_routes_with_locations():
    a = analyze()
    keys = [(r.method.value, r.path) for r in a.routes]
    assert sorted(keys) == sorted([
        ("GET", "/orders/{order_id}"), ("POST", "/orders"), ("POST", "/orders/{order_id}/ship"),
        ("GET", "/public/ping"), ("DELETE", "/orders/{order_id}"),
    ])
    r = route(a, HttpMethod.POST, "/orders")
    assert r.location.file == "svc/orders.py" and r.location.function == "create_order"
    assert r.decorator_line == r.location.line - 1 and r.status_code == 201
    assert a.source_bytes == len(ROUTER_SRC.encode())


def test_auth_dependency_forms():
    a = analyze()
    assert route(a, HttpMethod.GET, "/orders/{order_id}").auth_dependency == "get_user"   # alias
    assert route(a, HttpMethod.POST, "/orders").auth_dependency == "get_user"             # inline Annotated
    assert route(a, HttpMethod.POST, "/orders/{order_id}/ship").auth_dependency == "get_user"  # default
    ping = route(a, HttpMethod.GET, "/public/ping")
    assert ping.auth_dependency is None and 401 not in ping.raised_statuses
    assert 401 in route(a, HttpMethod.GET, "/orders/{order_id}").raised_statuses


def test_guards_follow_helpers_and_ignore_feature_flags():
    a = analyze()
    read = route(a, HttpMethod.GET, "/orders/{order_id}")
    assert read.ownership_guard and 403 in read.raised_statuses  # found via load_order()
    ship = route(a, HttpMethod.POST, "/orders/{order_id}/ship")
    assert ship.role_guards == {"MERCHANT"}
    assert ship.required_state == "PAID" and ship.state_requirements[0].status == 409
    assert ship.produced_state == "SHIPPED"
    assert ship.feature_flags == {"skip_check"}
    assert {"status", "shipped_at"} <= ship.mutated_fields


def test_get_routes_never_produce_state():
    a = analyze()
    assert route(a, HttpMethod.GET, "/public/ping").produced_state is None
    assert route(a, HttpMethod.POST, "/orders").produced_state == "PENDING"


def test_request_models_and_roles():
    a = analyze()
    schema = a.model("OrderIn").json_schema()
    assert schema["required"] == ["sku", "quantity"]
    assert schema["properties"]["quantity"] == {"type": "integer", "minimum": 1, "maximum": 10}
    assert a.roles == {"MERCHANT"}


def test_candidates_form_a_valid_graph_with_unique_ids():
    nodes, edges = extract_candidates(analyze())
    ids = [e.id for e in edges]
    assert len(ids) == len(set(ids)) and "edge.ship_order" in ids
    assert any(i.startswith("edge.ship_order_") for i in ids)
    by_id = {e.id: e for e in edges}
    ship = next(e for e in edges if e.path == "/orders/{order_id}/ship")
    assert (ship.source, ship.target) == ("order.paid", "order.shipped")
    assert ship.allowed_roles == {"MERCHANT"} and ship.required_headers == {"Authorization"}
    assert ship.metadata["source"] == {"file": "svc/orders.py", "function": "ship_order", "line": ship.metadata["source"]["line"]}
    create = by_id["edge.create_order"]
    assert (create.source, create.target) == ("api.orders", "order.pending")
    assert create.payload_schema["properties"]["sku"] == {"type": "string"}
    assert 201 in create.response_statuses
    graph = StateGraph(app_name="orders", nodes=nodes, edges=edges)
    assert StateGraph.from_json(graph.to_json()) == graph


def test_fintech_testbed_static_facts():
    a = analyze_files(["src/cubic/gym/fintech/app.py"], root="src")
    assert a.files == ("cubic/gym/fintech/app.py",)
    assert len(a.routes) == 7 and a.roles == {"USER", "ADMIN"}
    assert a.entity_collections == {"users", "wallets", "loans"}
    assert a.ledger is not None
    assert (a.ledger.entity, a.ledger.balance_field, a.ledger.account_field) == ("wallets", "balance", "wallet_id")

    approve = route(a, HttpMethod.POST, "/loans/{loan_id}/approve")
    assert approve.role_guards == {"ADMIN"} and approve.required_state == "APPLIED"
    assert [(p.field, p.value) for p in approve.preconditions] == [("kyc_status", "APPROVED")]
    disburse = route(a, HttpMethod.POST, "/loans/{loan_id}/disburse")
    assert disburse.required_state == "APPROVED" and disburse.produced_state == "DISBURSED"
    assert disburse.writes_ledger and disburse.feature_flags == {"loan_state_skip"}
    withdraw = route(a, HttpMethod.POST, "/wallets/{wallet_id}/withdraw")
    assert withdraw.ownership_guard and withdraw.writes_ledger and withdraw.uses_lock
    read = route(a, HttpMethod.GET, "/wallets/{wallet_id}")
    assert read.ownership_guard and not read.role_guards  # admin bypass is not a role restriction
    assert not read.writes_ledger and read.produced_state is None


def test_static_analysis_is_deterministic():
    one = extract_candidates(analyze_files(["src/cubic/gym/fintech/app.py"], root="src"))
    two = extract_candidates(analyze_files(["src/cubic/gym/fintech/app.py"], root="src"))
    assert one == two
