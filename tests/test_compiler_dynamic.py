import contextlib
import json
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI, Header, HTTPException

from cubic.compiler import TargetError, compile_app, introspect, load_target
from cubic.compiler.dynamic import canonicalize, resolve_refs
from cubic.gym.fintech import create_app
from cubic.gym.fintech.graph import build_graph


# -- a small app exercising lifespan, dynamic-only routes and auth conflicts --------

def _maybe_user(authorization: str | None = Header(None)):
    if authorization == "Bearer revoked":
        raise HTTPException(401)
    return {"id": "guest"}


MaybeUser = Annotated[dict, Depends(_maybe_user)]


def make_lifespan_app() -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app):
        app.state.ready = True
        yield
        app.state.ready = False

    app = FastAPI(title="Lifespan Demo", lifespan=lifespan)
    app.state.ready = False

    @app.get("/ready")
    async def ready():
        if not app.state.ready:
            raise HTTPException(503)
        return {"ok": True}

    @app.get("/profile/{profile_id}")
    async def profile(profile_id: str, user: MaybeUser):
        return {"id": profile_id}

    async def legacy_delete(item_id: int):
        return None

    app.add_api_route("/legacy/{item_id}", legacy_delete, methods=["DELETE"], status_code=204)
    return app


def test_lifespan_startup_runs_before_probing():
    obs = introspect(make_lifespan_app)
    ready = next(r for r in obs.routes if r.path == "/ready")
    assert ready.observed_unauthenticated_status == 200  # would be 503 without lifespan


def test_lifespan_tolerates_apps_without_lifespan_support():
    async def bare_asgi(scope, receive, send):
        if scope["type"] == "lifespan":
            raise RuntimeError("lifespan unsupported")
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    obs = introspect(lambda: bare_asgi, probe=False)
    assert obs.routes == () and obs.openapi_bytes == 0


def test_dynamic_only_routes_and_auth_conflicts_are_reported():
    result = compile_app(make_lifespan_app)
    report = result.report
    assert report.route_coverage == 1.0 and report.routes_total == 3
    assert report.dynamic_only_routes == ["DELETE /legacy/{item_id}"]
    legacy = next(e for e in result.graph.edges if e.path == "/legacy/{item_id}")
    assert legacy.metadata["evidence"] == ["openapi", "runtime"]
    assert 204 in legacy.response_statuses
    assert legacy.payload_schema is None

    profile = next(e for e in result.graph.edges if e.path == "/profile/{profile_id}")
    assert profile.metadata["auth_conflict"] is True  # source can 401, but anonymous got 200
    assert profile.metadata["observed_unauthenticated_status"] == 200


def test_fintech_routes_and_bindings():
    obs = introspect(lambda: create_app())
    assert obs.openapi_version.startswith("3.")
    assert obs.source_files[0][0] == "cubic/gym/fintech/app.py"
    assert len(obs.routes) == 7
    for route in obs.routes:
        expected = 200 if route.path == "/health" else 401
        assert route.observed_unauthenticated_status == expected, route.path
        assert route.first_line and route.operation_id
    transfer = next(r for r in obs.routes if r.path == "/transfers")
    assert "$ref" not in json.dumps(transfer.body_schema) and "title" not in transfer.body_schema
    assert transfer.body_schema["properties"]["amount"] == {"type": "integer"}
    assert transfer.success_statuses == {200}


def test_resolve_refs_and_canonicalize():
    components = {"A": {"title": "A", "properties": {"b": {"$ref": "#/components/schemas/B"}}},
                  "B": {"type": "integer", "title": "B"},
                  "Loop": {"properties": {"next": {"$ref": "#/components/schemas/Loop"}}}}
    assert resolve_refs({"$ref": "#/components/schemas/A"}, components) == {"properties": {"b": {"type": "integer"}}}
    loop = resolve_refs({"$ref": "#/components/schemas/Loop"}, components)
    assert loop == {"properties": {"next": {"$ref": "#/components/schemas/Loop"}}}
    assert list(canonicalize({"b": 1, "a": {"d": 2, "c": [{"z": 1, "y": 2}]}})) == ["a", "b"]


def test_compile_is_byte_identical_and_valid():
    personas = build_graph().actors
    one = compile_app(lambda: create_app(), personas=personas).graph
    two = compile_app(lambda: create_app(), personas=reversed(personas)).graph
    assert one.to_json() == two.to_json()
    assert one.to_json(indent=None) == two.to_json(indent=None)
    assert type(one).from_json(one.to_json()) == one
    assert [a.id for a in one.actors] == sorted(a.id for a in personas)


def test_merge_prefers_runtime_schema_and_keeps_static_provenance():
    graph = compile_app(lambda: create_app()).graph
    withdraw = graph.edge("edge.withdraw")
    assert withdraw.payload_schema["properties"]["amount"]["exclusiveMinimum"] == 0
    assert withdraw.metadata["source"]["file"] == "cubic/gym/fintech/app.py"
    assert withdraw.metadata["evidence"] == ["openapi", "runtime", "static"]
    assert withdraw.metadata["writes_ledger"] and withdraw.metadata["guards"]["ownership"]
    assert {401, 403, 404, 422} <= withdraw.response_statuses
    assert graph.edge("edge.approve_loan").allowed_roles == {"ADMIN"}
    assert not graph.edge("edge.health").auth_required


@pytest.mark.parametrize("spec", [
    "cubic.gym.fintech.app:create_app",
    "cubic.gym.fintech.app",
    "src/cubic/gym/fintech/app.py",
    "src/cubic/gym/fintech/app.py:create_app",
])
def test_load_target_forms(spec):
    factory = load_target(spec)
    a, b = factory(), factory()
    assert a is not b and a.title == "Cubic Fintech Testbed"


def test_load_target_instance_and_errors(tmp_path):
    module = tmp_path / "svc_app.py"
    module.write_text("from fastapi import FastAPI\napp = FastAPI(title='Svc')\n")
    factory = load_target(str(module))
    assert factory() is factory() and factory().title == "Svc"
    with pytest.raises(TargetError, match="no such file"):
        load_target("missing/app.py")
    with pytest.raises(TargetError, match="cannot import"):
        load_target("not_a_real_module_xyz")
    with pytest.raises(TargetError, match="no ASGI app"):
        load_target("cubic.gym.fintech.app:does_not_exist")
