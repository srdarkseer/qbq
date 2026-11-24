"""Static compiler on multi-module FastAPI projects: routers, prefixes, Depends chains, schemas."""

import textwrap

import pytest

from qbq.compiler import analyze_sources, compile_app, extract_candidates
from qbq.compiler.sources import collect_sources, scope_root
from qbq.graph import HttpMethod


def _src(**files):
    return [(name.replace("__", "/") + ".py", textwrap.dedent(text)) for name, text in files.items()]


PROJECT = _src(
    app__deps='''
        from typing import Annotated
        from fastapi import Depends, Header, HTTPException

        def get_current_user(authorization: Annotated[str | None, Header()] = None):
            if authorization is None:
                raise HTTPException(401, "auth")
            return {"id": "u", "role": "USER"}

        CurrentUser = Annotated[dict, Depends(get_current_user)]

        def require_admin(user: CurrentUser):
            if user["role"] != "ADMIN":
                raise HTTPException(403, "admin only")
            return user

        def require_role(*roles):
            def checker(user: CurrentUser):
                if user["role"] not in roles:
                    raise HTTPException(403, "forbidden")
                return user
            return checker

        def get_owned_doc(doc_id: str, user: CurrentUser):
            doc = DOCS.get(doc_id)
            if doc is None or doc["owner_id"] != user["id"]:
                raise HTTPException(404, "nope")
            return doc

        OwnedDoc = Annotated[dict, Depends(get_owned_doc)]
    ''',
    app__schemas='''
        from typing import Annotated
        from pydantic import BaseModel, Field

        class DocBase(BaseModel):
            title: str = Field(min_length=1)

        class DocCreate(DocBase):
            pages: Annotated[int, Field(gt=0, le=500)]
            tags: list[str] = Field(default_factory=list)
    ''',
    app__routers__docs='''
        from fastapi import APIRouter, Depends
        from ..deps import CurrentUser, OwnedDoc, require_admin
        from ..schemas import DocCreate

        router = APIRouter(prefix="/docs", tags=["docs"])

        @router.post("", status_code=201)
        async def create_doc(body: DocCreate, user: CurrentUser):
            return {}

        @router.get("/{doc_id}")
        async def read_doc(doc: OwnedDoc):
            return doc

        @router.get("/{doc_id}/raw")
        async def read_raw(doc_id: str, user: CurrentUser):
            return {}

        @router.delete("/{doc_id}", dependencies=[Depends(require_admin)])
        async def purge_doc(doc_id: str):
            return {}
    ''',
    app__routers__admin='''
        import app.deps as deps
        from fastapi import APIRouter, Depends

        router = APIRouter(prefix="/admin", dependencies=[Depends(deps.require_role("OPS", "ADMIN"))])
        audit = APIRouter(prefix="/audit")

        @audit.get("/events")
        async def list_events():
            return []

        router.include_router(audit)

        @router.post("/reindex")
        async def reindex():
            return {}
    ''',
    app__main='''
        from fastapi import FastAPI
        from .routers import admin
        from .routers.docs import router as docs_router

        API = "/api" + "/v2"

        def create_app():
            app = FastAPI()
            app.include_router(docs_router, prefix=API)
            app.include_router(admin.router, prefix=API)
            return app
    ''',
)


@pytest.fixture(scope="module")
def analysis():
    return analyze_sources(PROJECT)


def _route(analysis, method, path):
    return next(r for r in analysis.routes if r.method is method and r.path == path)


def test_prefixes_from_routers_mounts_and_nested_routers(analysis):
    assert sorted((r.method.value, r.path) for r in analysis.routes) == [
        ("DELETE", "/api/v2/docs/{doc_id}"),
        ("GET", "/api/v2/admin/audit/events"),
        ("GET", "/api/v2/docs/{doc_id}"),
        ("GET", "/api/v2/docs/{doc_id}/raw"),
        ("POST", "/api/v2/admin/reindex"),
        ("POST", "/api/v2/docs"),
    ]
    assert {e.path for e in extract_candidates(analysis)[1]} == {r.path for r in analysis.routes}


def test_imported_aliases_and_transitive_dependency_chains(analysis):
    create = _route(analysis, HttpMethod.POST, "/api/v2/docs")
    assert create.auth_dependency == "get_current_user" and 401 in create.raised_statuses
    read = _route(analysis, HttpMethod.GET, "/api/v2/docs/{doc_id}")
    assert read.auth_dependency == "get_owned_doc"  # authenticated through get_owned_doc -> get_current_user
    assert read.ownership_guard is True  # `or`-guard inside the dependency
    raw = _route(analysis, HttpMethod.GET, "/api/v2/docs/{doc_id}/raw")
    assert raw.auth_dependency == "get_current_user" and raw.ownership_guard is False


def test_roles_from_decorator_router_and_factory_dependencies(analysis):
    purge = _route(analysis, HttpMethod.DELETE, "/api/v2/docs/{doc_id}")
    assert purge.role_guards == {"ADMIN"} and purge.auth_dependency == "require_admin"
    reindex = _route(analysis, HttpMethod.POST, "/api/v2/admin/reindex")
    assert reindex.role_guards == {"OPS", "ADMIN"} and reindex.auth_dependency == "require_role"
    events = _route(analysis, HttpMethod.GET, "/api/v2/admin/audit/events")
    assert events.role_guards == {"OPS", "ADMIN"}  # inherited from the parent router


def test_pydantic_v2_models_across_files_with_inheritance(analysis):
    assert _route(analysis, HttpMethod.POST, "/api/v2/docs").body_model == "DocCreate"
    schema = analysis.model("DocCreate").json_schema()
    assert schema["properties"]["pages"] == {"type": "integer", "exclusiveMinimum": 0, "maximum": 500}
    assert schema["properties"]["title"] == {"type": "string", "minLength": 1}  # inherited from DocBase
    assert schema["required"] == ["title", "pages"]
    assert analysis.dependency_statuses["get_owned_doc"] == {401, 404}


def test_single_file_behaviour_is_unchanged():
    single = analyze_sources(_src(svc='''
        from fastapi import FastAPI, Depends, HTTPException
        app = FastAPI()
        def user(token: str = ""):
            if not token:
                raise HTTPException(401, "x")
            return {"id": "u"}
        @app.get("/items/{item_id}")
        async def item(item_id: str, u: dict = Depends(user)):
            return {}
    '''))
    (route,) = single.routes
    assert route.path == "/items/{item_id}" and route.auth_dependency == "user"


# --------------------------------------------------------------------------- on disk, end to end


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    for name, text in PROJECT:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for package in ("app", "app/routers"):
        (root / package / "__init__.py").write_text("")
    # make the DOCS store importable for runtime; deps references a module-level name
    deps = root / "app/deps.py"
    deps.write_text("DOCS = {}\n" + deps.read_text())
    monkeypatch.syspath_prepend(str(root))
    return root


def test_compile_multi_router_app_end_to_end(pkg):
    from app.main import create_app

    result = compile_app(create_app, app_module="app.main", probe=True)
    r = result.report
    assert r.routes_total == r.routes_compiled == 6 and r.route_coverage == 1.0
    assert r.dynamic_only_routes == [] and r.static_unmatched_routes == []
    assert set(r.source_files) == {"app/deps.py", "app/main.py", "app/routers/__init__.py",
                                   "app/routers/admin.py", "app/routers/docs.py", "app/schemas.py"}
    g = result.graph
    by_path = {(e.method.value, e.path): e for e in g.edges}
    assert by_path[("GET", "/api/v2/docs/{doc_id}")].metadata["guards"]["ownership"] is True
    assert by_path[("POST", "/api/v2/admin/reindex")].allowed_roles == {"OPS", "ADMIN"}
    assert "inv.tenant.read_raw" in {i.id for i in g.invariants}


def test_source_scope_never_leaves_the_application(pkg):
    endpoint = [("app/routers/docs.py", str(pkg / "app/routers/docs.py"))]
    found = {d for d, _ in collect_sources(endpoint)}
    assert "app/deps.py" in found and "app/schemas.py" in found
    assert not any(d.startswith(("fastapi", "pydantic", "qbq")) for d in found)
    assert scope_root("app.routers.v1.docs") == "app" and scope_root("qbq.gym.fintech.app") == "qbq.gym.fintech"
    assert scope_root("demo_shop") == ""
