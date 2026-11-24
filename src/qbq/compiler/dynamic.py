"""Dynamic pass: introspect a live ASGI app in-process (no network).

For a fresh app instance the introspector:

1. runs the ASGI lifespan startup (so apps that build state on startup work);
2. fetches the app's OpenAPI 3.x document over an in-process ASGI transport and
   resolves every ``$ref`` into a self-contained request-body schema;
3. walks the live route table to bind each operation to its endpoint function
   (module, file, first line) so it can be joined with the static pass;
4. optionally sends one unauthenticated probe per route, each against its own
   fresh instance, to observe the real auth behaviour (e.g. 401).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from qbq.graph import HttpMethod

AppFactory = Callable[[], Any]
PROBE_PLACEHOLDER = "qbq-probe"
_BASE_URL = "http://qbq-introspect"
_PATH_PARAM = re.compile(r"\{([^{}:]+)(?::[^{}]*)?\}")
_DROP_KEYS = frozenset({"title"})


@dataclass(frozen=True)
class DynamicRoute:
    method: HttpMethod
    path: str
    name: str
    operation_id: str | None
    module: str | None
    source_file: str | None  # display name, e.g. "qbq/gym/fintech/app.py"
    source_path: str | None  # absolute path on disk
    first_line: int | None  # co_firstlineno (first decorator line)
    path_params: tuple[str, ...]
    header_params: tuple[str, ...]
    body_schema: dict[str, Any] | None
    documented_statuses: frozenset[int]
    observed_unauthenticated_status: int | None = None

    @property
    def success_statuses(self) -> frozenset[int]:
        return frozenset(s for s in self.documented_statuses if 200 <= s < 300)


@dataclass(frozen=True)
class DynamicObservation:
    app_title: str
    openapi_version: str | None
    openapi_bytes: int
    routes: tuple[DynamicRoute, ...]

    @property
    def source_files(self) -> tuple[tuple[str, str], ...]:
        """Unique ``(display_name, absolute_path)`` pairs for every endpoint module."""
        pairs = {(r.source_file, r.source_path) for r in self.routes if r.source_file and r.source_path}
        return tuple(sorted(pairs))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- lifespan


@contextlib.asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """Minimal ASGI lifespan driver. Apps that do not support lifespan are tolerated."""
    to_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    from_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    scope = {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}, "state": {}}

    async def run() -> None:
        await app(scope, to_app.get, from_app.put)

    task = asyncio.create_task(run())

    async def exchange(message: str) -> bool:
        await to_app.put({"type": message})
        getter = asyncio.create_task(from_app.get())
        done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
        if getter not in done:  # app exited: lifespan unsupported
            getter.cancel()
            return False
        reply = getter.result()
        if reply["type"].endswith(".failed"):
            raise RuntimeError(f"ASGI {message} failed: {reply.get('message', '')}")
        return True

    supported = False
    try:
        supported = await exchange("lifespan.startup")
        yield
    finally:
        if supported:
            await exchange("lifespan.shutdown")
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await task


# --------------------------------------------------------------------------- schema utils


def canonicalize(value: Any) -> Any:
    """Recursively sort dict keys so equal structures serialize byte-identically."""
    if isinstance(value, dict):
        return {k: canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, list | tuple):
        return [canonicalize(v) for v in value]
    return value


def resolve_refs(schema: Any, components: dict[str, Any], _stack: tuple[str, ...] = ()) -> Any:
    """Inline ``#/components/schemas/...`` refs and drop presentation-only keys."""
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            name = ref.rsplit("/", 1)[-1]
            if name in _stack:  # recursive model: keep the ref
                return {"$ref": ref}
            return resolve_refs(copy.deepcopy(components.get(name, {})), components, (*_stack, name))
        return {k: resolve_refs(v, components, _stack) for k, v in schema.items() if k not in _DROP_KEYS}
    if isinstance(schema, list):
        return [resolve_refs(v, components, _stack) for v in schema]
    return schema


def module_display_name(module: str | None, source_path: str | None) -> str | None:
    if module and source_path:
        suffix = "/__init__.py" if Path(source_path).name == "__init__.py" else ".py"
        if module != "__main__":
            return module.replace(".", "/") + suffix
    return Path(source_path).name if source_path else None


def _routes_of(app: Any) -> list[Any]:
    router = getattr(app, "router", app)
    return list(getattr(router, "routes", []) or getattr(app, "routes", []))


def _endpoint_facts(endpoint: Any) -> tuple[str | None, str | None, str | None, int | None]:
    fn = inspect.unwrap(endpoint)
    module = getattr(fn, "__module__", None)
    try:
        path = inspect.getsourcefile(fn)
    except TypeError:
        path = None
    code = getattr(fn, "__code__", None)
    line = code.co_firstlineno if code is not None else None
    abspath = str(Path(path).resolve()) if path else None
    return module, module_display_name(module, abspath), abspath, line


# --------------------------------------------------------------------------- introspection


async def _fetch_openapi(app: Any) -> tuple[dict[str, Any] | None, int]:
    url = getattr(app, "openapi_url", "/openapi.json")
    if url:
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url=_BASE_URL) as client:
                response = await client.get(url)
            if response.status_code == 200:
                return response.json(), len(response.content)
        except Exception:  # broken or non-HTTP app: fall through to app.openapi()
            pass
    if callable(getattr(app, "openapi", None)):
        doc = app.openapi()
        return doc, len(json.dumps(doc).encode())
    return None, 0


async def _probe_unauthenticated(app_factory: AppFactory, method: HttpMethod, path: str) -> int:
    app = app_factory()
    concrete = _PATH_PARAM.sub(PROBE_PLACEHOLDER, path)
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=_BASE_URL) as client:
            response = await client.request(method.value, concrete)
    return response.status_code


async def aintrospect(app_factory: AppFactory, *, probe: bool = True) -> DynamicObservation:
    app = app_factory()
    async with lifespan(app):
        doc, doc_bytes = await _fetch_openapi(app)
    doc = doc or {}
    components = doc.get("components", {}).get("schemas", {})
    paths = doc.get("paths", {})

    routes: list[DynamicRoute] = []
    for route in _routes_of(app):
        methods = getattr(route, "methods", None)
        endpoint = getattr(route, "endpoint", None)
        path = getattr(route, "path", None)
        if not methods or endpoint is None or path is None or not getattr(route, "include_in_schema", True):
            continue
        module, display, abspath, line = _endpoint_facts(endpoint)
        for raw in sorted(methods):
            try:
                method = HttpMethod(raw.upper())
            except ValueError:  # HEAD / OPTIONS
                continue
            op = paths.get(path, {}).get(method.value.lower(), {})
            params = op.get("parameters", [])
            body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
            statuses = frozenset(int(code) for code in op.get("responses", {}) if str(code).isdigit())
            routes.append(DynamicRoute(
                method=method, path=path, name=getattr(route, "name", None) or endpoint.__name__,
                operation_id=op.get("operationId"), module=module, source_file=display,
                source_path=abspath, first_line=line,
                path_params=tuple(_PATH_PARAM.findall(path)),
                header_params=tuple(sorted(p["name"].lower() for p in params if p.get("in") == "header")),
                body_schema=canonicalize(resolve_refs(body, components)) if body else None,
                documented_statuses=statuses,
            ))

    if probe:
        observed = await asyncio.gather(*(_probe_unauthenticated(app_factory, r.method, r.path) for r in routes))
        routes = [
            DynamicRoute(**{**r.__dict__, "observed_unauthenticated_status": status})
            for r, status in zip(routes, observed, strict=True)
        ]

    routes.sort(key=lambda r: (r.path, r.method.value))
    return DynamicObservation(
        app_title=str(getattr(app, "title", "") or type(app).__name__),
        openapi_version=doc.get("openapi"),
        openapi_bytes=doc_bytes,
        routes=tuple(routes),
    )


def introspect(app_factory: AppFactory, *, probe: bool = True) -> DynamicObservation:
    return asyncio.run(aintrospect(app_factory, probe=probe))
