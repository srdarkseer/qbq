"""Merge the static and dynamic passes into one canonical ``StateGraph``.

Join key: ``(source file, first decorator line)`` binds a live route to the
static route that declared it. When that fails (e.g. ``add_api_route``), the
join falls back to ``(method, path)``. If there is still no match, the route
compiles from dynamic evidence alone.

The dynamic pass is authoritative for paths, methods, and payload schemas (that is
what the server really enforces). The static pass contributes guards, state
effects, and provenance. Output is sorted and canonicalized, so compiling the same
app twice yields byte-identical JSON.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from cubic.compiler.dynamic import DynamicObservation, DynamicRoute, aintrospect, canonicalize
from cubic.compiler.invariants import synthesize_invariants
from cubic.compiler.static import (
    StaticAnalysis,
    StaticRoute,
    analyze_sources,
    edge_endpoints,
    node_for,
    route_metadata,
)
from cubic.graph import ActorPersona, StateGraph, StateNode, TransitionEdge

AppFactory = Callable[[], Any]


class CompileReport(BaseModel):
    app_name: str
    source_files: list[str]
    routes_total: int
    routes_compiled: int
    route_coverage: float
    static_routes: int
    static_matched: int
    dynamic_only_routes: list[str]
    static_unmatched_routes: list[str]
    nodes: int
    edges: int
    invariants: int
    invariants_by_category: dict[str, int]
    source_bytes: int
    openapi_bytes: int
    graph_bytes: int
    structural_graph_bytes: int
    compression_ratio: float
    timings_s: dict[str, float]


@dataclass(frozen=True)
class CompileResult:
    graph: StateGraph
    report: CompileReport
    static: StaticAnalysis
    dynamic: DynamicObservation


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug or "app"


def _edge_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    return slug if slug and slug[0].isalpha() else f"x_{slug}"


def _join(
    dynamic: Iterable[DynamicRoute], static: Iterable[StaticRoute]
) -> list[tuple[DynamicRoute, StaticRoute | None]]:
    by_line = {(r.location.file, r.decorator_line): r for r in static}
    by_route = {(r.method, r.path): r for r in static}
    pairs = []
    for d in dynamic:
        match = by_line.get((d.source_file, d.first_line))
        if match is None or match.method is not d.method:
            match = by_route.get((d.method, d.path))
        pairs.append((d, match))
    return pairs


def _assign_ids(pairs: list[tuple[DynamicRoute, StaticRoute | None]]) -> list[str]:
    counts: dict[str, int] = {}
    ids = []
    for d, s in pairs:
        base = f"edge.{_edge_slug(s.location.function if s else d.name)}"
        counts[base] = counts.get(base, 0) + 1
        ids.append(base if counts[base] == 1 else f"{base}_{d.method.value.lower()}_{counts[base]}")
    return ids


def merge_edge(d: DynamicRoute, s: StaticRoute | None, analysis: StaticAnalysis, edge_id: str) -> TransitionEdge:
    required = s.required_state if s else None
    produced = s.produced_state if s else None
    source, target = edge_endpoints(d.path, required, produced)

    static_auth = bool(s and s.auth_dependency)
    observed = d.observed_unauthenticated_status
    auth = static_auth or observed == 401
    statuses = set(d.documented_statuses) | ({observed} if observed else set())
    if s:
        statuses |= set(s.raised_statuses) | {s.status_code or 200}

    payload = d.body_schema
    if payload is None and s and s.body_model and (model := analysis.model(s.body_model)):
        payload = model.json_schema()

    if s:
        metadata = route_metadata(s)
    else:
        metadata = {"source": {"file": d.source_file, "function": d.name, "line": d.first_line}}
    evidence = {"openapi"} if d.operation_id else set()
    evidence |= {"runtime"} if observed is not None else set()
    evidence |= {"static"} if s else set()
    metadata |= {
        "evidence": sorted(evidence),
        "observed_unauthenticated_status": observed,
        "operation_id": d.operation_id,
    }
    if static_auth and observed is not None and observed not in {401, 403}:
        metadata["auth_conflict"] = True  # source declares auth but the server let us in

    headers = {"Authorization"} if auth else set()
    headers |= {h.title() for h in d.header_params if h != "authorization"}
    return TransitionEdge(
        id=edge_id, source=source, target=target, method=d.method, path=d.path,
        auth_required=auth,
        allowed_roles=frozenset(s.role_guards) if (s and auth) else frozenset(),
        required_headers=frozenset(headers),
        payload_schema=canonicalize(payload) if payload is not None else None,
        response_statuses=frozenset(statuses),
        metadata=canonicalize(metadata),
    )


def build_graph(
    dynamic: DynamicObservation,
    static: StaticAnalysis,
    *,
    app_name: str | None = None,
    personas: Iterable[ActorPersona] = (),
    synthesize: bool = True,
) -> tuple[StateGraph, list[tuple[DynamicRoute, StaticRoute | None]]]:
    pairs = _join(dynamic.routes, static.routes)
    edges = [merge_edge(d, s, static, eid) for (d, s), eid in zip(pairs, _assign_ids(pairs), strict=True)]
    nodes: dict[str, StateNode] = {}
    for edge in edges:
        for node_id in (edge.source, edge.target):
            nodes.setdefault(node_id, node_for(node_id, edge.path))
    graph = StateGraph(
        app_name=app_name or _slug(dynamic.app_title),
        nodes=tuple(sorted(nodes.values(), key=lambda n: n.id)),
        edges=tuple(sorted(edges, key=lambda e: e.id)),
        actors=tuple(sorted(personas, key=lambda a: a.id)),
    )
    if synthesize:
        graph = graph.model_copy(update={"invariants": synthesize_invariants(graph, static.ledger)})
        graph = StateGraph.model_validate(graph.model_dump())  # re-run referential checks
    return graph, pairs


async def acompile_app(
    app_factory: AppFactory,
    *,
    app_name: str | None = None,
    personas: Iterable[ActorPersona] = (),
    synthesize: bool = True,
    probe: bool = True,
) -> CompileResult:
    t0 = time.perf_counter()
    dynamic = await aintrospect(app_factory, probe=probe)
    t1 = time.perf_counter()
    sources = []
    for display, path in dynamic.source_files:
        with open(path, encoding="utf-8") as fh:
            sources.append((display, fh.read()))
    static = analyze_sources(sources)
    t2 = time.perf_counter()
    graph, pairs = build_graph(dynamic, static, app_name=app_name, personas=personas, synthesize=synthesize)
    t3 = time.perf_counter()

    matched = {id(s) for _, s in pairs if s is not None}
    structural = graph.model_copy(update={"invariants": ()})
    graph_bytes = len(graph.to_json(indent=None).encode())
    by_category: dict[str, int] = {}
    for inv in graph.invariants:
        by_category[inv.category.value] = by_category.get(inv.category.value, 0) + 1
    report = CompileReport(
        app_name=graph.app_name,
        source_files=list(static.files),
        routes_total=len(dynamic.routes),
        routes_compiled=len(graph.edges),
        route_coverage=len(graph.edges) / len(dynamic.routes) if dynamic.routes else 1.0,
        static_routes=len(static.routes),
        static_matched=len(matched),
        dynamic_only_routes=sorted(f"{d.method.value} {d.path}" for d, s in pairs if s is None),
        static_unmatched_routes=sorted(
            f"{s.method.value} {s.path}" for s in static.routes if id(s) not in matched
        ),
        nodes=len(graph.nodes),
        edges=len(graph.edges),
        invariants=len(graph.invariants),
        invariants_by_category=dict(sorted(by_category.items())),
        source_bytes=static.source_bytes,
        openapi_bytes=dynamic.openapi_bytes,
        graph_bytes=graph_bytes,
        structural_graph_bytes=len(structural.to_json(indent=None).encode()),
        compression_ratio=static.source_bytes / graph_bytes if graph_bytes else 0.0,
        timings_s={
            "dynamic": t1 - t0, "static": t2 - t1, "merge_and_synthesis": t3 - t2, "total": t3 - t0,
        },
    )
    return CompileResult(graph=graph, report=report, static=static, dynamic=dynamic)


def compile_app(app_factory: AppFactory, **kwargs: Any) -> CompileResult:
    import asyncio

    return asyncio.run(acompile_app(app_factory, **kwargs))
