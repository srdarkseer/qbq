"""Structural fidelity of a compiled graph against a reference (hand-written) graph.

Edges are matched by ``(method, path)``; ids are free to differ. The structural
score averages the dimensions a compiler must get right. Role agreement is
reported separately because "intended" roles are often policy, not code.
"""

from __future__ import annotations

from pydantic import BaseModel

from qbq.graph import NodeKind, StateGraph, TransitionEdge

RouteKey = tuple[str, str]


class FidelityReport(BaseModel):
    reference_routes: int
    compiled_routes: int
    route_coverage: float
    missing_routes: list[str]
    extra_routes: list[str]
    auth_agreement: float
    path_param_agreement: float
    payload_presence_agreement: float
    lifecycle_agreement: float
    lifecycle_mismatches: list[str]
    role_agreement: float
    role_mismatches: list[str]
    structural_fidelity: float


def _key(edge: TransitionEdge) -> RouteKey:
    return edge.method.value, edge.path


def _fmt(key: RouteKey) -> str:
    return f"{key[0]} {key[1]}"


def _ratio(hits: int, total: int) -> float:
    return hits / total if total else 1.0


def _state(graph: StateGraph, node_id: str, lifecycle_entities: set[str]) -> tuple[str, str] | None:
    node = graph.node(node_id)
    if node.kind is NodeKind.ENTITY_STATE and node.entity in lifecycle_entities:
        return node.entity, node.state  # type: ignore[return-value]
    return None


def _effective_roles(edge: TransitionEdge, universe: frozenset[str]) -> frozenset[str]:
    if not edge.auth_required:
        return frozenset()
    return edge.allowed_roles or universe


def compare_graphs(reference: StateGraph, compiled: StateGraph) -> FidelityReport:
    ref = {_key(e): e for e in reference.edges}
    comp = {_key(e): e for e in compiled.edges}
    shared = sorted(ref.keys() & comp.keys())

    # Entities with a real lifecycle (two or more states) in the reference graph.
    states: dict[str, set[str]] = {}
    for node in reference.nodes:
        if node.kind is NodeKind.ENTITY_STATE and node.entity and node.state:
            states.setdefault(node.entity, set()).add(node.state)
    lifecycle_entities = {e for e, s in states.items() if len(s) > 1}

    lifecycle_total = lifecycle_hits = 0
    lifecycle_mismatches: list[str] = []
    for key in shared:
        r, c = ref[key], comp[key]
        expected = (_state(reference, r.source, lifecycle_entities), _state(reference, r.target, lifecycle_entities))
        if expected == (None, None):
            continue
        lifecycle_total += 1
        actual = (_state(compiled, c.source, lifecycle_entities), _state(compiled, c.target, lifecycle_entities))
        if actual == expected:
            lifecycle_hits += 1
        else:
            lifecycle_mismatches.append(f"{_fmt(key)}: expected {expected}, got {actual}")

    universe = frozenset(a.role for a in reference.actors if a.is_authenticated)
    role_mismatches = [
        f"{_fmt(k)}: reference {sorted(_effective_roles(ref[k], universe))}, "
        f"compiled {sorted(_effective_roles(comp[k], universe))}"
        for k in shared
        if _effective_roles(ref[k], universe) != _effective_roles(comp[k], universe)
    ]

    coverage = _ratio(len(shared), len(ref))
    auth = _ratio(sum(ref[k].auth_required == comp[k].auth_required for k in shared), len(shared))
    params = _ratio(sum(ref[k].path_params == comp[k].path_params for k in shared), len(shared))
    payload = _ratio(
        sum((ref[k].payload_schema is None) == (comp[k].payload_schema is None) for k in shared), len(shared)
    )
    lifecycle = _ratio(lifecycle_hits, lifecycle_total)
    return FidelityReport(
        reference_routes=len(ref),
        compiled_routes=len(comp),
        route_coverage=coverage,
        missing_routes=[_fmt(k) for k in sorted(ref.keys() - comp.keys())],
        extra_routes=[_fmt(k) for k in sorted(comp.keys() - ref.keys())],
        auth_agreement=auth,
        path_param_agreement=params,
        payload_presence_agreement=payload,
        lifecycle_agreement=lifecycle,
        lifecycle_mismatches=lifecycle_mismatches,
        role_agreement=_ratio(len(shared) - len(role_mismatches), len(shared)),
        role_mismatches=role_mismatches,
        structural_fidelity=(coverage + auth + params + payload + lifecycle) / 5,
    )
