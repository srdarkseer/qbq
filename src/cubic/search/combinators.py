"""Graph transition combinators: typed test-case generators over a ``StateGraph``.

Every combinator turns one ``TransitionEdge`` plus one or more ``ActorPersona``
contexts into a ``ProposedTest``: a replayable ``ActionTrace`` paired with the
``InvariantAssertion`` it is meant to break. Combinators are deterministic and
use only facts that are explicit in the graph: auth requirements, ``allowed_roles``,
path and payload schemas, lifecycle nodes, and the resources each persona owns.
Heuristic judgments, such as whether an oracle is ambiguous for some persona,
belong to the search policy (``cubic.search.policy``). They do not belong here.

Combinators
-----------
``cross_persona_check``    replay persona A's request with persona B's credentials,
                           keeping A's resource ids (tenant isolation / auth).
``parallel_contention``    fire N barrier-synchronized copies of a state-mutating
                           edge (ledger conservation under concurrency).
``lifecycle_order_check``  call a downstream lifecycle edge on a resource still in an
                           earlier state, skipping prerequisite edges.
``boundary_perturb``       substitute boundary values into typed payload fields
                           (input validation) or send the nominal request (functional).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Iterator
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cubic.graph import (
    ActorPersona,
    InvariantAssertion,
    LedgerConservationCheck,
    NodeKind,
    StateGraph,
    StateGuardCheck,
    StatusCodeCheck,
    TransitionEdge,
)
from cubic.graph.ids import Identifier
from cubic.verifier.trace import ActionStep, ActionTrace, CandidateFinding

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
CONTENTION_LEVELS: tuple[int, ...] = (8, 16)
_WRONG_TYPE = "not-a-number"
_MISSING = object()  # boundary sentinel: remove the field from the body


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- oracles


class Oracle(StrEnum):
    """What an invariant expects a probing request to observe, derived from its checks."""

    UNAUTHENTICATED = "unauthenticated"          # status in {401}
    FORBIDDEN = "forbidden"                      # status in {403, 404}
    REJECTED_INPUT = "rejected_input"            # status in {400, 422}
    REJECTED_TRANSITION = "rejected_transition"  # any 4xx including 409
    SUCCESS = "success"                          # 2xx only
    CONSERVATION = "conservation"                # ledger reconciliation
    STATE_GUARD = "state_guard"                  # entity-state precondition
    UNSUPPORTED = "unsupported"


def classify_invariant(invariant: InvariantAssertion) -> Oracle:
    """Classify an invariant from its check structure alone (never its id or text)."""
    status = [c for c in invariant.checks if isinstance(c, StatusCodeCheck)]
    if status:
        if len({c.step for c in status}) != 1:
            return Oracle.UNSUPPORTED
        allowed = frozenset().union(*(c.allowed for c in status))
        if allowed == {401}:
            return Oracle.UNAUTHENTICATED
        if allowed <= {403, 404}:
            return Oracle.FORBIDDEN
        if allowed <= {400, 422}:
            return Oracle.REJECTED_INPUT
        if all(400 <= s < 500 for s in allowed) and 409 in allowed:
            return Oracle.REJECTED_TRANSITION
        if all(200 <= s < 300 for s in allowed):
            return Oracle.SUCCESS
        return Oracle.UNSUPPORTED
    if any(isinstance(c, LedgerConservationCheck) for c in invariant.checks):
        return Oracle.CONSERVATION
    if any(isinstance(c, StateGuardCheck) for c in invariant.checks):
        return Oracle.STATE_GUARD
    return Oracle.UNSUPPORTED


def probe_step(invariant: InvariantAssertion) -> int:
    """Trace index the invariant inspects (0 when it only inspects final DB state)."""
    return max((c.step for c in invariant.checks if isinstance(c, StatusCodeCheck)), default=0)


# --------------------------------------------------------------------------- model


class CombinatorKind(StrEnum):
    CROSS_PERSONA = "cross_persona"
    PARALLEL_CONTENTION = "parallel_contention"
    LIFECYCLE_ORDER = "lifecycle_order"
    BOUNDARY_PERTURB = "boundary_perturb"


class BoundaryCase(StrEnum):
    NOMINAL = "nominal"
    MIN_VALID = "min_valid"
    MAX_VALID = "max_valid"
    NEGATIVE = "negative"
    ZERO = "zero"
    BELOW_MIN = "below_min"
    ABOVE_MAX = "above_max"
    MISSING = "missing"
    EMPTY = "empty"
    WRONG_TYPE = "wrong_type"

    @property
    def valid(self) -> bool:
        return self in _VALID_CASES


_VALID_CASES = frozenset({BoundaryCase.NOMINAL, BoundaryCase.MIN_VALID, BoundaryCase.MAX_VALID})

COMBINATOR_ORACLES: dict[CombinatorKind, frozenset[Oracle]] = {
    CombinatorKind.CROSS_PERSONA: frozenset({Oracle.UNAUTHENTICATED, Oracle.FORBIDDEN}),
    CombinatorKind.PARALLEL_CONTENTION: frozenset({Oracle.CONSERVATION}),
    CombinatorKind.LIFECYCLE_ORDER: frozenset({Oracle.REJECTED_TRANSITION, Oracle.STATE_GUARD}),
    CombinatorKind.BOUNDARY_PERTURB: frozenset({Oracle.REJECTED_INPUT, Oracle.SUCCESS}),
}


class ProposedTest(_Frozen):
    """A synthesized test: a trace hypothesized to violate ``invariant_id``.

    ``features`` holds the facts the combinator used, such as personas, resource
    bindings and boundary values. Search policies score from these features and do
    not re-derive them.
    """

    id: Identifier
    combinator: CombinatorKind
    invariant_id: Identifier
    oracle: Oracle
    edge_id: Identifier
    node_id: Identifier
    trace: ActionTrace
    hypothesis: str = Field(min_length=1)
    features: dict[str, Any] = Field(default_factory=dict)

    @property
    def request_count(self) -> int:
        return sum(step.concurrency for step in self.trace.steps)

    @property
    def actor_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(step.actor_id for step in self.trace.steps))

    def to_candidate(self) -> CandidateFinding:
        return CandidateFinding(
            id=self.id, invariant_id=self.invariant_id, hypothesis=self.hypothesis, trace=self.trace
        )


# --------------------------------------------------------------------------- resource binding


def _singular(name: str) -> str:
    if name.endswith("ies"):
        return name[:-3] + "y"
    if name.endswith("ses") or name.endswith("xes"):
        return name[:-2]
    return name[:-1] if name.endswith("s") else name


def resource_stem(param: str) -> str | None:
    """``wallet_id`` -> ``wallet``; ``from_wallet_id`` -> ``from_wallet``; ``id`` -> ``""``."""
    if param == "id":
        return ""
    return param[: -len("_id")] if param.endswith("_id") else None


def _entity_matches(owned_key: str, stem: str, edge: TransitionEdge) -> bool:
    singular = _singular(owned_key)
    if stem == "":  # bare {id}: resolve through the edge's entity
        stem = str(edge.metadata.get("entity") or "")
    return bool(stem) and (stem == singular or stem.endswith("_" + singular))


def owned_ids(actor: ActorPersona, stem: str, edge: TransitionEdge) -> tuple[str, ...]:
    ids: set[str] = set()
    for key, values in actor.owned_resources.items():
        if _entity_matches(key, stem, edge):
            ids |= values
    return tuple(sorted(ids))


def all_resources(graph: StateGraph, stem: str, edge: TransitionEdge) -> tuple[tuple[str, str], ...]:
    """Every seeded ``(owner_actor_id, resource_id)`` pair for a resource stem."""
    return tuple(
        sorted((a.id, rid) for a in graph.actors for rid in owned_ids(a, stem, edge))
    )


def is_legitimate_principal(edge: TransitionEdge, actor: ActorPersona) -> bool:
    """Whether the graph says ``actor`` may call ``edge`` at all."""
    if edge.auth_required and not actor.is_authenticated:
        return False
    return not edge.allowed_roles or actor.role in edge.allowed_roles


def _bind_path(graph: StateGraph, edge: TransitionEdge, actor: ActorPersona) -> dict[str, str] | None:
    """Bind path params to the actor's own resources.

    If the edge is role-guarded and the actor owns nothing that fits, bind to another
    persona's resource instead. Admin-style edges such as approvals act on
    resources the caller does not own.
    """
    params: dict[str, str] = {}
    for name in edge.path_params:
        stem = resource_stem(name)
        if stem is None:
            return None
        own = owned_ids(actor, stem, edge)
        if own:
            params[name] = own[0]
            continue
        if not edge.allowed_roles:
            return None
        others = all_resources(graph, stem, edge)
        if not others:
            return None
        params[name] = others[0][1]
    return params


def _owns_bound(actor: ActorPersona, edge: TransitionEdge, params: dict[str, str]) -> bool:
    return bool(params) and all(
        rid in owned_ids(actor, resource_stem(name) or "", edge) for name, rid in params.items()
    )


def _numeric_bounds(prop: dict[str, Any]) -> tuple[float | None, bool, float | None, bool]:
    """Returns ``(lower, lower_exclusive, upper, upper_exclusive)``."""
    lower, lower_ex = prop.get("minimum"), False
    if "exclusiveMinimum" in prop and isinstance(prop["exclusiveMinimum"], (int, float)):
        lower, lower_ex = prop["exclusiveMinimum"], True
    upper, upper_ex = prop.get("maximum"), False
    if "exclusiveMaximum" in prop and isinstance(prop["exclusiveMaximum"], (int, float)):
        upper, upper_ex = prop["exclusiveMaximum"], True
    return lower, lower_ex, upper, upper_ex


def _min_valid(prop: dict[str, Any]) -> int | float:
    lower, lower_ex, _, _ = _numeric_bounds(prop)
    if lower is None:
        return 1  # unbounded numeric field: the smallest positive quantity
    return math.floor(lower) + 1 if lower_ex else math.ceil(lower)


def _max_valid(prop: dict[str, Any]) -> int | float | None:
    _, _, upper, upper_ex = _numeric_bounds(prop)
    if upper is None:
        return None
    return math.ceil(upper) - 1 if upper_ex else math.floor(upper)


def _properties(edge: TransitionEdge) -> dict[str, dict[str, Any]]:
    return dict(sorted(((edge.payload_schema or {}).get("properties") or {}).items()))


def _required(edge: TransitionEdge) -> frozenset[str]:
    return frozenset((edge.payload_schema or {}).get("required") or ())


def _nominal_body(
    graph: StateGraph, edge: TransitionEdge, actor: ActorPersona
) -> dict[str, Any] | None:
    """A schema-valid body for ``actor``.

    The first ``*_id`` field gets one of the actor's own resources. Each later
    ``*_id`` field gets a different resource of the same kind, normally a peer's
    resource, as needed for a transfer destination.
    """
    if edge.payload_schema is None:
        return None
    body: dict[str, Any] = {}
    used: set[str] = set()
    for name, prop in _properties(edge).items():
        kind = prop.get("type")
        if "enum" in prop and prop["enum"]:
            body[name] = prop["enum"][0]
        elif kind in {"integer", "number"}:
            body[name] = _min_valid(prop)
        elif kind == "boolean":
            body[name] = True
        elif kind == "string" and (stem := resource_stem(name)) is not None:
            own = [r for r in owned_ids(actor, stem, edge) if r not in used]
            pool = own if not used else [
                rid for _, rid in all_resources(graph, stem, edge) if rid not in used
            ]
            if not pool:
                if name in _required(edge):
                    return {"__unbindable__": name}
                continue
            body[name] = pool[0]
            used.add(pool[0])
        elif kind == "string":
            body[name] = "cubic"
        elif name in _required(edge):
            return {"__unbindable__": name}
    return body


def nominal_step(graph: StateGraph, edge: TransitionEdge, actor: ActorPersona) -> ActionStep | None:
    """The request ``actor`` would legitimately send on ``edge`` (a "recorded" step)."""
    if not is_legitimate_principal(edge, actor):
        return None
    params = _bind_path(graph, edge, actor)
    if params is None:
        return None
    body = _nominal_body(graph, edge, actor)
    if body is not None and "__unbindable__" in body:
        return None
    return ActionStep(edge_id=edge.id, actor_id=actor.id, path_params=params, body=body)


def _observation_step(
    graph: StateGraph, edge: TransitionEdge, actor: ActorPersona, params: dict[str, str]
) -> ActionStep | None:
    """A read-only GET on the same resource, used for a control or observation step."""
    if not params:
        return None
    for other in graph.edges:
        if (
            other.method.value == "GET"
            and set(other.path_params) == set(params)
            and is_legitimate_principal(other, actor)
        ):
            return ActionStep(edge_id=other.id, actor_id=actor.id, path_params=dict(params))
    return None


# --------------------------------------------------------------------------- lifecycle helpers


def _entity_nodes(graph: StateGraph, entity: str) -> dict[str, Any]:
    return {n.id: n for n in graph.nodes if n.kind is NodeKind.ENTITY_STATE and n.entity == entity}


def initial_state_node(graph: StateGraph, entity: str) -> str | None:
    """The entity state reached by a creation edge, from outside the entity lifecycle."""
    nodes = _entity_nodes(graph, entity)
    entries = sorted(
        e.target for e in graph.edges if e.target in nodes and e.source not in nodes
    )
    return entries[0] if entries else None


def prerequisite_path(graph: StateGraph, start: str, goal: str) -> tuple[str, ...] | None:
    """Shortest edge-id path ``start -> goal`` through the graph (``()`` if equal)."""
    if start == goal:
        return ()
    queue: deque[tuple[str, tuple[str, ...]]] = deque([(start, ())])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        for edge in sorted(graph.outgoing(node), key=lambda e: e.id):
            if edge.target in seen:
                continue
            if edge.target == goal:
                return (*path, edge.id)
            seen.add(edge.target)
            queue.append((edge.target, (*path, edge.id)))
    return None


def required_state(graph: StateGraph, edge: TransitionEdge) -> tuple[str, str] | None:
    """``(entity, source_node_id)`` if ``edge`` only makes sense from one entity state.

    That is a transition out of an entity state, or a self-loop on one guarded by an
    explicit state requirement (e.g. ``upgrade`` requires ACTIVE and stays ACTIVE).
    """
    source = graph.node(edge.source)
    if source.kind is not NodeKind.ENTITY_STATE:
        return None
    if edge.source == edge.target and not (edge.metadata.get("guards") or {}).get("state_requires"):
        return None
    return source.entity or "", source.id


# --------------------------------------------------------------------------- combinators


def _tid(kind: CombinatorKind, edge: TransitionEdge, invariant: InvariantAssertion, *parts: object) -> str:
    detail = ".".join(str(p).replace("/", "_") for p in parts if p not in (None, ""))
    return f"auto:{kind.value}:{invariant.id}:{edge.id.removeprefix('edge.')}:{detail}"[:128]


def _make(
    kind: CombinatorKind, graph: StateGraph, edge: TransitionEdge, invariant: InvariantAssertion,
    steps: Iterable[ActionStep], hypothesis: str, features: dict[str, Any], *id_parts: object,
) -> ProposedTest | None:
    steps = tuple(steps)
    oracle = classify_invariant(invariant)
    if oracle not in COMBINATOR_ORACLES[kind] or edge.id not in invariant.edge_ids:
        return None
    if probe_step(invariant) >= len(steps) or steps[probe_step(invariant)].edge_id != edge.id:
        return None  # the invariant must inspect the probing request itself
    return ProposedTest(
        id=_tid(kind, edge, invariant, *id_parts), combinator=kind, invariant_id=invariant.id,
        oracle=oracle, edge_id=edge.id, node_id=edge.source, trace=ActionTrace(steps=steps),
        hypothesis=hypothesis, features=features,
    )


def cross_persona_check(
    graph: StateGraph, edge: TransitionEdge, invariant: InvariantAssertion,
    owner: ActorPersona, intruder: ActorPersona,
) -> ProposedTest | None:
    """Replay ``owner``'s recorded request as ``intruder`` while keeping ``owner``'s ids.

    An unauthenticated intruder tests auth invariants (401). An authenticated
    intruder that does not own the bound resources tests tenant isolation
    (403/404). For read-only edges a second step replays the request as the owner,
    as a differential control.
    """
    if owner.id == intruder.id:
        return None
    recorded = nominal_step(graph, edge, owner)
    if recorded is None:
        return None
    oracle = classify_invariant(invariant)
    if oracle is Oracle.UNAUTHENTICATED and intruder.is_authenticated:
        return None
    if oracle is Oracle.FORBIDDEN:
        if not intruder.is_authenticated or not recorded.path_params:
            return None
        if _owns_bound(intruder, edge, recorded.path_params):
            return None  # the intruder legitimately owns the resource: not an isolation test
    steps = [recorded.model_copy(update={"actor_id": intruder.id})]
    if edge.method.value == "GET":
        steps.append(recorded)
    features = {
        "owner": owner.id, "owner_role": owner.role, "intruder": intruder.id,
        "intruder_role": intruder.role, "intruder_authenticated": intruder.is_authenticated,
        "resources": dict(recorded.path_params), "control_step": len(steps) > 1,
    }
    ids = ",".join(f"{k}={v}" for k, v in sorted(recorded.path_params.items()))
    hypothesis = (
        f"Replay {owner.id}'s {edge.method.value} {edge.path}"
        + (f" ({ids})" if ids else "")
        + f" as {intruder.id}; expect {invariant.id} to break."
    )
    return _make(CombinatorKind.CROSS_PERSONA, graph, edge, invariant, steps, hypothesis, features,
                 owner.id, intruder.id)


def parallel_contention(
    graph: StateGraph, edge: TransitionEdge, invariant: InvariantAssertion,
    actor: ActorPersona, concurrency: int,
) -> ProposedTest | None:
    """Fire ``concurrency`` barrier-synchronized copies of a state-mutating edge.

    The amount is the smallest valid one. A check-then-act race breaks
    conservation however small the amount is, because each lost update leaves
    one more ledger entry than the balance reflects.
    """
    if edge.method.value not in MUTATING_METHODS or concurrency < 2:
        return None
    nominal = nominal_step(graph, edge, actor)
    if nominal is None:
        return None
    steps = [nominal.model_copy(update={"concurrency": concurrency})]
    if (observe := _observation_step(graph, edge, actor, nominal.path_params)) is not None:
        steps.append(observe)
    features = {
        "actor": actor.id, "concurrency": concurrency, "resources": dict(nominal.path_params),
        "body": nominal.body, "observation_step": len(steps) > 1,
    }
    hypothesis = (
        f"Fire {concurrency} barrier-synchronized {edge.method.value} {edge.path} as {actor.id}; "
        f"concurrent writes should desynchronize balances from the ledger."
    )
    return _make(CombinatorKind.PARALLEL_CONTENTION, graph, edge, invariant, steps, hypothesis,
                 features, actor.id, f"x{concurrency}")


def lifecycle_order_check(
    graph: StateGraph, edge: TransitionEdge, invariant: InvariantAssertion, actor: ActorPersona,
) -> ProposedTest | None:
    """Invoke a downstream lifecycle edge on a resource still in its initial state.

    Seeded resources are assumed to be in the entity's initial state, the node a
    creation edge leads to. The prerequisite edges between that state and
    ``edge.source`` are skipped on purpose.
    """
    requirement = required_state(graph, edge)
    if requirement is None:
        return None
    entity, source = requirement
    initial = initial_state_node(graph, entity)
    if initial is None:
        return None
    skipped = prerequisite_path(graph, initial, source)
    if not skipped:
        return None  # nothing to skip: the seeded state already satisfies the edge
    nominal = nominal_step(graph, edge, actor)
    if nominal is None or not nominal.path_params:
        return None
    features = {
        "actor": actor.id, "resources": dict(nominal.path_params), "from_node": initial,
        "skipped_edges": list(skipped), "actor_owns_resource": _owns_bound(actor, edge, nominal.path_params),
    }
    hypothesis = (
        f"Call {edge.method.value} {edge.path} as {actor.id} while the {entity} is still at "
        f"{initial}, skipping {', '.join(skipped)}."
    )
    return _make(CombinatorKind.LIFECYCLE_ORDER, graph, edge, invariant, [nominal], hypothesis,
                 features, actor.id, *nominal.path_params.values())


def boundary_values(prop: dict[str, Any], required: bool) -> dict[BoundaryCase, Any]:
    """Boundary inputs for one typed field (``_MISSING`` marks field removal)."""
    out: dict[BoundaryCase, Any] = {}
    kind = prop.get("type")
    if kind in {"integer", "number"}:
        lower, lower_ex, upper, _ = _numeric_bounds(prop)

        def rejected(value: float) -> bool:
            # values a sane lower bound would reject: below it, or non-positive if unbounded
            if lower is None:
                return value <= 0
            return value <= lower if lower_ex else value < lower

        if rejected(-1):
            out[BoundaryCase.NEGATIVE] = -1
        if rejected(0):
            out[BoundaryCase.ZERO] = 0
        if lower is not None:
            below = math.floor(lower) if lower_ex else math.ceil(lower) - 1
            if below not in (-1, 0):
                out[BoundaryCase.BELOW_MIN] = below
        out[BoundaryCase.MIN_VALID] = _min_valid(prop)
        if (top := _max_valid(prop)) is not None:
            out[BoundaryCase.MAX_VALID] = top
            out[BoundaryCase.ABOVE_MAX] = top + 1
        out[BoundaryCase.WRONG_TYPE] = _WRONG_TYPE
    elif kind == "string":
        out[BoundaryCase.EMPTY] = ""
    elif kind == "boolean":
        out[BoundaryCase.WRONG_TYPE] = _WRONG_TYPE
    if required:
        out[BoundaryCase.MISSING] = _MISSING
    return out



def boundary_perturb(
    graph: StateGraph, edge: TransitionEdge, invariant: InvariantAssertion, actor: ActorPersona,
    field: str | None, case: BoundaryCase,
) -> ProposedTest | None:
    """Send ``actor``'s nominal request with ``field`` set to a ``case`` boundary value.

    ``field=None`` with ``BoundaryCase.NOMINAL`` sends the unmodified request: the
    functional baseline for every edge. Invalid cases target input-rejection
    invariants. Valid cases target functional success invariants.
    """
    nominal = nominal_step(graph, edge, actor)
    if nominal is None:
        return None
    oracle = classify_invariant(invariant)
    if (oracle is Oracle.SUCCESS) != case.valid:
        return None
    body = None if nominal.body is None else dict(nominal.body)
    value: Any = None
    if field is None:
        if case is not BoundaryCase.NOMINAL:
            return None
    else:
        if body is None or field not in _properties(edge):
            return None
        prop = _properties(edge)[field]
        values = boundary_values(prop, field in _required(edge))
        if case not in values:
            return None
        value = values[case]
        if value is _MISSING:
            body.pop(field, None)
            value = None
        else:
            body[field] = value
    step = nominal.model_copy(update={"body": body})
    features = {
        "actor": actor.id, "field": field, "case": case.value, "value": value,
        "resources": dict(nominal.path_params),
    }
    if field is None:
        hypothesis = f"Nominal {edge.method.value} {edge.path} as {actor.id} fails to succeed."
    else:
        shown = "<missing>" if case is BoundaryCase.MISSING else repr(value)
        hypothesis = (
            f"{edge.method.value} {edge.path} as {actor.id} with {field}={shown} ({case.value}); "
            f"expect {invariant.id} to break."
        )
    return _make(CombinatorKind.BOUNDARY_PERTURB, graph, edge, invariant, [step], hypothesis,
                 features, actor.id, field or "", case.value)


# --------------------------------------------------------------------------- enumeration


def _actors(graph: StateGraph) -> tuple[ActorPersona, ...]:
    return tuple(sorted(graph.actors, key=lambda a: a.id))


def expand(
    graph: StateGraph, kind: CombinatorKind, edge: TransitionEdge, invariant: InvariantAssertion,
) -> tuple[ProposedTest, ...]:
    """Every distinct concrete test combinator ``kind`` yields for one (edge, invariant) frame.

    Tests whose traces are identical (e.g. anonymous replays of two owners' identical
    requests) are collapsed onto the first one generated.
    """
    seen: dict[str, ProposedTest] = {}
    for test in _expand(graph, kind, edge, invariant):
        if test is not None:
            seen.setdefault(test.trace.model_dump_json(), test)
    return tuple(seen.values())


def _expand(
    graph: StateGraph, kind: CombinatorKind, edge: TransitionEdge, invariant: InvariantAssertion,
) -> Iterator[ProposedTest | None]:
    actors = _actors(graph)
    if kind is CombinatorKind.CROSS_PERSONA:
        for owner in actors:
            for intruder in actors:
                yield cross_persona_check(graph, edge, invariant, owner, intruder)
    elif kind is CombinatorKind.PARALLEL_CONTENTION:
        for actor in actors:
            for n in CONTENTION_LEVELS:
                yield parallel_contention(graph, edge, invariant, actor, n)
    elif kind is CombinatorKind.LIFECYCLE_ORDER:
        for actor in actors:
            yield lifecycle_order_check(graph, edge, invariant, actor)
    elif kind is CombinatorKind.BOUNDARY_PERTURB:
        props = _properties(edge)
        for actor in actors:
            yield boundary_perturb(graph, edge, invariant, actor, None, BoundaryCase.NOMINAL)
            for name, prop in props.items():
                cases = boundary_values(prop, name in _required(edge))
                for case in cases:
                    yield boundary_perturb(graph, edge, invariant, actor, name, case)


def applicable_combinators(invariant: InvariantAssertion) -> tuple[CombinatorKind, ...]:
    oracle = classify_invariant(invariant)
    return tuple(k for k in CombinatorKind if oracle in COMBINATOR_ORACLES[k])


def generate_pool(graph: StateGraph) -> tuple[ProposedTest, ...]:
    """The full, unranked candidate universe: every combinator over every frame.

    Ids are unique, and so is every ``(invariant, trace)`` pair: when two frames yield
    the same test, only the first is kept.
    """
    seen: dict[tuple[str, str], ProposedTest] = {}
    for invariant in sorted(graph.invariants, key=lambda i: i.id):
        for edge_id in invariant.edge_ids:
            edge = graph.edge(edge_id)
            for kind in applicable_combinators(invariant):
                for test in expand(graph, kind, edge, invariant):
                    seen.setdefault((test.invariant_id, test.trace.model_dump_json()), test)
    return tuple(seen.values())
