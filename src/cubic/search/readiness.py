"""State readiness: is a persona actually able to exercise a test, given observed state?

These helpers read ``ActorPersona.observed_state`` (filled in by
``cubic.compiler.grounding``) and answer three questions about a concrete step:

* **Ledger readiness.** Do the conservation accounts the actor's request touches
  hold at least the requested amount? This prefers funded wallets for contention,
  for example.
* **Lifecycle state.** What state is the bound resource observed in? This
  replaces the assumption that seeded resources sit in the entity's initial state.
* **Precondition readiness.** Do the data preconditions extracted from the
  edge's guards (e.g. ``kyc_status == APPROVED``) hold? They are checked
  against records of the caller and of the owners of the bound resources.

Each answer is READY, NOT_READY or UNKNOWN. UNKNOWN means nothing relevant was
observed. Policies treat it exactly like the Milestone 3 behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from cubic.graph import ActorPersona, LedgerConservationCheck, StateGraph, TransitionEdge
from cubic.search.combinators import _entity_matches, _singular, required_state, resource_stem
from cubic.verifier.trace import ActionStep

LIFECYCLE_FIELDS = ("status", "state")


class Readiness(StrEnum):
    READY = "ready"
    NOT_READY = "not_ready"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Fact:
    status: Readiness
    reason: str = ""

    @property
    def ready(self) -> bool:
        return self.status is Readiness.READY


@dataclass(frozen=True)
class BoundRecord:
    """A resource referenced by a step, together with what its owner observed of it."""

    name: str        # path param or body field
    stem: str
    resource_id: str
    owner_id: str
    entity_key: str  # owned_resources key, e.g. "wallets"
    record: dict[str, Any]


def bound_records(graph: StateGraph, edge: TransitionEdge, step: ActionStep) -> list[BoundRecord]:
    """Observed records for every path param / ``*_id`` body field the step binds."""
    refs: list[tuple[str, str]] = list(step.path_params.items())
    for name, value in sorted((step.body or {}).items()):
        if isinstance(value, str) and resource_stem(name) is not None:
            refs.append((name, value))
    out: list[BoundRecord] = []
    for name, rid in refs:
        stem = resource_stem(name) or ""
        for owner in sorted(graph.actors, key=lambda a: a.id):
            for key, rows in owner.observed_state.items():
                if rid in rows and _entity_matches(key, stem, edge):
                    out.append(BoundRecord(name, stem, rid, owner.id, key, rows[rid]))
    return out


def _amount(step: ActionStep) -> float:
    numbers = [v for v in (step.body or {}).values()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return max(numbers, default=1)


def ledger_readiness(graph: StateGraph, edge: TransitionEdge, step: ActionStep) -> Fact:
    """Every conservation account the actor owns in this step can cover the amount."""
    checks = [c for inv in graph.invariants for c in inv.checks if isinstance(c, LedgerConservationCheck)]
    if not checks:
        return Fact(Readiness.UNKNOWN, "no ledger invariant")
    amount = _amount(step)
    own = [b for b in bound_records(graph, edge, step) if b.owner_id == step.actor_id]
    accounts = [(b, c) for b in own for c in checks if _singular(b.entity_key) == _singular(c.entity)]
    if not accounts:  # follow one reference hop, e.g. agent.tenant_id -> tenants[tenant_id]
        actor = graph.actor(step.actor_id)
        for bound in own:
            for check in checks:
                rid = bound.record.get(check.account_field)
                record = actor.observed(check.entity, rid) if isinstance(rid, str) else None
                if record is not None:
                    accounts.append((BoundRecord(check.account_field, _singular(check.entity), rid,
                                                 actor.id, check.entity, record), check))
    if not accounts:
        return Fact(Readiness.UNKNOWN, "no observed account bound by the actor")
    for bound, check in accounts:
        balance = bound.record.get(check.balance_field)
        if not isinstance(balance, (int, float)):
            return Fact(Readiness.UNKNOWN, f"{bound.resource_id}.{check.balance_field} not observed")
        if balance < amount:
            return Fact(Readiness.NOT_READY,
                        f"{bound.resource_id}.{check.balance_field}={balance} cannot cover {amount}")
    ids = ", ".join(sorted({b.resource_id for b, _ in accounts}))
    return Fact(Readiness.READY, f"funded account(s) {ids}")


def observed_state_of(graph: StateGraph, edge: TransitionEdge, step: ActionStep) -> str | None:
    """Observed lifecycle state of the resource the edge transitions, if any."""
    requirement = required_state(graph, edge)
    entity = requirement[0] if requirement else str(edge.metadata.get("entity") or "")
    for bound in bound_records(graph, edge, step):
        if bound.name not in step.path_params or _singular(bound.entity_key) != entity:
            continue
        for field in LIFECYCLE_FIELDS:
            value = bound.record.get(field)
            if isinstance(value, str):
                return value.upper()
    return None


def required_state_value(graph: StateGraph, edge: TransitionEdge) -> str | None:
    requirement = required_state(graph, edge)
    if requirement is None:
        return None
    state = graph.node(requirement[1]).state
    return state.upper() if state else None


def _context_actors(graph: StateGraph, edge: TransitionEdge, step: ActionStep) -> list[ActorPersona]:
    ids = [step.actor_id] + [b.owner_id for b in bound_records(graph, edge, step)]
    return [graph.actor(i) for i in dict.fromkeys(ids)]


def precondition_readiness(graph: StateGraph, edge: TransitionEdge, step: ActionStep) -> Fact:
    """Data preconditions from static guard metadata, checked against observed records."""
    preconditions = (edge.metadata.get("guards") or {}).get("preconditions") or []
    if not preconditions:
        return Fact(Readiness.READY, "no data preconditions")
    actors = _context_actors(graph, edge, step)
    pending: list[str] = []
    for pre in preconditions:
        field, expected = pre["field"], pre["value"]
        values = [
            row[field] for a in actors for rows in a.observed_state.values()
            for row in rows.values() if field in row
        ]
        if expected in values:
            continue
        if values:
            return Fact(Readiness.NOT_READY, f"observed {field}={values[0]!r}, edge requires {expected!r}")
        pending.append(field)
    if pending:
        return Fact(Readiness.UNKNOWN, f"precondition field(s) {', '.join(pending)} not observed")
    return Fact(Readiness.READY, "preconditions observed to hold")
