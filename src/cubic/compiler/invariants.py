"""Rule-based invariant synthesizer.

Derives a baseline set of executable ``InvariantAssertion`` objects purely from a
compiled ``StateGraph`` (plus the static ledger shape). No LLM is involved; the
rules are deliberately conservative and each one is phrased so that it *holds* on
a correct application. Status-code checks inspect step 0 of a probing trace.

Rules
-----
``inv.auth.<edge>``         authenticated edges reject anonymous callers (401).
``inv.tenant.<edge>``       parameterized resource edges reject non-owning actors (403/404).
``inv.input.<edge>.<f>``    money-like numeric fields reject non-positive values (400/422).
``inv.ledger.conservation`` balances reconcile with the ledger on every ledger-writing edge.
``inv.lifecycle.<edge>``    out-of-order lifecycle transitions are rejected (4xx).
``inv.functional.<edge>``   a valid request succeeds with a documented 2xx.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from cubic.compiler.static import LedgerSpec
from cubic.graph import (
    InvariantAssertion,
    InvariantCategory,
    LedgerConservationCheck,
    Severity,
    StateGraph,
    StatusCodeCheck,
    TransitionEdge,
)

MONEY_FIELD = re.compile(r"(^|_)(amount|qty|quantity|price|total|fee|value)$")
REJECTED_TRANSITION = frozenset({400, 403, 404, 409, 422})
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


def _suffix(edge: TransitionEdge) -> str:
    return edge.id.removeprefix("edge.")


def _describe(edge: TransitionEdge) -> str:
    return f"{edge.method.value} {edge.path}"


def _auth(edge: TransitionEdge) -> InvariantAssertion | None:
    if not edge.auth_required:
        return None
    return InvariantAssertion(
        id=f"inv.auth.{_suffix(edge)}", category=InvariantCategory.SECURITY, severity=Severity.HIGH,
        edge_ids=(edge.id,),
        description=f"Anonymous callers of {_describe(edge)} must be rejected with 401.",
        checks=(StatusCodeCheck(step=0, allowed=frozenset({401})),),
    )


def _tenant(edge: TransitionEdge) -> InvariantAssertion | None:
    resource_params = [p for p in edge.path_params if p.endswith("_id") or p == "id"]
    if not edge.auth_required or not resource_params:
        return None
    guards = edge.metadata.get("guards", {})
    if guards.get("ownership"):
        observed = "ownership guard found in source"
    elif guards.get("roles"):
        observed = f"no ownership guard; role guard {sorted(guards['roles'])} found in source"
    else:
        observed = "no ownership guard found in source"
    return InvariantAssertion(
        id=f"inv.tenant.{_suffix(edge)}", category=InvariantCategory.SECURITY,
        severity=Severity.CRITICAL if edge.method.value in _MUTATING else Severity.HIGH,
        edge_ids=(edge.id,),
        description=(
            f"An actor that does not own {{{resource_params[0]}}} must get 403/404 from "
            f"{_describe(edge)} ({observed})."
        ),
        checks=(StatusCodeCheck(step=0, allowed=frozenset({403, 404})),),
    )


def _inputs(edge: TransitionEdge) -> list[InvariantAssertion]:
    schema = edge.payload_schema or {}
    out = []
    for name, prop in sorted((schema.get("properties") or {}).items()):
        if prop.get("type") not in {"integer", "number"} or not MONEY_FIELD.search(name):
            continue
        constrained = any(k in prop for k in ("exclusiveMinimum", "minimum"))
        note = "schema enforces a lower bound" if constrained else "schema declares NO lower bound"
        out.append(InvariantAssertion(
            id=f"inv.input.{_suffix(edge)}.{name}", category=InvariantCategory.FINANCIAL,
            severity=Severity.CRITICAL, edge_ids=(edge.id,),
            description=f"Non-positive '{name}' on {_describe(edge)} must be rejected with 400/422 ({note}).",
            checks=(StatusCodeCheck(step=0, allowed=frozenset({400, 422})),),
        ))
    return out


def _lifecycle(edge: TransitionEdge) -> InvariantAssertion | None:
    required = edge.metadata.get("guards", {}).get("state_requires")
    if not required:
        return None
    return InvariantAssertion(
        id=f"inv.lifecycle.{_suffix(edge)}", category=InvariantCategory.SECURITY,
        severity=Severity.CRITICAL, edge_ids=(edge.id,),
        description=(
            f"{_describe(edge)} transitions {edge.source} -> {edge.target}; calling it while the "
            f"{edge.metadata.get('entity', 'entity')} is not {required} must be rejected (4xx)."
        ),
        checks=(StatusCodeCheck(step=0, allowed=REJECTED_TRANSITION),),
    )


def _functional(edge: TransitionEdge) -> InvariantAssertion:
    success = frozenset(s for s in edge.response_statuses if 200 <= s < 300) or frozenset({200})
    return InvariantAssertion(
        id=f"inv.functional.{_suffix(edge)}", category=InvariantCategory.FUNCTIONAL,
        severity=Severity.MEDIUM, edge_ids=(edge.id,),
        description=f"A valid {_describe(edge)} request succeeds with {sorted(success)}.",
        checks=(StatusCodeCheck(step=0, allowed=success),),
    )


def _ledger(edges: Iterable[TransitionEdge], ledger: LedgerSpec | None) -> InvariantAssertion | None:
    if ledger is None or not ledger.entity or not ledger.balance_field or ledger.amount_field != "amount":
        return None
    writers = tuple(sorted(e.id for e in edges if e.metadata.get("writes_ledger")))
    if not writers:
        return None
    return InvariantAssertion(
        id="inv.ledger.conservation", category=InvariantCategory.FINANCIAL, severity=Severity.CRITICAL,
        edge_ids=writers,
        description=(
            f"Every {ledger.entity}.{ledger.balance_field} equals the signed sum of "
            f"{ledger.collection} entries keyed by {ledger.account_field}, and never goes negative."
        ),
        checks=(LedgerConservationCheck(
            entity=ledger.entity, balance_field=ledger.balance_field, account_field=ledger.account_field,
        ),),
    )


def synthesize_invariants(
    graph: StateGraph, ledger: LedgerSpec | None = None
) -> tuple[InvariantAssertion, ...]:
    out: list[InvariantAssertion] = []
    for edge in graph.edges:
        out += [inv for inv in (_auth(edge), _tenant(edge), _lifecycle(edge)) if inv is not None]
        out += _inputs(edge)
        out.append(_functional(edge))
    if (conservation := _ledger(graph.edges, ledger)) is not None:
        out.append(conservation)
    return tuple(sorted(out, key=lambda inv: inv.id))
