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

AI / agent endpoints (tool calling, MCP-style gateways, LLM completions):

``inv.tool_authz.<edge>``      a mutating edge that names a tool (``tool_id``, ``tool_name``...)
                               rejects tools not granted to the caller's tenant (403/404).
``inv.ai_bounds.<edge>.<f>``   generation, agent-loop and rate parameters (``max_tokens``,
                               ``max_steps``, ``timeout``...) reject values above a sane bound
                               (400/413/422).

Every invariant carries OWASP API Security 2023 / OWASP LLM 2025 / CWE ``tags``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from qbq.compiler.static import LedgerSpec
from qbq.graph import (
    InvariantAssertion,
    InvariantCategory,
    LedgerConservationCheck,
    Severity,
    StateGraph,
    StatusCodeCheck,
    TransitionEdge,
)

MONEY_FIELD = re.compile(r"(^|_)(amount|qty|quantity|price|total|fee|value)$")
AI_LIMIT_FIELD = re.compile(
    r"(^|_)(max_tokens|max_new_tokens|max_output_tokens|max_completion_tokens|max_steps|max_iterations"
    r"|max_turns|max_tool_calls|top_k|num_results|n_results|requests_per_minute|rpm|rate_limit"
    r"|timeout|timeout_s|timeout_seconds)$"
)
TOOL_FIELD = re.compile(r"^(tool|tool_id|tool_name|function_name)$")
AI_BOUND_STATUSES = frozenset({400, 413, 422})

TAGS = {
    "auth": ("OWASP-API2:2023", "CWE-306"),
    "tenant": ("OWASP-API1:2023", "CWE-639"),
    "input": ("CWE-20",),
    "ledger": ("OWASP-API6:2023",),
    "lifecycle": ("OWASP-API6:2023", "CWE-841"),
    "tool_authz": ("OWASP-LLM06:2025", "OWASP-API5:2023", "CWE-863"),
    "ai_bounds": ("OWASP-LLM10:2025", "OWASP-API4:2023", "CWE-770"),
}
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
        checks=(StatusCodeCheck(step=0, allowed=frozenset({401})),), tags=TAGS["auth"],
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
        checks=(StatusCodeCheck(step=0, allowed=frozenset({403, 404})),), tags=TAGS["tenant"],
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
            checks=(StatusCodeCheck(step=0, allowed=frozenset({400, 422})),), tags=TAGS["input"],
        ))
    return out


def _tool_authz(edge: TransitionEdge) -> InvariantAssertion | None:
    if not edge.auth_required or edge.method.value not in _MUTATING:
        return None
    fields = sorted(
        {p for p in edge.path_params if TOOL_FIELD.match(p)}
        | {n for n in ((edge.payload_schema or {}).get("properties") or {}) if TOOL_FIELD.match(n)}
    )
    if not fields:
        return None
    return InvariantAssertion(
        id=f"inv.tool_authz.{_suffix(edge)}", category=InvariantCategory.SECURITY, severity=Severity.CRITICAL,
        edge_ids=(edge.id,),
        description=(
            f"Invoking a tool that is not granted to the caller's tenant via '{fields[0]}' on "
            f"{_describe(edge)} must be rejected with 403/404 (cross-tenant tool authorization)."
        ),
        checks=(StatusCodeCheck(step=0, allowed=frozenset({403, 404})),), tags=TAGS["tool_authz"],
    )


def _ai_bounds(edge: TransitionEdge) -> list[InvariantAssertion]:
    schema = edge.payload_schema or {}
    out = []
    for name, prop in sorted((schema.get("properties") or {}).items()):
        if prop.get("type") not in {"integer", "number"} or not AI_LIMIT_FIELD.search(name):
            continue
        capped = any(k in prop for k in ("maximum", "exclusiveMaximum"))
        note = "schema enforces an upper bound" if capped else "schema declares NO upper bound"
        out.append(InvariantAssertion(
            id=f"inv.ai_bounds.{_suffix(edge)}.{name}", category=InvariantCategory.SECURITY,
            severity=Severity.HIGH, edge_ids=(edge.id,),
            description=(
                f"Values of '{name}' on {_describe(edge)} above a sane bound must be rejected with "
                f"400/413/422 ({note}; unbounded consumption)."
            ),
            checks=(StatusCodeCheck(step=0, allowed=AI_BOUND_STATUSES),), tags=TAGS["ai_bounds"],
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
        checks=(StatusCodeCheck(step=0, allowed=REJECTED_TRANSITION),), tags=TAGS["lifecycle"],
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
        tags=TAGS["ledger"],
    )


def synthesize_invariants(
    graph: StateGraph, ledger: LedgerSpec | None = None
) -> tuple[InvariantAssertion, ...]:
    out: list[InvariantAssertion] = []
    for edge in graph.edges:
        out += [inv for inv in (_auth(edge), _tenant(edge), _lifecycle(edge), _tool_authz(edge)) if inv]
        out += _inputs(edge)
        out += _ai_bounds(edge)
        out.append(_functional(edge))
    if (conservation := _ledger(graph.edges, ledger)) is not None:
        out.append(conservation)
    return tuple(sorted(out, key=lambda inv: inv.id))
