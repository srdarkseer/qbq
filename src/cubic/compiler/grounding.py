"""Persona grounding: observe each persona's seeded resources at compile time.

The search needs to know which persona is actually *ready* to exercise an edge:
a funded wallet for withdrawal contention, a subscription that really is in its
initial state for a lifecycle skip, a verified billing profile for activation.
Persona definitions only say which resources each persona owns. This pass fills
in what those resources look like.

For every authenticated persona, and every read-only (``GET``) edge whose path
parameters all bind to resources that persona owns, the pass sends one
authenticated request to a fresh app instance and records the scalar fields of a
``200`` JSON object response. Records go into ``ActorPersona.observed_state``.
Field names that look like credentials are dropped. Only safe methods are used,
and nothing is sent outside the process.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from typing import Any

import httpx

from cubic.compiler.dynamic import lifespan
from cubic.graph import ActorPersona, StateGraph, TransitionEdge
from cubic.search.combinators import _entity_matches, resource_stem

AppFactory = Callable[[], Any]
_BASE_URL = "http://cubic-ground"
# credential-like *names* (``api_token``, ``password_hash``, ``client_secret``), not every field
# containing the substring: ``token_balance`` and ``max_tokens_limit`` are state, not secrets.
_SECRET_FIELD = re.compile(
    r"(^|_)(secret|token|password|passwd|hash|salt|credentials?|private_key|api_key|apikey|bearer)$",
    re.IGNORECASE,
)
_SCALARS = (str, int, float, bool, type(None))


def _owned_bindings(actor: ActorPersona, edge: TransitionEdge) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Every ``(path_params, {param: owned_resource_key})`` binding of ``edge`` to ``actor``."""
    choices: list[list[tuple[str, str, str]]] = []  # per param: (param, resource id, owned key)
    for param in edge.path_params:
        stem = resource_stem(param)
        if stem is None:
            return []
        options = sorted(
            (param, rid, key)
            for key, ids in actor.owned_resources.items() if _entity_matches(key, stem, edge)
            for rid in ids
        )
        if not options:
            return []
        choices.append(options)
    return [
        ({p: rid for p, rid, _ in combo}, {p: key for p, _, key in combo})
        for combo in itertools.product(*choices)
    ]


def _scalar_fields(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    return {
        k: v for k, v in sorted(body.items())
        if isinstance(v, _SCALARS) and not _SECRET_FIELD.search(k)
    }


async def aground_personas(app_factory: AppFactory, graph: StateGraph) -> tuple[ActorPersona, ...]:
    """Return ``graph.actors`` with ``observed_state`` filled in from live reads."""
    reads = sorted(
        (e for e in graph.edges if e.method.value == "GET" and e.path_params),
        key=lambda e: e.id,
    )
    personas = sorted(graph.actors, key=lambda a: a.id)
    plan = [
        (actor, edge, params, keys)
        for actor in personas if actor.is_authenticated
        for edge in reads
        for params, keys in _owned_bindings(actor, edge)
    ]
    observed: dict[str, dict[str, dict[str, dict[str, Any]]]] = {a.id: {} for a in personas}
    if plan:
        app = app_factory()
        async with lifespan(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=_BASE_URL) as client:
                for actor, edge, params, keys in plan:
                    response = await client.get(edge.render_path(params), headers=actor.auth_headers())
                    if response.status_code != 200:
                        continue
                    try:
                        fields = _scalar_fields(response.json())
                    except ValueError:
                        continue
                    if not fields:
                        continue
                    # a single-param read describes that resource
                    if len(params) == 1:
                        (param, rid), = params.items()
                        slot = observed[actor.id].setdefault(keys[param], {}).setdefault(rid, {})
                        slot.update(fields)
    return tuple(
        actor.model_copy(update={"observed_state": {
            key: dict(sorted(rows.items())) for key, rows in sorted(observed[actor.id].items())
        }}) if observed[actor.id] else actor
        for actor in graph.actors
    )
