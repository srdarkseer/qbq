"""Behavioral State Graph: a typed, compact world model of an application.

``G = (V, E)`` where nodes are UI screens, API endpoints, or database entity
states, and edges are the HTTP actions that move between them. Actors are the
seeded personas that execute edges; invariants are the assertions the verifier
enforces.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cubic.graph.ids import Identifier, IntSet, StrSet
from cubic.graph.invariants import InvariantAssertion

SCHEMA_VERSION = "1"

_PATH_PARAM = re.compile(r"\{([^{}]*)\}")
_PARAM_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NodeKind(StrEnum):
    UI_SCREEN = "ui_screen"
    API_ENDPOINT = "api_endpoint"
    ENTITY_STATE = "entity_state"


class HttpMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class StateNode(_Frozen):
    """A vertex: a screen, an endpoint, or an entity in a specific state (``Order: PAID``)."""

    id: Identifier
    kind: NodeKind
    label: str = Field(min_length=1)
    entity: str | None = None
    state: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _entity_state_fields(self) -> StateNode:
        if self.kind is NodeKind.ENTITY_STATE:
            if not self.entity or not self.state:
                raise ValueError(f"entity_state node {self.id!r} requires 'entity' and 'state'")
        elif self.entity is not None or self.state is not None:
            raise ValueError(f"only entity_state nodes may set 'entity'/'state' ({self.id!r})")
        return self


class TransitionEdge(_Frozen):
    """A directed edge: an HTTP action that transitions ``source`` to ``target``."""

    id: Identifier
    source: Identifier
    target: Identifier
    method: HttpMethod
    path: str = Field(description="Path template, e.g. /wallets/{wallet_id}/withdraw")
    auth_required: bool = True
    allowed_roles: StrSet = Field(
        default_factory=frozenset,
        description="Roles the application is *intended* to permit; empty means any authenticated role.",
    )
    required_headers: StrSet = Field(default_factory=frozenset)
    payload_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema of the request body, if any."
    )
    latency_ms: float | None = Field(default=None, ge=0)
    response_statuses: IntSet = Field(
        default_factory=frozenset, description="HTTP statuses this edge is known to produce."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Provenance (source file/line) and extracted guard facts."
    )

    @field_validator("response_statuses")
    @classmethod
    def _valid_statuses(cls, value: frozenset[int]) -> frozenset[int]:
        bad = sorted(code for code in value if not 100 <= code <= 599)
        if bad:
            raise ValueError(f"invalid HTTP status codes: {bad}")
        return value

    @field_validator("path")
    @classmethod
    def _valid_path_template(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("path must start with '/'")
        names = _PATH_PARAM.findall(value)
        for name in names:
            if not _PARAM_NAME.match(name):
                raise ValueError(f"invalid path parameter name {name!r}")
        if len(set(names)) != len(names):
            raise ValueError("duplicate path parameter names")
        if "{" in _PATH_PARAM.sub("", value) or "}" in _PATH_PARAM.sub("", value):
            raise ValueError("unbalanced braces in path template")
        return value

    @model_validator(mode="after")
    def _auth_consistency(self) -> TransitionEdge:
        if not self.auth_required and self.allowed_roles:
            raise ValueError(f"edge {self.id!r}: unauthenticated edges cannot restrict roles")
        return self

    @property
    def path_params(self) -> tuple[str, ...]:
        return tuple(_PATH_PARAM.findall(self.path))

    def render_path(self, params: dict[str, str]) -> str:
        """Substitute path parameters; raises ``KeyError`` on missing/extra params."""
        expected = set(self.path_params)
        missing, extra = expected - params.keys(), params.keys() - expected
        if missing or extra:
            raise KeyError(
                f"edge {self.id!r} path params mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            )
        return _PATH_PARAM.sub(lambda m: str(params[m.group(1)]), self.path)


class ActorPersona(_Frozen):
    """A seeded identity the swarm acts as (``Unauthenticated``, ``User_A`` ...)."""

    id: Identifier
    display_name: str = Field(min_length=1)
    role: str = Field(min_length=1, description="Application role, e.g. USER / MERCHANT / ADMIN.")
    tenant_id: str | None = None
    auth_token: str | None = Field(default=None, repr=False)
    owned_resources: dict[str, StrSet] = Field(
        default_factory=dict,
        description="Resource ids this persona legitimately owns, keyed by entity name.",
    )
    observed_state: dict[str, dict[str, dict[str, Any]]] = Field(
        default_factory=dict,
        description=(
            "Initial attributes of owned resources observed at compile time, keyed by entity name "
            "then resource id (e.g. {'wallets': {'w_1': {'balance': 100}}}). Scalars only."
        ),
    )

    @property
    def is_authenticated(self) -> bool:
        return self.auth_token is not None

    def observed(self, entity: str, resource_id: str) -> dict[str, Any] | None:
        return self.observed_state.get(entity, {}).get(resource_id)

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}


class StateGraph(_Frozen):
    """The full compiled world model of one application."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    app_name: str = Field(min_length=1)
    nodes: tuple[StateNode, ...] = ()
    edges: tuple[TransitionEdge, ...] = ()
    actors: tuple[ActorPersona, ...] = ()
    invariants: tuple[InvariantAssertion, ...] = ()

    @model_validator(mode="after")
    def _referential_integrity(self) -> StateGraph:
        for label, items in (
            ("node", self.nodes),
            ("edge", self.edges),
            ("actor", self.actors),
            ("invariant", self.invariants),
        ):
            seen: set[str] = set()
            for item in items:
                if item.id in seen:
                    raise ValueError(f"duplicate {label} id {item.id!r}")
                seen.add(item.id)

        node_ids = {n.id for n in self.nodes}
        for edge in self.edges:
            for end in (edge.source, edge.target):
                if end not in node_ids:
                    raise ValueError(f"edge {edge.id!r} references unknown node {end!r}")

        edge_ids = {e.id for e in self.edges}
        for inv in self.invariants:
            unknown = [e for e in inv.edge_ids if e not in edge_ids]
            if unknown:
                raise ValueError(f"invariant {inv.id!r} references unknown edges {unknown}")
        return self

    # -- lookup helpers -------------------------------------------------------

    def node(self, node_id: str) -> StateNode:
        return _find(self.nodes, node_id, "node")

    def edge(self, edge_id: str) -> TransitionEdge:
        return _find(self.edges, edge_id, "edge")

    def actor(self, actor_id: str) -> ActorPersona:
        return _find(self.actors, actor_id, "actor")

    def invariant(self, invariant_id: str) -> InvariantAssertion:
        return _find(self.invariants, invariant_id, "invariant")

    def outgoing(self, node_id: str) -> tuple[TransitionEdge, ...]:
        self.node(node_id)
        return tuple(e for e in self.edges if e.source == node_id)

    def incoming(self, node_id: str) -> tuple[TransitionEdge, ...]:
        self.node(node_id)
        return tuple(e for e in self.edges if e.target == node_id)

    # -- serialization --------------------------------------------------------

    def to_json(self, *, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str | bytes) -> StateGraph:
        return cls.model_validate_json(data)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()


def _find(items: tuple[Any, ...], item_id: str, label: str) -> Any:
    for item in items:
        if item.id == item_id:
            return item
    raise KeyError(f"unknown {label} {item_id!r}")
