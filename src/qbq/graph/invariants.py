"""Invariant assertions: declarative, executable checks over a replayed trace.

Invariants are pure data. They never contain free-form code and are never judged
by an LLM; the deterministic verifier (``qbq.verifier``) evaluates each ``check``
against recorded HTTP responses and database snapshots.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qbq.graph.ids import Identifier, IntSet, StrSet


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class InvariantCategory(StrEnum):
    SECURITY = "security"
    FINANCIAL = "financial"
    FUNCTIONAL = "functional"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StatusCodeCheck(_Frozen):
    """Every response produced by trace step ``step`` must have a status in ``allowed``.

    Example (security): a cross-tenant read must be answered with 403/404.
    """

    kind: Literal["status_code"] = "status_code"
    step: int = Field(ge=0, description="Zero-based index of the trace step to inspect.")
    allowed: IntSet = Field(min_length=1)

    @field_validator("allowed")
    @classmethod
    def _valid_http_statuses(cls, value: frozenset[int]) -> frozenset[int]:
        bad = sorted(code for code in value if not 100 <= code <= 599)
        if bad:
            raise ValueError(f"invalid HTTP status codes: {bad}")
        return value


class LedgerConservationCheck(_Frozen):
    """Each account balance must equal the signed sum of its ledger entries.

    Example (financial): ``sum(credits) - sum(debits) == final_wallet_balance``.
    """

    kind: Literal["ledger_conservation"] = "ledger_conservation"
    entity: str = Field(default="wallets", min_length=1)
    balance_field: str = Field(default="balance", min_length=1)
    account_field: str = Field(
        default="wallet_id", min_length=1, description="Ledger entry field naming the account."
    )
    account_ids: StrSet | None = Field(
        default=None, description="Restrict the check to these accounts; None checks all."
    )
    allow_negative: bool = False


class StateGuardCheck(_Frozen):
    """If ``entity[entity_id].when_field == when_value`` then a precondition must hold.

    Example (state machine): a loan may only be ``DISBURSED`` if its borrower's
    KYC is ``APPROVED``.
    """

    kind: Literal["state_guard"] = "state_guard"
    entity: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    when_field: str = Field(min_length=1)
    when_value: Any
    require_entity: str = Field(min_length=1)
    require_id: str = Field(min_length=1)
    require_field: str = Field(min_length=1)
    require_value: Any


InvariantCheck = Annotated[
    StatusCodeCheck | LedgerConservationCheck | StateGuardCheck,
    Field(discriminator="kind"),
]


class InvariantAssertion(_Frozen):
    """A named business/security/functional invariant over the state graph."""

    id: Identifier
    category: InvariantCategory
    description: str = Field(min_length=1)
    severity: Severity = Severity.MEDIUM
    edge_ids: tuple[Identifier, ...] = Field(
        default=(), description="State-graph edges this invariant constrains."
    )
    checks: tuple[InvariantCheck, ...] = Field(
        min_length=1, description="All checks must hold for the invariant to hold."
    )
    tags: tuple[str, ...] = Field(
        default=(), description="Taxonomy references, e.g. OWASP-API1:2023, OWASP-LLM06:2025, CWE-639."
    )

    @model_validator(mode="after")
    def _unique_edge_refs(self) -> InvariantAssertion:
        if len(set(self.edge_ids)) != len(self.edge_ids):
            raise ValueError(f"invariant {self.id!r} references an edge more than once")
        return self
