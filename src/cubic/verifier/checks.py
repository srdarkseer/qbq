"""Deterministic evaluation of invariant checks. No LLM is ever consulted here."""

from __future__ import annotations

from collections.abc import Sequence
from functools import singledispatch
from typing import Any

from pydantic import BaseModel

from cubic.graph import (
    InvariantAssertion,
    LedgerConservationCheck,
    StateGuardCheck,
    StatusCodeCheck,
)
from cubic.verifier.snapshot import DbSnapshot


class Exchange(BaseModel):
    """One recorded HTTP request/response pair (HAR-like)."""

    step: int
    edge_id: str
    actor_id: str
    method: str
    path: str
    request_body: Any = None
    status_code: int
    response_body: Any = None


StepExchanges = Sequence[Sequence[Exchange]]


class CheckOutcome(BaseModel):
    holds: bool
    detail: str


@singledispatch
def evaluate_check(check: Any, exchanges: StepExchanges, after: DbSnapshot) -> CheckOutcome:
    raise TypeError(f"no evaluator for check type {type(check).__name__}")


@evaluate_check.register
def _(check: StatusCodeCheck, exchanges: StepExchanges, after: DbSnapshot) -> CheckOutcome:
    responses = exchanges[check.step]
    statuses = [ex.status_code for ex in responses]
    bad = [s for s in statuses if s not in check.allowed]
    if bad:
        return CheckOutcome(
            holds=False,
            detail=f"step {check.step}: {len(bad)}/{len(statuses)} responses had status "
            f"{sorted(set(bad))}, allowed {sorted(check.allowed)}",
        )
    return CheckOutcome(holds=True, detail=f"step {check.step}: statuses {sorted(set(statuses))} allowed")


@evaluate_check.register
def _(check: LedgerConservationCheck, exchanges: StepExchanges, after: DbSnapshot) -> CheckOutcome:
    accounts = after.entities.get(check.entity, {})
    ids = sorted(accounts if check.account_ids is None else check.account_ids)
    problems: list[str] = []
    for account_id in ids:
        row = accounts.get(account_id)
        if row is None:
            problems.append(f"{check.entity}[{account_id}] missing")
            continue
        balance = row[check.balance_field]
        ledger_sum = sum(e["amount"] for e in after.ledger if e.get(check.account_field) == account_id)
        if balance != ledger_sum:
            problems.append(f"{account_id}: balance {balance} != ledger sum {ledger_sum}")
        if not check.allow_negative and balance < 0:
            problems.append(f"{account_id}: negative balance {balance}")
    if problems:
        return CheckOutcome(holds=False, detail="; ".join(problems))
    return CheckOutcome(holds=True, detail=f"{len(ids)} accounts reconcile with ledger")


@evaluate_check.register
def _(check: StateGuardCheck, exchanges: StepExchanges, after: DbSnapshot) -> CheckOutcome:
    row = after.entities.get(check.entity, {}).get(check.entity_id)
    if row is None or row.get(check.when_field) != check.when_value:
        return CheckOutcome(holds=True, detail="guard condition not triggered")
    required = after.entities.get(check.require_entity, {}).get(check.require_id)
    actual = None if required is None else required.get(check.require_field)
    if actual != check.require_value:
        return CheckOutcome(
            holds=False,
            detail=f"{check.entity}[{check.entity_id}].{check.when_field}={check.when_value!r} "
            f"but {check.require_entity}[{check.require_id}].{check.require_field}={actual!r} "
            f"(required {check.require_value!r})",
        )
    return CheckOutcome(holds=True, detail="precondition satisfied")


def evaluate_invariant(
    invariant: InvariantAssertion, exchanges: StepExchanges, after: DbSnapshot
) -> list[CheckOutcome]:
    return [evaluate_check(check, exchanges, after) for check in invariant.checks]
