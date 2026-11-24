"""The fintech testbed, driven by its *auto-compiled* graph and synthesized invariants.

This environment proves that Milestone 2's compiler output can drive the
Milestone 1 deterministic verifier. Ground-truth bugs map to synthesized invariant
ids, and the baseline probes are re-targeted onto compiled edges by
``(method, path)``.
"""

from __future__ import annotations

from functools import cache

from cubic.compiler import compile_app
from cubic.graph import StateGraph
from cubic.gym.base import AppSource, Environment, GroundTruthBug
from cubic.gym.fintech import app as testbed
from cubic.gym.fintech.graph import build_graph
from cubic.gym.fintech.manifest import BUGS, baseline_candidates
from cubic.verifier.trace import ActionStep, ActionTrace, CandidateFinding

BUG_TO_SYNTHESIZED_INVARIANT = {
    "idor_wallet_read": "inv.tenant.read_wallet",
    "race_withdraw": "inv.ledger.conservation",
    "loan_state_skip": "inv.lifecycle.disburse_loan",
    "negative_transfer": "inv.input.transfer.amount",
}

PROBE_TO_SYNTHESIZED_INVARIANT = {
    "probe.cross_account_wallet_read": "inv.tenant.read_wallet",
    "probe.withdraw_concurrency_spike": "inv.ledger.conservation",
    "probe.loan_skip_approval": "inv.lifecycle.disburse_loan",
    "probe.negative_transfer": "inv.input.transfer.amount",
    "probe.anonymous_wallet_read": "inv.auth.read_wallet",
    "probe.cross_account_withdraw": "inv.tenant.withdraw",
    "probe.sequential_withdrawals": "inv.ledger.conservation",
    "qa.transfer_happy_path": "inv.functional.transfer",
    "qa.loan_happy_path": "inv.functional.apply_loan",
}


@cache
def compiled_graph() -> StateGraph:
    return compile_app(lambda: testbed.create_app(), personas=build_graph().actors).graph


def _retarget(candidate: CandidateFinding, reference: StateGraph, compiled: StateGraph) -> CandidateFinding:
    by_route = {(e.method, e.path): e.id for e in compiled.edges}
    steps = tuple(
        step.model_copy(update={"edge_id": by_route[(reference.edge(step.edge_id).method,
                                                     reference.edge(step.edge_id).path)]})
        for step in candidate.trace.steps
    )
    return candidate.model_copy(update={
        "invariant_id": PROBE_TO_SYNTHESIZED_INVARIANT[candidate.id],
        "trace": ActionTrace(steps=steps),
    })


def _extra_controls() -> tuple[CandidateFinding, ...]:
    """Probes for synthesized invariants that hold in every build (FPR controls)."""

    def one(cid: str, inv: str, edge: str, actor: str, hypothesis: str, **step) -> CandidateFinding:
        return CandidateFinding(id=cid, invariant_id=inv, hypothesis=hypothesis,
                                trace=ActionTrace(steps=(ActionStep(edge_id=edge, actor_id=actor, **step),)))

    return (
        one("probe.anonymous_withdraw", "inv.auth.withdraw", "edge.withdraw", "anonymous",
            "Withdraw without credentials.", path_params={"wallet_id": "w_alice"}, body={"amount": 1}),
        one("probe.cross_account_disburse", "inv.tenant.disburse_loan", "edge.disburse_loan", "user_b",
            "Disburse another borrower's loan.", path_params={"loan_id": "loan_carol"}),
        one("probe.user_approves_loan", "inv.tenant.approve_loan", "edge.approve_loan", "user_b",
            "A non-admin approves someone else's loan.", path_params={"loan_id": "loan_carol"}),
        one("probe.zero_withdraw", "inv.input.withdraw.amount", "edge.withdraw", "user_a",
            "Withdraw a zero amount.", path_params={"wallet_id": "w_alice"}, body={"amount": 0}),
        one("probe.negative_loan", "inv.input.apply_loan.amount", "edge.apply_loan", "user_a",
            "Apply for a negative loan.", body={"amount": -1}),
    )


def candidates() -> tuple[CandidateFinding, ...]:
    reference, compiled = build_graph(), compiled_graph()
    return tuple(_retarget(c, reference, compiled) for c in baseline_candidates()) + _extra_controls()


def environment() -> Environment:
    bugs = tuple(
        GroundTruthBug(b.id, BUG_TO_SYNTHESIZED_INVARIANT[b.id], b.title, b.cwe) for b in BUGS
    )
    return Environment(
        name="fintech-compiled",
        graph=compiled_graph(),
        bugs=bugs,
        build_app=lambda active: testbed.create_app(active),
        snapshot=testbed.snapshot,
        baseline_candidates=candidates,
        app_source=AppSource(module="cubic.gym.fintech.app"),
    )
