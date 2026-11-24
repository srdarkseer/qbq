"""Ground truth and the scripted baseline prober for the fintech testbed.

The baseline candidates stand in for the agent swarm in Milestone 1: each is a
structured graph mutation (cross-account replay, concurrency spike, state-machine
skip, boundary value) or a functional happy path. Several target invariants that
hold in *every* build, so they measure the false-positive rate.
"""

from __future__ import annotations

from qbq.gym.base import Environment, GroundTruthBug
from qbq.gym.fintech import app as testbed
from qbq.gym.fintech.graph import build_graph
from qbq.verifier.trace import ActionStep, ActionTrace, CandidateFinding

BUGS: tuple[GroundTruthBug, ...] = (
    GroundTruthBug("idor_wallet_read", "inv.wallet_read_isolation",
                   "Wallet read lacks ownership check (IDOR)", "CWE-639"),
    GroundTruthBug("race_withdraw", "inv.ledger_conservation",
                   "Withdraw TOCTOU allows double-spend", "CWE-362"),
    GroundTruthBug("loan_state_skip", "inv.loan_requires_kyc",
                   "Loan disbursal skips the approval state", "CWE-841"),
    GroundTruthBug("negative_transfer", "inv.transfer_amount_positive",
                   "Regression: negative transfer amounts accepted", "CWE-20"),
)


def _trace(*steps: ActionStep) -> ActionTrace:
    return ActionTrace(steps=steps)


def baseline_candidates() -> tuple[CandidateFinding, ...]:
    return (
        CandidateFinding(
            id="probe.cross_account_wallet_read", invariant_id="inv.wallet_read_isolation",
            hypothesis="Replay User A's wallet read with User B's token.",
            trace=_trace(ActionStep(edge_id="edge.wallet_read", actor_id="user_b",
                                    path_params={"wallet_id": "w_alice"})),
        ),
        CandidateFinding(
            id="probe.withdraw_concurrency_spike", invariant_id="inv.ledger_conservation",
            hypothesis="Fire 10 barrier-synchronized withdrawals of 6000 from a 10000 balance.",
            trace=_trace(ActionStep(edge_id="edge.withdraw", actor_id="user_a",
                                    path_params={"wallet_id": "w_alice"},
                                    body={"amount": 6_000}, concurrency=10)),
        ),
        CandidateFinding(
            id="probe.loan_skip_approval", invariant_id="inv.loan_requires_kyc",
            hypothesis="Jump loan.applied -> loan.disbursed, bypassing edge.loan_approve.",
            trace=_trace(ActionStep(edge_id="edge.loan_disburse", actor_id="user_c",
                                    path_params={"loan_id": "loan_carol"})),
        ),
        CandidateFinding(
            id="probe.negative_transfer", invariant_id="inv.transfer_amount_positive",
            hypothesis="Transfer a negative amount to pull funds from User B.",
            trace=_trace(ActionStep(edge_id="edge.transfer", actor_id="user_a",
                                    body={"from_wallet_id": "w_alice", "to_wallet_id": "w_bob",
                                          "amount": -5_000})),
        ),
        # --- hypotheses that are wrong in every build (false-positive controls) ---
        CandidateFinding(
            id="probe.anonymous_wallet_read", invariant_id="inv.wallet_auth_required",
            hypothesis="Read a wallet without credentials.",
            trace=_trace(ActionStep(edge_id="edge.wallet_read", actor_id="anonymous",
                                    path_params={"wallet_id": "w_alice"})),
        ),
        CandidateFinding(
            id="probe.cross_account_withdraw", invariant_id="inv.withdraw_ownership",
            hypothesis="Replay User A's withdraw with User B's token, keeping A's wallet_id.",
            trace=_trace(ActionStep(edge_id="edge.withdraw", actor_id="user_b",
                                    path_params={"wallet_id": "w_alice"},
                                    body={"amount": 1_000})),
        ),
        CandidateFinding(
            id="probe.sequential_withdrawals", invariant_id="inv.ledger_conservation",
            hypothesis="Sequential withdrawals that overdraw should corrupt the ledger.",
            trace=_trace(*(ActionStep(edge_id="edge.withdraw", actor_id="user_a",
                                      path_params={"wallet_id": "w_alice"},
                                      body={"amount": 6_000}) for _ in range(2))),
        ),
        # --- functional QA happy paths ---
        CandidateFinding(
            id="qa.transfer_happy_path", invariant_id="inv.transfer_happy_path",
            hypothesis="A valid transfer fails or desyncs the ledger.",
            trace=_trace(ActionStep(edge_id="edge.transfer", actor_id="user_a",
                                    body={"from_wallet_id": "w_alice", "to_wallet_id": "w_bob",
                                          "amount": 2_500})),
        ),
        CandidateFinding(
            id="qa.loan_happy_path", invariant_id="inv.loan_happy_path",
            hypothesis="An eligible borrower cannot complete apply -> approve.",
            trace=_trace(
                ActionStep(edge_id="edge.loan_apply", actor_id="user_a", body={"amount": 50_000}),
                ActionStep(edge_id="edge.loan_approve", actor_id="merchant_admin",
                           path_params={"loan_id": "loan_0001"}),
            ),
        ),
    )


def environment() -> Environment:
    return Environment(
        name="fintech",
        graph=build_graph(),
        bugs=BUGS,
        build_app=lambda bugs: testbed.create_app(bugs),
        snapshot=testbed.snapshot,
        baseline_candidates=baseline_candidates,
    )
