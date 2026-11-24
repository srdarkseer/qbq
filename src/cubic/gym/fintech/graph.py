"""Hand-compiled Behavioral State Graph for the fintech testbed.

In later milestones this graph is produced automatically by the static + dynamic
ingestion pipeline; here it serves as the reference ground truth.
"""

from __future__ import annotations

from cubic.graph import (
    ActorPersona,
    HttpMethod,
    InvariantAssertion,
    InvariantCategory,
    LedgerConservationCheck,
    NodeKind,
    Severity,
    StateGraph,
    StateGuardCheck,
    StateNode,
    StatusCodeCheck,
    TransitionEdge,
)

_AMOUNT_SCHEMA = {
    "type": "object",
    "required": ["amount"],
    "properties": {"amount": {"type": "integer", "exclusiveMinimum": 0}},
}

_TRANSFER_SCHEMA = {
    "type": "object",
    "required": ["from_wallet_id", "to_wallet_id", "amount"],
    "properties": {
        "from_wallet_id": {"type": "string"},
        "to_wallet_id": {"type": "string"},
        "amount": {"type": "integer", "exclusiveMinimum": 0},
    },
}


def _entity(id: str, entity: str, state: str) -> StateNode:
    return StateNode(id=id, kind=NodeKind.ENTITY_STATE, label=f"{entity.title()}: {state}",
                     entity=entity, state=state)


def build_graph() -> StateGraph:
    nodes = (
        StateNode(id="api.health", kind=NodeKind.API_ENDPOINT, label="Health"),
        StateNode(id="api.wallets", kind=NodeKind.API_ENDPOINT, label="Wallets API"),
        StateNode(id="api.loans", kind=NodeKind.API_ENDPOINT, label="Loans API"),
        _entity("wallet.active", "wallet", "ACTIVE"),
        _entity("transfer.completed", "transfer", "COMPLETED"),
        _entity("loan.applied", "loan", "APPLIED"),
        _entity("loan.approved", "loan", "APPROVED"),
        _entity("loan.disbursed", "loan", "DISBURSED"),
    )
    users = frozenset({"USER"})
    edges = (
        TransitionEdge(id="edge.health", source="api.health", target="api.health",
                       method=HttpMethod.GET, path="/health", auth_required=False),
        TransitionEdge(id="edge.wallet_read", source="api.wallets", target="wallet.active",
                       method=HttpMethod.GET, path="/wallets/{wallet_id}",
                       allowed_roles=frozenset({"USER", "ADMIN"}),
                       required_headers=frozenset({"Authorization"})),
        TransitionEdge(id="edge.withdraw", source="wallet.active", target="wallet.active",
                       method=HttpMethod.POST, path="/wallets/{wallet_id}/withdraw",
                       allowed_roles=users, required_headers=frozenset({"Authorization"}),
                       payload_schema=_AMOUNT_SCHEMA),
        TransitionEdge(id="edge.transfer", source="wallet.active", target="transfer.completed",
                       method=HttpMethod.POST, path="/transfers", allowed_roles=users,
                       required_headers=frozenset({"Authorization"}),
                       payload_schema=_TRANSFER_SCHEMA),
        TransitionEdge(id="edge.loan_apply", source="api.loans", target="loan.applied",
                       method=HttpMethod.POST, path="/loans", allowed_roles=users,
                       required_headers=frozenset({"Authorization"}),
                       payload_schema=_AMOUNT_SCHEMA),
        TransitionEdge(id="edge.loan_approve", source="loan.applied", target="loan.approved",
                       method=HttpMethod.POST, path="/loans/{loan_id}/approve",
                       allowed_roles=frozenset({"ADMIN"}),
                       required_headers=frozenset({"Authorization"})),
        TransitionEdge(id="edge.loan_disburse", source="loan.approved", target="loan.disbursed",
                       method=HttpMethod.POST, path="/loans/{loan_id}/disburse",
                       allowed_roles=users, required_headers=frozenset({"Authorization"})),
    )
    actors = (
        ActorPersona(id="anonymous", display_name="Unauthenticated", role="ANONYMOUS"),
        ActorPersona(id="user_a", display_name="Alice (User A)", role="USER", tenant_id="u_alice",
                     auth_token="tok_alice", owned_resources={"wallets": frozenset({"w_alice"})}),
        ActorPersona(id="user_b", display_name="Bob (User B)", role="USER", tenant_id="u_bob",
                     auth_token="tok_bob", owned_resources={"wallets": frozenset({"w_bob"})}),
        ActorPersona(id="user_c", display_name="Carol (KYC pending)", role="USER", tenant_id="u_carol",
                     auth_token="tok_carol",
                     owned_resources={"wallets": frozenset({"w_carol"}),
                                      "loans": frozenset({"loan_carol"})}),
        ActorPersona(id="merchant_admin", display_name="Merchant Admin", role="ADMIN",
                     tenant_id="u_admin", auth_token="tok_admin"),
    )
    invariants = (
        InvariantAssertion(
            id="inv.wallet_read_isolation", category=InvariantCategory.SECURITY,
            severity=Severity.HIGH, edge_ids=("edge.wallet_read",),
            description="A user reading another user's wallet must get 403/404.",
            checks=(StatusCodeCheck(step=0, allowed=frozenset({403, 404})),),
        ),
        InvariantAssertion(
            id="inv.wallet_auth_required", category=InvariantCategory.SECURITY,
            severity=Severity.HIGH, edge_ids=("edge.wallet_read",),
            description="Unauthenticated wallet reads must get 401.",
            checks=(StatusCodeCheck(step=0, allowed=frozenset({401})),),
        ),
        InvariantAssertion(
            id="inv.withdraw_ownership", category=InvariantCategory.SECURITY,
            severity=Severity.CRITICAL, edge_ids=("edge.withdraw",),
            description="A user withdrawing from another user's wallet must get 403/404.",
            checks=(StatusCodeCheck(step=0, allowed=frozenset({403, 404})),
                    LedgerConservationCheck()),
        ),
        InvariantAssertion(
            id="inv.ledger_conservation", category=InvariantCategory.FINANCIAL,
            severity=Severity.CRITICAL, edge_ids=("edge.withdraw", "edge.transfer"),
            description="Every wallet balance equals sum(credits) - sum(debits) and never goes negative.",
            checks=(LedgerConservationCheck(),),
        ),
        InvariantAssertion(
            id="inv.transfer_amount_positive", category=InvariantCategory.FINANCIAL,
            severity=Severity.CRITICAL, edge_ids=("edge.transfer",),
            description="Transfers with non-positive amounts must be rejected (400/422).",
            checks=(StatusCodeCheck(step=0, allowed=frozenset({400, 422})),),
        ),
        InvariantAssertion(
            id="inv.loan_requires_kyc", category=InvariantCategory.SECURITY,
            severity=Severity.CRITICAL, edge_ids=("edge.loan_approve", "edge.loan_disburse"),
            description="A loan may only be DISBURSED if the borrower's KYC is APPROVED.",
            checks=(StateGuardCheck(
                entity="loans", entity_id="loan_carol", when_field="status", when_value="DISBURSED",
                require_entity="users", require_id="u_carol", require_field="kyc_status",
                require_value="APPROVED",
            ),),
        ),
        InvariantAssertion(
            id="inv.transfer_happy_path", category=InvariantCategory.FUNCTIONAL,
            severity=Severity.MEDIUM, edge_ids=("edge.transfer",),
            description="A valid transfer between users completes (200) and the ledger reconciles.",
            checks=(StatusCodeCheck(step=0, allowed=frozenset({200})), LedgerConservationCheck()),
        ),
        InvariantAssertion(
            id="inv.loan_happy_path", category=InvariantCategory.FUNCTIONAL,
            severity=Severity.MEDIUM, edge_ids=("edge.loan_apply", "edge.loan_approve"),
            description="An eligible borrower can apply (201) and an admin can approve (200).",
            checks=(StatusCodeCheck(step=0, allowed=frozenset({201})),
                    StatusCodeCheck(step=1, allowed=frozenset({200}))),
        ),
    )
    return StateGraph(app_name="cubic-fintech-testbed", nodes=nodes, edges=edges,
                      actors=actors, invariants=invariants)
