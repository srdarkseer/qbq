from cubic.compiler import compile_app, synthesize_invariants
from cubic.compiler.static import LedgerSpec
from cubic.graph import InvariantCategory, LedgerConservationCheck, StateGraph
from cubic.gym.fintech import create_app

GRAPH = compile_app(lambda: create_app()).graph
IDS = {inv.id for inv in GRAPH.invariants}


def test_authorization_and_tenant_isolation_rules():
    auth = {i for i in IDS if i.startswith("inv.auth.")}
    assert auth == {f"inv.auth.{n}" for n in
                    ("apply_loan", "approve_loan", "disburse_loan", "read_wallet", "transfer", "withdraw")}
    tenant = {i for i in IDS if i.startswith("inv.tenant.")}
    assert tenant == {"inv.tenant.read_wallet", "inv.tenant.withdraw",
                      "inv.tenant.approve_loan", "inv.tenant.disburse_loan"}
    assert "role guard ['ADMIN']" in GRAPH.invariant("inv.tenant.approve_loan").description
    assert GRAPH.invariant("inv.tenant.read_wallet").checks[0].allowed == {403, 404}


def test_input_and_ledger_rules():
    inputs = {i for i in IDS if i.startswith("inv.input.")}
    assert inputs == {"inv.input.transfer.amount", "inv.input.withdraw.amount", "inv.input.apply_loan.amount"}
    assert "NO lower bound" in GRAPH.invariant("inv.input.transfer.amount").description
    assert "enforces a lower bound" in GRAPH.invariant("inv.input.withdraw.amount").description

    ledger = GRAPH.invariant("inv.ledger.conservation")
    assert ledger.category is InvariantCategory.FINANCIAL
    assert set(ledger.edge_ids) == {"edge.withdraw", "edge.transfer", "edge.disburse_loan"}
    check = ledger.checks[0]
    assert isinstance(check, LedgerConservationCheck)
    assert (check.entity, check.balance_field, check.account_field) == ("wallets", "balance", "wallet_id")


def test_lifecycle_and_functional_rules():
    lifecycle = {i for i in IDS if i.startswith("inv.lifecycle.")}
    assert lifecycle == {"inv.lifecycle.approve_loan", "inv.lifecycle.disburse_loan"}
    assert "not APPROVED" in GRAPH.invariant("inv.lifecycle.disburse_loan").description
    assert GRAPH.invariant("inv.functional.apply_loan").checks[0].allowed == {201}
    assert {i for i in IDS if i.startswith("inv.functional.")} == {
        f"inv.functional.{e.id.removeprefix('edge.')}" for e in GRAPH.edges
    }


def test_invariants_are_sorted_and_reference_real_edges():
    assert [i.id for i in GRAPH.invariants] == sorted(IDS)
    edge_ids = {e.id for e in GRAPH.edges}
    assert all(set(i.edge_ids) <= edge_ids for i in GRAPH.invariants)


def test_ledger_rule_requires_a_complete_spec():
    bare = StateGraph.model_validate({**GRAPH.model_dump(), "invariants": ()})
    assert "inv.ledger.conservation" not in {i.id for i in synthesize_invariants(bare)}
    partial = LedgerSpec(collection="ledger", account_field="wallet_id", amount_field="amount")
    assert "inv.ledger.conservation" not in {i.id for i in synthesize_invariants(bare, partial)}


def test_synthesizer_works_on_hand_written_graphs():
    from cubic.gym.fintech.graph import build_graph

    hand = build_graph()
    invs = {i.id for i in synthesize_invariants(hand)}
    assert "inv.tenant.wallet_read" in invs and "inv.input.transfer.amount" in invs
    assert not any(i.startswith("inv.lifecycle.") for i in invs)  # no guard metadata to go on
