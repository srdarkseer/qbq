import asyncio
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from qbq.gym import load_environment
from qbq.gym.fintech import app as fintech_app
from qbq.remediate import (
    ConfirmedFinding,
    FixVerdict,
    LLMPatcher,
    PatchCandidate,
    PatchError,
    Patcher,
    PatchPrompt,
    RulePatcher,
    apply_unified_diff,
    averify_fix,
    export_regression_test,
    load_context,
    patched_module,
    propose_patches,
    regression_file_name,
)
from qbq.search import Oracle, search
from qbq.verifier import DeterministicVerifier

FINTECH = load_environment("fintech-compiled")
SAAS = load_environment("saas-compiled")
VULNERABLE = frozenset(b.id for b in FINTECH.bugs)


def _findings(env, active, runs=1):
    plan = search(env.graph)

    async def go():
        verifier = DeterministicVerifier(env.graph, env.app_factory(active), env.snapshot, runs=runs)
        out = {}
        for candidate in plan.candidates():
            result = await verifier.averify(candidate)
            if result.confirmed:
                out[candidate.invariant_id] = ConfirmedFinding.from_verification(
                    env.graph, candidate, result, variant="vulnerable")
        return out

    return asyncio.run(go()), [t.to_candidate() for t in plan.tests if t.oracle is Oracle.SUCCESS]


@pytest.fixture(scope="module")
def fintech():
    return _findings(FINTECH, VULNERABLE)


def _verify(env, finding, patch, functional, active=VULNERABLE):
    return asyncio.run(averify_fix(
        finding, patch, graph=env.graph, app_source=env.app_source, active_bugs=active,
        original_factory=env.app_factory(active), original_snapshot=env.snapshot,
        functional_tests=functional, runs=3,
    ))


def _patch(finding, source, rule="manual"):
    return PatchCandidate(finding_id=finding.id, patcher="test", rule=rule, file=finding.source.file,
                          description="test", diff="", patched_source=source)


# --------------------------------------------------------------------------- findings & context


def test_confirmed_finding_carries_source_provenance(fintech):
    findings, _ = fintech
    assert set(findings) == {"inv.tenant.read_wallet", "inv.ledger.conservation",
                             "inv.lifecycle.disburse_loan", "inv.input.transfer.amount"}
    race = findings["inv.ledger.conservation"]
    assert race.oracle is Oracle.CONSERVATION and race.edge_id == "edge.withdraw"
    assert race.source.file == "qbq/gym/fintech/app.py" and race.source.function == "withdraw"
    assert race.reproduction.startswith("#!/usr/bin/env bash")


def test_unconfirmed_results_are_rejected():
    plan = search(FINTECH.graph)
    candidate = plan.candidates()[0]
    result = DeterministicVerifier(FINTECH.graph, FINTECH.app_factory(()), FINTECH.snapshot, runs=1).verify(candidate)
    with pytest.raises(ValueError, match="not confirmed"):
        ConfirmedFinding.from_verification(FINTECH.graph, candidate, result)


def test_context_resolves_handler_helpers_and_models(fintech):
    findings, _ = fintech
    ctx = load_context(findings["inv.input.transfer.amount"], FINTECH.app_source)
    assert ctx.handler.name == "transfer"
    assert "owned_wallet" in {h.name for h in ctx.helpers}
    assert [m.name for m in ctx.body_models] == ["TransferIn"]


# --------------------------------------------------------------------------- rule patcher


def test_rule_patcher_picks_rule_by_oracle(fintech):
    findings, _ = fintech
    rules = {inv: [p.rule for p in propose_patches(f, FINTECH.app_source)] for inv, f in findings.items()}
    assert rules == {
        "inv.tenant.read_wallet": ["bypassed_guard"],
        "inv.ledger.conservation": ["locked_branch"],
        "inv.lifecycle.disburse_loan": ["bypassed_guard"],
        "inv.input.transfer.amount": ["schema_lower_bound", "bypassed_guard"],
    }


def test_rule_patches_are_minimal_unified_diffs(fintech):
    findings, _ = fintech
    schema = propose_patches(findings["inv.input.transfer.amount"], FINTECH.app_source)[0]
    assert schema.diff.startswith("--- a/qbq/gym/fintech/app.py\n+++ b/qbq/gym/fintech/app.py\n")
    assert "+    amount: int = Field(gt=0)\n" in schema.diff
    (race,) = propose_patches(findings["inv.ledger.conservation"], FINTECH.app_source)
    assert '-        if "race_withdraw" in bugs:' in race.diff and "+        async with store.locks" in race.diff
    original = Path(fintech_app.__file__).read_text()
    for finding in findings.values():
        for patch in propose_patches(finding, FINTECH.app_source):
            assert apply_unified_diff(original, patch.diff) == patch.patched_source


def test_schema_rule_adds_missing_field_import():
    findings, _ = _findings(SAAS, frozenset({"negative_seat_quantity"}))
    schema = propose_patches(findings["inv.input.buy_seats.quantity"], SAAS.app_source)[0]
    assert "+from pydantic import BaseModel, Field\n" in schema.diff
    assert "+    quantity: int = Field(gt=0)\n" in schema.diff


def test_apply_unified_diff_tolerates_drift_and_rejects_mismatch():
    source = "a\nb\nc\nd\n"
    diff = "--- a/x\n+++ b/x\n@@ -2,2 +2,2 @@\n b\n-c\n+C\n"
    assert apply_unified_diff("z\n" + source, diff) == "z\na\nb\nC\nd\n"
    with pytest.raises(PatchError):
        apply_unified_diff("q\nr\n", diff)
    with pytest.raises(PatchError):
        apply_unified_diff(source, "not a diff")


# --------------------------------------------------------------------------- verification gate


def test_every_rule_patch_is_verified(fintech):
    findings, functional = fintech
    for finding in findings.values():
        patch = propose_patches(finding, FINTECH.app_source)[0]
        result = _verify(FINTECH, finding, patch, functional)
        assert result.verdict is FixVerdict.VERIFIED, (finding.invariant.id, result)
        assert result.exploit_violations == 0 and result.runs == 3
        assert result.functional and not result.regressions


def test_gate_rejects_patch_that_does_not_fix(fintech):
    findings, functional = fintech
    finding = findings["inv.tenant.read_wallet"]
    source = Path(fintech_app.__file__).read_text().replace('"""Reference fintech', '"""Patched fintech', 1)
    result = _verify(FINTECH, finding, _patch(finding, source), functional)
    assert result.verdict is FixVerdict.EXPLOIT_REPRODUCES and result.exploit_violations == 3


def test_gate_rejects_functional_regression(fintech):
    findings, functional = fintech
    finding = findings["inv.ledger.conservation"]
    source = Path(fintech_app.__file__).read_text().replace(
        "        wallet = owned_wallet(wallet_id, user)\n        ref = store.next_id(\"wd\")\n",
        "        wallet = owned_wallet(wallet_id, user)\n        raise HTTPException(503, \"withdrawals disabled\")\n",
        1,
    )
    result = _verify(FINTECH, finding, _patch(finding, source), functional)
    assert result.exploit_violations == 0  # "fixed" the race by breaking withdrawals...
    assert result.verdict is FixVerdict.FUNCTIONAL_REGRESSION  # ...so the gate rejects it
    assert any("functional.withdraw" in t for t in result.regressions)


def test_gate_rejects_unimportable_patch(fintech):
    findings, functional = fintech
    finding = findings["inv.tenant.read_wallet"]
    result = _verify(FINTECH, finding, _patch(finding, "def broken(:\n"), functional)
    assert result.verdict is FixVerdict.PATCH_INVALID and "SyntaxError" in result.error


def test_patched_module_is_isolated(fintech):
    findings, _ = fintech
    finding = findings["inv.tenant.read_wallet"]
    patch = propose_patches(finding, FINTECH.app_source)[0]
    before = Path(fintech_app.__file__).read_text()
    with patched_module(patch, FINTECH.app_source) as module:
        assert module is not fintech_app and module.__name__ in sys.modules
        name = module.__name__
    assert name not in sys.modules
    assert Path(fintech_app.__file__).read_text() == before
    # the original buggy build still reproduces after verification
    assert DeterministicVerifier(FINTECH.graph, FINTECH.app_factory(VULNERABLE), FINTECH.snapshot,
                                 runs=1).verify(finding.candidate).confirmed


# --------------------------------------------------------------------------- LLM patcher


class _DiffModel:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, prompt: PatchPrompt) -> str:
        return self.reply(prompt) if callable(self.reply) else self.reply


def test_llm_patcher_applies_model_diff_and_gate_verifies(fintech):
    findings, functional = fintech
    finding = findings["inv.lifecycle.disburse_loan"]
    rule_diff = propose_patches(finding, FINTECH.app_source)[0].diff
    patcher = LLMPatcher(_DiffModel(f"Here is the fix:\n```diff\n{rule_diff}```\n"))
    (candidate,) = propose_patches(finding, FINTECH.app_source, [patcher])
    assert candidate.patcher == "llm" and candidate.diff == rule_diff
    assert _verify(FINTECH, finding, candidate, functional).verified
    (prompt,) = patcher.prompts
    text = prompt.render()
    assert "tok_" not in text and "disburse_loan" in text and "loan_state_skip" in text
    assert prompt.finding["oracle"] == "rejected_transition"


def test_llm_patcher_records_unusable_replies(fintech):
    findings, _ = fintech
    finding = findings["inv.tenant.read_wallet"]
    patcher = LLMPatcher(_DiffModel("I would add an ownership check."))
    assert propose_patches(finding, FINTECH.app_source, [patcher]) == []
    assert len(patcher.errors) == 1
    assert isinstance(RulePatcher(), Patcher) and isinstance(patcher, Patcher)


# --------------------------------------------------------------------------- regression exporter


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exported_tests_pass_when_fixed_and_fail_when_bug_returns(fintech, tmp_path):
    findings, _ = fintech
    by_invariant = {f.invariant.id: f for f in findings.values()}
    bugs = {b.invariant_id: b.id for b in FINTECH.bugs}
    for inv_id, finding in by_invariant.items():
        path = export_regression_test(finding, FINTECH.graph, FINTECH.app_source, tmp_path)
        assert path.name == regression_file_name(inv_id)
        source = path.read_text()
        assert "import qbq" not in source.replace("from qbq.gym.fintech.app import", "")
        module = _load(path)
        test = getattr(module, path.stem)
        test(0)  # default factory = the fixed (patched) build
        module.APP_FACTORY_ARGS = (frozenset({bugs[inv_id]}),)
        with pytest.raises(AssertionError):
            test(0)


def test_exported_test_runs_under_pytest(fintech, tmp_path):
    findings, _ = fintech
    path = export_regression_test(findings["inv.ledger.conservation"], FINTECH.graph, FINTECH.app_source,
                                  tmp_path)
    assert "asyncio.Barrier" in path.read_text() and "_snapshot" in path.read_text()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "3 passed" in proc.stdout
