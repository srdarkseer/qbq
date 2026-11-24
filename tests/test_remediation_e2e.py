import importlib.util
import json
from pathlib import Path

import pytest

from cubic.cli import main
from cubic.gym import load_environment
from cubic.remediate import FixVerdict, RulePatcher, remediate

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "benchmarks/results/remediation_bench.json"


@pytest.mark.parametrize(("env_name", "rules"), [
    ("fintech-compiled", {"inv.tenant.read_wallet": "bypassed_guard", "inv.ledger.conservation": "locked_branch",
                          "inv.lifecycle.disburse_loan": "bypassed_guard",
                          "inv.input.transfer.amount": "schema_lower_bound"}),
    ("saas-compiled", {"inv.tenant.read_workspace": "bypassed_guard", "inv.ledger.conservation": "locked_branch",
                       "inv.lifecycle.upgrade_subscription": "bypassed_guard",
                       "inv.input.buy_seats.quantity": "schema_lower_bound"}),
])
def test_closed_loop_remediates_every_planted_bug(env_name, rules, tmp_path):
    env = load_environment(env_name)
    report = remediate(env, variant="vulnerable", runs=3, out_dir=tmp_path)
    assert {f.invariant_id: f.verified_patch.rule for f in report.findings} == rules
    assert report.patch_pass_rate == 1.0
    for f in report.findings:
        final = f.attempts[-1]
        assert final.verdict is FixVerdict.VERIFIED and final.exploit_violations == 0
        assert final.functional and not final.regressions
        assert f.regression_test.fails_on_unpatched and f.regression_test.passes_on_patched
        assert (tmp_path / f.regression_test.file).is_file()
    assert set(report.timings_s) == {"compile", "search", "verify", "patch_and_verify", "export", "total"}


def test_patched_build_has_no_findings():
    report = remediate(load_environment("saas-compiled"), variant="patched", runs=1)
    assert report.findings == [] and report.patch_pass_rate is None


def test_unknown_variant_and_missing_app_source():
    with pytest.raises(KeyError, match="unknown variant"):
        remediate(load_environment("fintech-compiled"), variant="nope")
    with pytest.raises(ValueError, match="no app_source"):
        remediate(load_environment("fintech"))


def test_cli_remediate_writes_tests_patches_and_report(tmp_path, capsys):
    out, patches, report = tmp_path / "tests", tmp_path / "patches", tmp_path / "report.json"
    code = main(["remediate", "--env", "saas", "--out-dir", str(out), "--patch-dir", str(patches),
                 "--json", str(report), "--require-verified"])
    assert code == 0
    text = capsys.readouterr().out
    assert "environment=saas-compiled" in text and "4/4 findings have a verified patch (100%)" in text
    assert len(list(out.glob("test_regression_*.py"))) == 4
    assert len(list(patches.glob("*.patch"))) == 4
    data = json.loads(report.read_text())
    assert all(f["verified_patch"]["diff"].startswith("--- a/cubic/gym/saas/app.py") for f in data["findings"])
    assert "patched_source" not in data["findings"][0]["verified_patch"]


def test_cli_remediate_gate_fails_without_verified_patch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(RulePatcher, "propose", lambda self, finding, ctx: [])
    code = main(["remediate", "--env", "fintech", "--runs", "1", "--out-dir", str(tmp_path),
                 "--require-verified"])
    assert code == 1
    captured = capsys.readouterr()
    assert "0/4 findings have a verified patch" in captured.out and "GATE FAILED" in captured.err


# --------------------------------------------------------------------------- benchmark


def _bench():
    spec = importlib.util.spec_from_file_location("remediation_bench", ROOT / "benchmarks/remediation_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remediation_benchmark_small_run():
    result = _bench().run(runs=1, apps_to_run=("saas",))
    saas = result["apps"]["saas"]
    assert saas["detection"]["grounded"]["recall_at_1"] == 1.0
    assert saas["detection"]["grounded"]["false_positive_rate"] == 0.0
    assert saas["detection"]["ungrounded_m3_policy"]["recall_at_1"] < 1.0
    assert saas["remediation"]["totals"] == {"findings": 8, "verified_patches": 8,
                                             "regression_tests_valid": 8, "patch_pass_rate": 1.0}
    assert set(result["latency_s"]["remediate_vulnerable"]["saas"]) >= {"compile", "total"}


def test_committed_remediation_snapshot_is_consistent():
    data = json.loads(SNAPSHOT.read_text())
    assert data["benchmark"] == "remediation" and data["runs_per_candidate"] == 3
    summary = data["summary"]
    assert summary["recall_at_1"] == {"fintech": 1.0, "saas": 1.0, "agent_gateway": 1.0}
    assert summary["false_positive_rate"] == {"fintech": 0.0, "saas": 0.0, "agent_gateway": 0.0}
    assert summary["findings_remediated"] == summary["verified_patches"] == 24
    assert summary["patch_pass_rate"] == 1.0
    for app in data["apps"].values():
        assert app["remediation"]["totals"]["regression_tests_valid"] == 8
        for variant in app["remediation"]["by_variant"].values():
            assert all(fix["exploit_violations_after"] == 0 and fix["regressions"] == []
                       for fix in variant["fixes"].values())
