"""Real-world validation on examples/storefront: a multi-file FastAPI project with an unscripted bug.

Every LLM call goes through ``httpx.MockTransport``. No API key or network is needed.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

from qbq.cli import main
from qbq.compiler.adapter import external_environment, load_personas
from qbq.llm import GeminiProvider, ProviderPatchModel, graph_secrets
from qbq.llm import providers as providers_mod
from qbq.remediate import FixVerdict, LLMPatcher, RulePatcher, remediate
from qbq.remediate.patcher import unified_diff
from qbq.search import search
from qbq.verifier import DeterministicVerifier, Verdict

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
ORDERS = EXAMPLES / "storefront/routers/orders.py"
KEY = "AIzaSy-storefront-test-key-42"

BUGGY_INVOICE = '''async def read_invoice(order_id: str, user: CurrentUser, store: StoreDep) -> dict[str, Any]:
    order = store.orders.get(order_id)
    if order is None:
        raise HTTPException(404, "order not found")
'''
FIXED_INVOICE = '''async def read_invoice(order: OwnedOrder, store: StoreDep) -> dict[str, Any]:
'''


@pytest.fixture(autouse=True)
def storefront_importable(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))


@pytest.fixture(scope="module")
def env():
    sys.path.insert(0, str(EXAMPLES))
    try:
        yield external_environment("storefront.main:create_app",
                                   load_personas(EXAMPLES / "storefront/personas.json"),
                                   snapshot="storefront.main:snapshot")
    finally:
        sys.path.remove(str(EXAMPLES))


def _diff(old: str, new: str) -> str:
    source = ORDERS.read_text()
    assert old in source
    return unified_diff("storefront/routers/orders.py", source, source.replace(old, new))


# --------------------------------------------------------------------------- compile


def test_compile_resolves_prefixes_depends_chains_and_imported_schemas(env):
    edges = {e.id: e for e in env.graph.edges}
    assert {e.path for e in edges.values()} == {
        "/health", "/api/v1/products/{product_id}", "/api/v1/orders", "/api/v1/orders/{order_id}",
        "/api/v1/orders/{order_id}/invoice", "/api/v1/orders/{order_id}/cancel",
        "/api/v1/admin/orders/{order_id}/refund",
    }
    guards = {k: e.metadata["guards"] for k, e in edges.items()}
    assert guards["edge.read_order"]["ownership"] and guards["edge.cancel_order"]["ownership"]  # via deps.py
    assert guards["edge.read_invoice"]["ownership"] is False  # the bug: no ownership dependency
    assert edges["edge.refund_order"].allowed_roles == {"ADMIN"}  # router-level require_role("ADMIN")
    assert guards["edge.cancel_order"]["state_requires"] == "PLACED"
    assert all(e.auth_required for k, e in edges.items() if k not in {"edge.health", "edge.read_product"})
    assert all("static" in e.metadata["evidence"] for e in edges.values())
    props = edges["edge.create_order"].payload_schema["properties"]
    assert props["quantity"]["exclusiveMinimum"] == 0 and props["quantity"]["maximum"] == 100
    assert env.graph.actor("ann").observed("orders", "o_ann")["status"] == "PLACED"


def test_cli_compile_reads_every_project_module(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(EXAMPLES)
    report = tmp_path / "report.json"
    assert main(["compile", "--app", "storefront.main:create_app", "--personas", "storefront/personas.json",
                 "--out", str(tmp_path / "g.json"), "--report", str(report)]) == 0
    data = json.loads(report.read_text())
    assert data["route_coverage"] == 1.0 and data["static_unmatched_routes"] == []
    assert {"storefront/deps.py", "storefront/schemas.py", "storefront/models.py",
            "storefront/routers/orders.py", "storefront/main.py"} <= set(data["source_files"])
    assert "7/7 routes" in capsys.readouterr().out


# --------------------------------------------------------------------------- detection


def test_search_finds_only_the_unscripted_idor(env):
    plan = search(env.graph)
    verifier = DeterministicVerifier(env.graph, lambda: env.build_app(frozenset()), env.snapshot, runs=3)
    confirmed = {c.invariant_id for c in plan.candidates() if verifier.verify(c).verdict is Verdict.CONFIRMED}
    assert confirmed == {"inv.tenant.read_invoice"}
    assert plan.coverage["inv.tenant.read_order"].tests  # the guarded sibling is tested too, and holds


# --------------------------------------------------------------------------- remediation


class _Model:
    def __init__(self, diff):
        self.diff = diff
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return f"```diff\n{self.diff}```"


def test_rule_patcher_has_no_answer_for_a_missing_dependency(env):
    report = remediate(env, runs=1)
    (finding,) = report.findings
    assert finding.patches == [] and not finding.verified


def test_mocked_gemini_patch_passes_gate_across_modules(env, tmp_path):
    rec = []

    def handler(request):
        rec.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content":
                              f"Reuse the existing dependency:\n```diff\n{_diff(BUGGY_INVOICE, FIXED_INVOICE)}```"}}]})

    provider = GeminiProvider(api_key=KEY, transport=httpx.MockTransport(handler))
    patcher = LLMPatcher(ProviderPatchModel(provider, secrets=graph_secrets(env.graph)))
    before = ORDERS.read_text()
    orders_module = sys.modules.get("storefront.routers.orders")
    report = remediate(env, runs=3, patchers=[RulePatcher(), patcher], out_dir=tmp_path)

    (finding,) = report.findings
    assert finding.verified and finding.verified_patch.patcher == "llm"
    final = finding.attempts[-1]
    assert final.verdict is FixVerdict.VERIFIED and final.exploit_violations == 0 and final.runs == 3
    assert final.functional and not final.regressions
    assert finding.regression_test.fails_on_unpatched and finding.regression_test.passes_on_patched
    # the prompt carried cross-module context but no credentials
    sent = rec[0].content.decode()
    assert "get_owned_order" in sent and "storefront/deps.py" in sent
    assert "tok_" not in sent and KEY not in sent
    # isolation: originals untouched and restored
    assert ORDERS.read_text() == before
    assert sys.modules.get("storefront.routers.orders") is orders_module
    exported = (tmp_path / "test_regression_inv_tenant_read_invoice.py").read_text()
    assert "from storefront.main import create_app" in exported and KEY not in exported


@pytest.mark.parametrize(("bad", "verdict"), [
    # drops authentication instead of checking ownership: the IDOR remains
    ("async def read_invoice(order_id: str, store: StoreDep) -> dict[str, Any]:\n", FixVerdict.EXPLOIT_REPRODUCES),
    # "fixes" the leak by refusing everyone: the owner's own invoice regresses
    (FIXED_INVOICE.replace("-> dict[str, Any]:\n",
                           '-> dict[str, Any]:\n    raise HTTPException(404, "order not found")\n'),
     FixVerdict.FUNCTIONAL_REGRESSION),
])
def test_gate_rejects_plausible_but_wrong_model_patches(env, bad, verdict):
    old = BUGGY_INVOICE.splitlines(keepends=True)[0] if verdict is FixVerdict.EXPLOIT_REPRODUCES else BUGGY_INVOICE
    report = remediate(env, runs=3, patchers=[LLMPatcher(_Model(_diff(old, bad)))])
    (finding,) = report.findings
    assert not finding.verified and finding.attempts[-1].verdict is verdict


def test_cli_remediate_live_flags_with_mocked_transport(tmp_path, monkeypatch, capsys):
    diff = _diff(BUGGY_INVOICE, FIXED_INVOICE)

    def handler(request):
        body = json.loads(request.content)
        if body.get("response_format"):  # ranking request: keep the heuristic order
            user = json.loads(body["messages"][1]["content"])
            keys = [c["key"] for c in user["candidates"]]
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"ranking": [{"key": k, "score": 1 - i / len(keys)} for i, k in enumerate(keys)]})}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": f"```diff\n{diff}```"}}]})

    monkeypatch.setattr(providers_mod, "default_transport", httpx.MockTransport(handler))
    monkeypatch.setenv("QBQ_TEST_GEMINI_KEY", KEY)
    monkeypatch.chdir(EXAMPLES)
    report, patches = tmp_path / "report.json", tmp_path / "patches"
    code = main(["remediate", "--app", "storefront.main:create_app", "--personas", "storefront/personas.json",
                 "--snapshot", "storefront.main:snapshot", "--llm-provider", "gemini",
                 "--api-key-env", "QBQ_TEST_GEMINI_KEY", "--out-dir", str(tmp_path / "rt"),
                 "--patch-dir", str(patches), "--json", str(report), "--require-verified"])
    out = capsys.readouterr()
    assert code == 0, out.out + out.err
    assert "1/1 findings have a verified patch (100%)" in out.out
    assert "llm: gemini gemini-2.5-flash" in out.err
    (patch_file,) = patches.glob("*.patch")
    assert "+async def read_invoice(order: OwnedOrder, store: StoreDep)" in patch_file.read_text()
    for text in (out.out, out.err, report.read_text(), patch_file.read_text()):
        assert KEY not in text
