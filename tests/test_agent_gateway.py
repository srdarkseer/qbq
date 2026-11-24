import asyncio

import httpx
import pytest

from cubic.compiler.invariants import synthesize_invariants
from cubic.eval import run_eval
from cubic.graph import HttpMethod, NodeKind, StateGraph, StateNode, TransitionEdge
from cubic.gym import load_environment
from cubic.gym.agent_gateway import BUG_IDS, create_app
from cubic.gym.agent_gateway.environment import BUGS
from cubic.remediate import remediate
from cubic.search import CombinatorKind, Oracle, classify_invariant, generate_pool, search

ALPHA = {"Authorization": "Bearer tok_alpha_dev"}
BETA = {"Authorization": "Bearer tok_beta_dev"}


def _call(app, method, path, headers=None, json=None):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            return await c.request(method, path, headers=headers, json=json)
    return asyncio.run(go())


@pytest.mark.parametrize("bug", sorted(BUG_IDS))
def test_each_defect_toggles(bug):
    def exploit(app):
        if bug == "cross_tenant_conversation_read":
            return _call(app, "GET", "/conversations/conv_alpha", BETA).status_code == 200
        if bug == "cross_tenant_tool_invocation":
            return _call(app, "POST", "/agents/agent_beta/tool_calls", BETA,
                         {"tool_id": "tool_alpha_crm", "input": "dump"}).status_code == 200
        if bug == "unbounded_max_tokens":
            return _call(app, "POST", "/agents/agent_alpha/completions", ALPHA,
                         {"prompt": "hi", "max_tokens": 10**9}).status_code == 200
        return _call(app, "POST", "/actions/act_alpha/execute", ALPHA).status_code == 200

    assert not exploit(create_app())
    assert exploit(create_app({bug}))


def test_patched_gateway_happy_paths():
    app = create_app()
    assert _call(app, "POST", "/agents/agent_alpha/tool_calls", ALPHA,
                 {"tool_id": "tool_web_search", "input": "q"}).status_code == 200  # public tool
    assert _call(app, "POST", "/agents/agent_alpha/completions", ALPHA,
                 {"prompt": "hi", "max_tokens": 64}).status_code == 200
    assert _call(app, "POST", "/actions/act_alpha/approve", ALPHA).status_code == 200
    assert _call(app, "POST", "/actions/act_alpha/execute", ALPHA).status_code == 200


# --------------------------------------------------------------------------- AI invariant synthesis


def _graph(*edges):
    ids = sorted({n for e in edges for n in (e.source, e.target)})
    nodes = tuple(StateNode(id=n, kind=NodeKind.API_ENDPOINT, label=n) for n in ids)
    return StateGraph(app_name="t", nodes=nodes, edges=edges)


def test_ai_rules_detect_tool_and_generation_endpoints():
    tool = TransitionEdge(id="edge.call_tool", source="api.mcp", target="api.mcp", method=HttpMethod.POST,
                          path="/mcp/{session_id}/call", payload_schema={
                              "type": "object", "properties": {"tool_name": {"type": "string"},
                                                               "arguments": {"type": "object"}}})
    gen = TransitionEdge(id="edge.generate", source="api.llm", target="api.llm", method=HttpMethod.POST,
                         path="/llm/generate", payload_schema={"type": "object", "properties": {
                             "max_new_tokens": {"type": "integer"}, "top_k": {"type": "integer", "maximum": 50},
                             "temperature": {"type": "number"}}})
    read = TransitionEdge(id="edge.list_tools", source="api.mcp", target="api.mcp", method=HttpMethod.GET,
                          path="/tools/{tool_id}")
    invs = {i.id: i for i in synthesize_invariants(_graph(tool, gen, read))}
    assert "inv.tool_authz.call_tool" in invs and "inv.tool_authz.list_tools" not in invs  # reads excluded
    assert invs["inv.tool_authz.call_tool"].tags[0] == "OWASP-LLM06:2025"
    assert "NO upper bound" in invs["inv.ai_bounds.generate.max_new_tokens"].description
    assert "enforces an upper bound" in invs["inv.ai_bounds.generate.top_k"].description
    assert "inv.ai_bounds.generate.temperature" not in invs
    assert classify_invariant(invs["inv.ai_bounds.generate.max_new_tokens"]) is Oracle.REJECTED_INPUT
    assert all(i.tags for i in invs.values() if not i.id.startswith("inv.functional."))


def test_existing_gyms_have_no_ai_invariants_and_carry_owasp_tags():
    g = load_environment("fintech-compiled").graph
    assert not [i for i in g.invariants if i.id.startswith(("inv.tool_authz", "inv.ai_bounds"))]
    assert g.invariant("inv.tenant.read_wallet").tags == ("OWASP-API1:2023", "CWE-639")


# --------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def env():
    return load_environment("agent-gateway-autonomous")


def test_gateway_compiles_ai_invariants_and_grounds_token_balances(env):
    g = env.graph
    for bug in BUGS:
        g.invariant(bug.invariant_id)
    assert g.actor("alpha_dev").observed("tenants", "t_alpha")["token_balance"] == 50_000
    assert g.actor("beta_dev").observed("actions", "act_beta")["status"] == "APPROVED"


def test_confused_deputy_combinator_drives_tool_authz(env):
    plan = search(env.graph)
    first = plan.coverage["inv.tool_authz.create_tool_call"].tests[0]
    assert first.startswith("auto:cross_tenant_reference:inv.tool_authz.create_tool_call:create_tool_call:")
    pool = generate_pool(env.graph)
    swaps = [t for t in pool if t.combinator is CombinatorKind.CROSS_TENANT_REFERENCE]
    assert swaps and all(t.features["field"] in {"tool_id", "agent_id"} for t in swaps)
    assert plan.coverage["inv.ai_bounds.create_completion.max_tokens"].tests[0].endswith("max_tokens.huge")
    assert plan.coverage["inv.functional.create_completion"].tests  # funded via agent.tenant_id hop


def test_gateway_autonomous_eval_and_remediation(env):
    report = run_eval(env, runs=3)
    m = report.metrics
    assert m.recall_at_1 == 1.0 and m.bugs_found_at_1 == 8 and m.false_positives == 0
    fixes = remediate(load_environment("agent-gateway-compiled"), runs=3)
    assert fixes.patch_pass_rate == 1.0 and len(fixes.findings) == 4
    assert all(f.regression_test.valid for f in fixes.findings)
