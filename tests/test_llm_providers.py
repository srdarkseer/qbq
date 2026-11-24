"""LLM provider adapter tests. Every HTTP call goes to an ``httpx.MockTransport``: no key, no network."""

import asyncio
import json

import httpx
import pytest

from qbq.cli import main
from qbq.gym import load_environment
from qbq.llm import (
    GeminiProvider,
    LLMError,
    OpenAICompatibleProvider,
    ProviderPatchModel,
    ProviderRankingModel,
    build_provider,
    graph_secrets,
    scrub,
)
from qbq.llm import providers as providers_mod
from qbq.remediate import ConfirmedFinding, LLMPatcher, averify_fix, propose_patches
from qbq.search import LLMGuidedPolicy, Oracle, search
from qbq.verifier import DeterministicVerifier

KEY = "AIzaSy-test-key-0123456789"  # fake; must never surface anywhere
FINTECH = load_environment("fintech-compiled")


def _reply(content, *, status=200, usage=None):
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if usage:
        body["usage"] = usage
    return httpx.Response(status, json=body)


class Recorder:
    """A MockTransport handler that records requests and replays scripted responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        nxt = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return nxt(request) if callable(nxt) else nxt

    @property
    def bodies(self):
        return [json.loads(r.content) for r in self.requests]


def _gemini(handler, **kw):
    return GeminiProvider(api_key=KEY, transport=httpx.MockTransport(handler), sleep=lambda s: None, **kw)


# --------------------------------------------------------------------------- wire format


def test_gemini_request_shape_and_json_mode():
    rec = Recorder(_reply('{"ok": true}', usage={"prompt_tokens": 12, "completion_tokens": 3}))
    provider = _gemini(rec)
    assert provider.chat([{"role": "user", "content": "hi"}], json_mode=True) == '{"ok": true}'
    (request,) = rec.requests
    assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = rec.bodies[0]
    assert body["model"] == "gemini-2.5-flash" and body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    assert (provider.usage.calls, provider.usage.prompt_tokens, provider.usage.completion_tokens) == (1, 12, 3)


def test_openai_compatible_without_json_mode_and_content_parts():
    rec = Recorder(_reply([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]))
    provider = OpenAICompatibleProvider(model="llama3.1", base_url="http://localhost:11434/v1",
                                        transport=httpx.MockTransport(rec))
    assert provider.chat([{"role": "user", "content": "x"}]) == "ab"
    assert str(rec.requests[0].url) == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in rec.requests[0].headers and "response_format" not in rec.bodies[0]


def test_transient_errors_are_retried_with_backoff():
    sleeps = []
    rec = Recorder(httpx.Response(429, text="slow down"), httpx.Response(503, text="busy"), _reply("done"))
    provider = GeminiProvider(api_key=KEY, transport=httpx.MockTransport(rec), sleep=sleeps.append)
    assert provider.chat([{"role": "user", "content": "x"}]) == "done"
    assert sleeps == [1.0, 2.0] and provider.usage.retries == 2


def test_transport_errors_are_retried_then_reported():
    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    provider = _gemini(boom, max_retries=1)
    with pytest.raises(LLMError, match="transport error: ConnectError"):
        provider.chat([{"role": "user", "content": "x"}])
    assert provider.usage.calls == 2 and provider.usage.failures == 1


def test_errors_never_leak_the_key():
    echo = Recorder(lambda r: httpx.Response(401, text=f"invalid key {KEY} for project"))
    provider = _gemini(echo)
    with pytest.raises(LLMError) as exc:
        provider.chat([{"role": "user", "content": "x"}])
    assert KEY not in str(exc.value) and "[REDACTED]" in str(exc.value)
    assert len(echo.requests) == 1  # 4xx (other than 429) is not retried
    assert KEY not in repr(provider) and "api_key=set" in repr(provider)
    with pytest.raises(LLMError, match="malformed response"):
        _gemini(Recorder(httpx.Response(200, json={"nope": 1}))).chat([{"role": "user", "content": "x"}])


def test_build_provider_reads_keys_from_environment_only():
    assert build_provider("none") is None
    with pytest.raises(LLMError, match=r"\$GEMINI_API_KEY"):
        build_provider("gemini", env={})
    gem = build_provider("gemini", env={"GEMINI_API_KEY": KEY})
    assert isinstance(gem, GeminiProvider) and gem.model == "gemini-2.5-flash" and gem.api_key == KEY
    oai = build_provider("openai", model="gpt-4o-mini", env={"OPENAI_API_KEY": "sk-test-1234"})
    assert oai.api_key == "sk-test-1234" and oai.base_url == "https://api.openai.com/v1"
    local = build_provider("openai", model="qwen2.5-coder", env={"OPENAI_BASE_URL": "http://localhost:8000/v1"})
    assert local.api_key is None and local.base_url == "http://localhost:8000/v1"
    with pytest.raises(LLMError, match="needs an API key"):
        build_provider("openai", env={})
    custom = build_provider("gemini", api_key_env="MY_KEY", env={"MY_KEY": KEY})
    assert custom.api_key == KEY
    with pytest.raises(LLMError, match="unknown"):
        build_provider("claude-but-not-really", env={})


def test_scrub_replaces_longest_secrets_first():
    assert scrub("tok_alpha tok_alpha_dev x", ["tok_alpha", "tok_alpha_dev"]) == "[REDACTED] [REDACTED] x"
    assert scrub("abc", ["ab"]) == "abc"  # too short to be a credential: left alone


# --------------------------------------------------------------------------- adapters


def _reverse_ranking(request):
    body = json.loads(request.content)
    user = json.loads(body["messages"][1]["content"])
    keys = [c["key"] for c in user["candidates"]][::-1]
    return _reply(json.dumps({"ranking": [{"key": k, "score": 1 - i / len(keys)} for i, k in enumerate(keys)]}))


def test_ranking_adapter_drives_llm_guided_search():
    rec = Recorder(_reverse_ranking)
    model = ProviderRankingModel(_gemini(rec), secrets=graph_secrets(FINTECH.graph))
    policy = LLMGuidedPolicy(model)
    result = search(FINTECH.graph, policy)
    assert policy.calls == len(rec.requests) > 0 and not policy.errors
    assert all(b["response_format"] == {"type": "json_object"} for b in rec.bodies)
    wire = "".join(r.content.decode() for r in rec.requests)
    assert KEY not in wire and "tok_" not in wire
    assert {k for k, c in result.coverage.items() if c.tests} == {
        k for k, c in search(FINTECH.graph).coverage.items() if c.tests}


def test_adapters_scrub_secrets_that_slip_into_prompts():
    rec = Recorder(_reply('{"ranking": []}'))
    model = ProviderRankingModel(_gemini(rec), secrets=["tok_alpha_dev"])
    from qbq.search.policy import RankingPrompt
    prompt = RankingPrompt(system="s", graph_slice={"note": "Bearer tok_alpha_dev"}, invariant={},
                           candidates=[], response_schema={})
    model.complete(prompt)
    assert "tok_alpha_dev" not in rec.requests[0].content.decode()
    assert "[REDACTED]" in model.sent[0][1]["content"]


def _finding(active, inv):
    plan = search(FINTECH.graph)
    candidate = next(c for c in plan.candidates() if c.invariant_id == inv)
    result = DeterministicVerifier(FINTECH.graph, FINTECH.app_factory(active), FINTECH.snapshot,
                                   runs=1).verify(candidate)
    return ConfirmedFinding.from_verification(FINTECH.graph, candidate, result), plan


def test_patch_adapter_diff_passes_the_fix_gate():
    active = frozenset({"loan_state_skip"})
    finding, plan = _finding(active, "inv.lifecycle.disburse_loan")
    rule_diff = propose_patches(finding, FINTECH.app_source)[0].diff
    rec = Recorder(_reply(f"Here you go:\n```diff\n{rule_diff}```"))
    patcher = LLMPatcher(ProviderPatchModel(_gemini(rec), secrets=graph_secrets(FINTECH.graph)))
    (candidate,) = propose_patches(finding, FINTECH.app_source, [patcher])
    body = rec.bodies[0]
    assert "response_format" not in body and "unified diff" in body["messages"][0]["content"]
    assert "tok_" not in rec.requests[0].content.decode()
    functional = [t.to_candidate() for t in plan.tests if t.oracle is Oracle.SUCCESS]
    fix = asyncio.run(averify_fix(finding, candidate, graph=FINTECH.graph, app_source=FINTECH.app_source,
                                  active_bugs=active, original_factory=FINTECH.app_factory(active),
                                  original_snapshot=FINTECH.snapshot, functional_tests=functional, runs=3))
    assert fix.verified and fix.exploit_violations == 0


def test_patch_adapter_failures_degrade_to_no_candidate():
    finding, _ = _finding(frozenset({"idor_wallet_read"}), "inv.tenant.read_wallet")
    patcher = LLMPatcher(ProviderPatchModel(_gemini(Recorder(httpx.Response(500, text="down")), max_retries=0)))
    assert propose_patches(finding, FINTECH.app_source, [patcher]) == []
    assert "HTTP 500" in patcher.errors[0]


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def mocked_gemini(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        if body.get("response_format"):
            return _reverse_ranking(request)
        return _reply("I cannot produce a diff for this.")

    rec = Recorder(handler)
    monkeypatch.setattr(providers_mod, "default_transport", httpx.MockTransport(rec))
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    return rec


def test_cli_search_with_gemini(mocked_gemini, tmp_path, capsys):
    out = tmp_path / "plan.json"
    assert main(["search", "--env", "saas-compiled", "--llm-provider", "gemini", "--json", str(out)]) == 0
    captured = capsys.readouterr()
    assert "policy=llm" in captured.out and "llm: gemini gemini-2.5-flash calls=" in captured.err
    assert mocked_gemini.requests
    for text in (captured.out, captured.err, out.read_text()):
        assert KEY not in text


def test_cli_remediate_with_gemini_keeps_secrets_out_of_artifacts(mocked_gemini, tmp_path, capsys):
    report, tests_dir = tmp_path / "report.json", tmp_path / "rt"
    code = main(["remediate", "--env", "fintech", "--runs", "1", "--llm-provider", "gemini",
                 "--out-dir", str(tests_dir), "--json", str(report), "--require-verified"])
    assert code == 0  # rule patches verify; the model's unusable replies are counted, not trusted
    captured = capsys.readouterr()
    patch_prompts = [b for b in mocked_gemini.bodies if not b.get("response_format")]
    data = json.loads(report.read_text())
    # one patch prompt per confirmed finding (the mock ranker reverses rankings, so it may cost recall)
    assert len(patch_prompts) == len(data["findings"]) >= 3
    assert f"unusable_replies={len(patch_prompts)}" in captured.err
    assert all(f["verified_patch"]["patcher"] == "rule" for f in data["findings"])
    artifacts = [report.read_text(), captured.out, captured.err] + [p.read_text() for p in tests_dir.iterdir()]
    assert all(KEY not in a for a in artifacts)


def test_cli_rejects_missing_key(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert main(["search", "--env", "saas-compiled", "--llm-provider", "gemini"]) == 2
    assert "needs an API key in $GEMINI_API_KEY" in capsys.readouterr().err
