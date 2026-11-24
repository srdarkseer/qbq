"""Live smoke test against a real model. Skipped unless ``pytest --live`` and an API key are present.

    GEMINI_API_KEY=... uv run pytest tests/test_live_llm.py --live -q

Optional: ``QBQ_LIVE_PROVIDER`` (gemini|openai), ``QBQ_LIVE_MODEL``, ``QBQ_LIVE_KEY_ENV``.
The model must fix the storefront's missing ownership check with a diff that passes the
3-of-3 exploit gate and the functional QA regression checks.
"""

import os
import sys
from pathlib import Path

import pytest

from qbq.compiler.adapter import external_environment, load_personas
from qbq.llm import ProviderPatchModel, build_provider, graph_secrets
from qbq.remediate import LLMPatcher, RulePatcher, remediate

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
KEY_ENV = os.environ.get("QBQ_LIVE_KEY_ENV", "GEMINI_API_KEY")

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get(KEY_ENV), reason=f"${KEY_ENV} not set")
def test_live_model_fixes_the_storefront_idor(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    provider = build_provider(os.environ.get("QBQ_LIVE_PROVIDER", "gemini"),
                              model=os.environ.get("QBQ_LIVE_MODEL", "gemini-2.5-flash"), api_key_env=KEY_ENV)
    env = external_environment("storefront.main:create_app", load_personas(EXAMPLES / "storefront/personas.json"),
                               snapshot="storefront.main:snapshot")
    patcher = LLMPatcher(ProviderPatchModel(provider, secrets=graph_secrets(env.graph)))
    report = remediate(env, runs=3, patchers=[RulePatcher(), patcher], out_dir=tmp_path)
    (finding,) = report.findings
    detail = [(a.verdict.value, a.error, a.regressions) for a in finding.attempts] or patcher.errors
    print(f"\nlive {provider.name} {provider.model}: usage={provider.usage.as_dict()} attempts={detail}",
          file=sys.stderr)
    assert finding.verified, f"model patch was not verified: {detail}"
    assert finding.regression_test.valid
    assert provider.api_key not in finding.verified_patch.diff
