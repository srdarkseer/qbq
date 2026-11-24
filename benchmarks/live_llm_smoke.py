"""Live LLM remediation smoke test on the multi-file storefront example.

    GEMINI_API_KEY=... uv run python benchmarks/live_llm_smoke.py [--provider gemini] [--model gemini-2.5-flash]
    OPENAI_BASE_URL=http://localhost:11434/v1 uv run python benchmarks/live_llm_smoke.py --provider openai --model qwen2.5-coder

QbQ compiles ``examples/storefront``, confirms its unscripted IDOR 3-of-3, and asks the
model for a patch. It prints whether the patch passed the fix gate, along with token
usage. Live model output is not deterministic, so unlike the other benchmarks this
script commits no snapshot. ``--out`` writes the result JSON. The API key is never
printed or written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=("gemini", "openai"), default="gemini")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "examples"))
    from qbq.compiler.adapter import external_environment, load_personas
    from qbq.llm import LLMError, ProviderPatchModel, build_provider, graph_secrets
    from qbq.remediate import LLMPatcher, RulePatcher, remediate

    try:
        provider = build_provider(args.provider, model=args.model, api_key_env=args.api_key_env,
                                  base_url=args.base_url)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    env = external_environment("storefront.main:create_app",
                               load_personas(ROOT / "examples/storefront/personas.json"),
                               snapshot="storefront.main:snapshot")
    patcher = LLMPatcher(ProviderPatchModel(provider, secrets=graph_secrets(env.graph)))
    started = time.perf_counter()
    report = remediate(env, runs=args.runs, patchers=[RulePatcher(), patcher])
    result = {
        "provider": provider.name, "model": provider.model, "runs": args.runs,
        "wall_clock_s": round(time.perf_counter() - started, 2), "usage": provider.usage.as_dict(),
        "unusable_replies": patcher.errors,
        "findings": [{
            "invariant_id": f.invariant_id, "source": f.source, "verified": f.verified,
            "patcher": f.verified_patch.patcher if f.verified_patch else None,
            "attempts": [{"verdict": a.verdict.value, "exploit_violations": a.exploit_violations,
                          "regressions": a.regressions} for a in f.attempts],
            "diff": f.verified_patch.diff if f.verified_patch else None,
        } for f in report.findings],
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0 if report.findings and all(f.verified for f in report.findings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
