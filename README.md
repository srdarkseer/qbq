# QbQ (pronounced "Cubic")

**An agentic QA & security testing engine that never trusts a model to judge a finding.**

QbQ compiles a web application into a typed **Behavioral State Graph**, searches that
graph for multi-persona test sequences that could break business, security or functional
invariants, and confirms every claim by replaying it **3 out of 3 times** on a freshly
seeded app. Confirmed findings go through a closed loop. A patch is synthesized and kept
only if the exploit stops reproducing *and* no functional QA flow regresses. Every
finding also becomes a standalone pytest regression test. Every proposal and every patch
attempt is recorded as a `(state-graph slice, action, verifiable reward)` trajectory for
RLVR training.

Models (an LLM ranker, an LLM patcher, or an RL-tuned policy) only ever *propose*.
Deterministic assertions on HTTP responses and database state decide.

| | Fintech | B2B SaaS | AI agent / MCP gateway |
| :--- | ---: | ---: | ---: |
| Recall@1 (8 planted (build, bug) pairs each) | 100% | 100% | 100% |
| False-positive rate | 0% (0/112) | 0% (0/172) | 0% (0/184) |
| Findings with a verified patch | 8/8 | 8/8 | 8/8 |

No hand-written probes, and no app-specific rules. Full tables, ablations and
reproduction commands are in [`benchmarks/SUMMARY.md`](benchmarks/SUMMARY.md).

## Quickstart

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                                  # full test suite

# 1. compile an app (source AST + live OpenAPI/runtime + persona grounding) into a State Graph
uv run qbq compile --app qbq.gym.saas.app:create_app --out graph.json

# 2. search the graph for a prioritized, multi-persona test plan
uv run qbq search --env saas-compiled

# 3. score the autonomous plan on every build of a Bug Gym app (CI gate)
uv run qbq eval --env saas-autonomous --min-recall 1.0 --max-fpr 0.0

# 4. closed loop: verify findings -> synthesize + gate patches -> export regression tests
uv run qbq remediate --env saas --out-dir regression_tests --patch-dir patches --require-verified

# 5. export verifiable-reward trajectories for GRPO / RLVR (train: fintech, eval: saas + agent_gateway)
uv run qbq export-rlvr --out datasets/qbq_rlvr.jsonl --stats rlvr_stats.json
```

### Your own FastAPI app

```bash
uv run qbq compile --app myservice.main:create_app --personas personas.json --out graph.json
uv run qbq remediate --app myservice.main:create_app --personas personas.json \
    [--snapshot myservice.main:db_snapshot] --out-dir regression_tests
```

- `--app` must name a **factory** that returns a fresh app with freshly seeded state on
  every call, because each replay gets its own instance.
- `personas.json` lists `ActorPersona` objects: the seeded test credentials and the
  resources each persona owns. A compiled `StateGraph` JSON works too.
- The optional snapshot hook returns a `DbSnapshot` and enables ledger and state-guard
  invariants. Without it, QbQ checks HTTP-level invariants only.
- Multi-file projects work as-is. The compiler follows `include_router` prefixes, `Depends`
  chains in `deps.py`/`auth.py` (including `require_role(...)` factories and router-level
  dependencies) and Pydantic v2 models in `schemas.py`.
  [`examples/storefront`](examples/storefront) is a worked example.

### Optional live model

The deterministic rule patcher only fixes known defect shapes. For anything else, such as a
deleted or forgotten guard, pass a model that *proposes* patches and test rankings. Every
proposal still has to pass the 3-of-3 fix gate.

```bash
export GEMINI_API_KEY=...            # read from the environment only; never logged or exported
uv run qbq remediate --app myservice.main:create_app --personas personas.json \
    --llm-provider gemini [--llm-model gemini-2.5-flash]
# any OpenAI-compatible server (OpenAI, vLLM, Ollama, LM Studio):
OPENAI_BASE_URL=http://localhost:11434/v1 uv run qbq search --env saas-compiled \
    --llm-provider openai --llm-model qwen2.5-coder
```

`--llm-provider none` (the default) keeps QbQ fully offline. API keys and persona credentials
are scrubbed from every outgoing prompt, and never appear in graphs, reports, patches or
datasets.

Only run QbQ against applications you own or are authorized to test.

## How it works

```mermaid
flowchart LR
    SRC["App source + running ASGI app"] --> C["Compiler<br/>static AST · OpenAPI/runtime<br/>persona grounding"]
    C --> G[("Behavioral State Graph<br/>+ synthesized invariants")]
    G --> S["Search<br/>5 combinators · policy · beam"]
    S --> V{"Deterministic verifier<br/>fresh app · replay 3x"}
    V -- confirmed --> R["Remediation<br/>patcher → fix gate"]
    R --> OUT["verified .patch<br/>+ pytest regression test"]
    V -- every proposal --> D[("RLVR trajectories<br/>JSONL")]
    R -- every patch --> D
    M["LLM ranker · LLM patcher<br/>RL-tuned policy"] -. proposes .-> S
    M -. proposes .-> R
```

| Layer | Package | What it does |
| :--- | :--- | :--- |
| Schema | `qbq.graph` | `StateNode`, `TransitionEdge`, `ActorPersona`, `InvariantAssertion`, `StateGraph`. Typed, validated, byte-deterministic JSON. |
| Compiler | `qbq.compiler` | Static AST pass (routes, auth, role/ownership/lifecycle guards, ledger writes, locks), dynamic pass (OpenAPI, live route binding, unauthenticated probes), persona grounding, and rule-based invariant synthesis with OWASP API 2023 / OWASP LLM 2025 / CWE tags. |
| Search | `qbq.search` | Five typed combinators: cross-persona replay, parallel contention, lifecycle skip, boundary perturbation, cross-tenant reference. Guard- and state-aware `HeuristicGraphPolicy`, pluggable `LLMGuidedPolicy`, best-first beam search with coverage reasons. |
| Verifier | `qbq.verifier` | Barrier-synchronized replay over in-process ASGI, DB snapshots and row diffs, deterministic checks, 3-of-3 gating, `curl` reproductions. |
| Remediation | `qbq.remediate` | AST rule patcher plus `LLMPatcher`. The fix gate requires 0/3 exploit runs and no functional regression. Also a standalone pytest exporter and the `qbq remediate` pipeline. |
| LLM | `qbq.llm` | Zero-dependency `httpx` providers for Gemini (OpenAI-compatible endpoint, JSON mode) and any OpenAI-compatible server: retries, usage accounting, secret scrubbing. Adapters plug them into `LLMGuidedPolicy` and `LLMPatcher`. |
| RLVR | `qbq.rlvr` | Trajectory recorder, programmatic rewards (+1 / 0 / −1), TRL-compatible `VerifierReward`, leakage-checked JSONL export. |
| Bug Gym | `qbq.gym` | Three in-process FastAPI testbeds with togglable defects: `fintech`, `saas`, `agent_gateway`. |
| Eval | `qbq.eval` | Recall@1 / FPR over every build, autonomous environments. |

The design, invariant rules, combinators, reward definitions and their limits are covered
in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The product roadmap is in
[`global_agentic_qa_security_engine_build_plan.md`](global_agentic_qa_security_engine_build_plan.md).

## Bug Gym

| Environment | App | Planted defects |
| :--- | :--- | :--- |
| `fintech-*` | Wallets, transfers, loans | IDOR wallet read · withdraw TOCTOU double-spend · loan disbursal skips approval · negative-transfer regression |
| `saas-*` | Workspaces, seat ledger, API keys, subscriptions | cross-tenant workspace read · seat-limit invite race · trial upgraded without activation · negative seat-purchase regression |
| `agent-gateway-*` | Tenants, token credits, agents, MCP-style tool calls, approval-gated actions | cross-tenant conversation read · cross-tenant tool invocation · unbounded `max_tokens` · action executed without human approval |

`*-compiled` environments use the auto-compiled graph. `*-autonomous` environments add the
search-generated plan as the probe set. `fintech` alone also ships a hand-written reference
graph and probes (`--env fintech`), used to measure compiler fidelity.

The testbeds are intentionally vulnerable. They run only in-process.

## CLI reference

| Command | Purpose |
| :--- | :--- |
| `qbq compile` | App → State Graph JSON (`--app`, `--personas`, `--report`, `--no-ground`, `--no-probe`, `--no-invariants`) |
| `qbq search` | Prioritized test plan and per-invariant coverage (`--env` or `--graph`, `--per-invariant`, `--max-tests`, `--llm-provider`) |
| `qbq eval` | Recall@1 / FPR across all builds, with CI gates (`--min-recall`, `--max-fpr`) |
| `qbq remediate` | Verify → patch → gate → export (`--env` or `--app`, `--variant`, `--patch-dir`, `--require-verified`, `--llm-provider`, `--llm-model`, `--api-key-env`) |
| `qbq export-rlvr` | Trajectory JSONL + stats (`--train`, `--eval`, `--sample-out`, `--stats`) |
| `qbq graph` / `qbq schema` | Dump an environment's graph / the State Graph JSON Schema |

## Repository layout

```
src/qbq/        graph · compiler · search · verifier · remediate · llm · rlvr · gym · eval · cli
tests/            pytest suite (unit, end-to-end, CLI, snapshot consistency)
benchmarks/       benchmark scripts, results/*.json snapshots, SUMMARY.md (generated)
datasets/         qbq_rlvr_sample.jsonl (stratified sample; full export is gitignored)
examples/         storefront: a multi-file FastAPI reference project with an unscripted IDOR
docs/             ARCHITECTURE.md
```

## Contributing, security, license

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md).

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your
option. Unless you explicitly state otherwise, any contribution you intentionally submit
for inclusion is dual-licensed as above, without additional terms or conditions.
