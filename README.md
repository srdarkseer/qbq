# QbQ ("Cubic")

An agentic QA & security testing engine. The LLM is treated as a search policy over a
typed **Behavioral State Graph**; only deterministic assertions on HTTP responses and
database state count as findings.

The full roadmap lives in [`global_agentic_qa_security_engine_build_plan.md`](global_agentic_qa_security_engine_build_plan.md).
This repository currently implements **Milestones 1 and 2**.

## Setup

```bash
uv sync
uv run pytest
uv run cubic eval --min-recall 1.0 --max-fpr 0.005   # CI gate
```

## Milestone 1 layout

| Package | Purpose |
| :--- | :--- |
| `cubic.graph` | Pydantic v2 schema: `StateNode`, `TransitionEdge`, `ActorPersona`, `InvariantAssertion`, `StateGraph`. Referential integrity is validated, and JSON round-trips losslessly and deterministically. |
| `cubic.verifier` | `ActionTrace` graph mutations (cross-account replay, barrier-synchronized concurrency spikes, state-machine skips), DB snapshots and diffs, deterministic check evaluation, and the `DeterministicVerifier`. |
| `cubic.gym` | Bug Gym environments. `fintech` is an in-process FastAPI wallet/lending testbed with seeded personas and four independently togglable planted defects. |
| `cubic.eval` | Runs candidates against every build variant (patched, each single bug, all bugs) and reports **Recall@1** and **False Positive Rate**. |

### Verification semantics

For each candidate the verifier builds a fresh, freshly seeded app instance, replays the
trace over ASGI, evaluates the invariant without an LLM, and repeats (default 3x).

- `confirmed`: the invariant broke on every run. The finding carries the HTTP exchanges,
  the DB row diff, and a replayable `curl` script.
- `flaky`: the invariant broke on some runs only. The finding is discarded.
- `not_reproduced` / `invalid`: the invariant held, or the trace doesn't fit the graph.

### Metrics

- **Recall@1**: for each (variant, planted bug), the *first* candidate that targets the bug's
  invariant must be confirmed.
- **FPR**: `FP / (FP + TN)` over candidates whose invariant is not broken by any active bug.

## Milestone 2: gray-box application compiler (`cubic.compiler`)

`cubic compile` turns an ASGI app into a `StateGraph` plus a baseline invariant set.
It needs no network access and no LLM.

| Module | Role |
| :--- | :--- |
| `static.py` | Walks the source with Python's `ast` module. It extracts routes, methods, path and body parameters, auth dependencies, role, ownership and lifecycle guards (followed into same-file helpers), state writes, mutated fields, ledger writes, lock usage, Pydantic models and store collections. Each result keeps its source file and line. |
| `dynamic.py` | Runs the app in-process. It starts the ASGI lifespan, fetches OpenAPI 3.x and inlines every `$ref`, binds live routes to their endpoint functions, and sends one unauthenticated probe per route to a fresh app instance. |
| `merge.py` | Joins the two passes on (file, decorator line), falling back to (method, path). Runtime data decides paths and schemas; source code supplies guards and provenance. Output is sorted and canonicalized, so the same app always produces byte-identical JSON. |
| `invariants.py` | Rule-based synthesis: auth (401), tenant isolation (403/404) on `{*_id}` routes, positivity on money-like numeric fields, ledger conservation, lifecycle preconditions, and functional 2xx checks. |
| `fidelity.py` | Scores a compiled graph against a reference graph: route coverage, auth, path params, payloads and lifecycle. Role agreement is reported separately. |

On the fintech testbed: 7/7 routes, structural fidelity 1.0 against the hand-written graph,
and 23 synthesized invariants. `cubic eval --env fintech-compiled` runs the Milestone 1
verifier on the compiled graph and gets Recall@1 100% and FPR 0%. Latency and compression
numbers are in [`benchmarks/`](benchmarks/).

### CLI

```bash
uv run cubic compile cubic.gym.fintech.app:create_app --out graph.json [--personas actors.json] [--report report.json]
uv run cubic compile path/to/app.py[:attr]   # module path or file; app instance or factory
uv run cubic eval [--env fintech|fintech-compiled] [--runs 3] [--json reports/eval.json]
uv run cubic graph            # dump the testbed State Graph
uv run cubic schema           # dump the State Graph JSON Schema
```

The testbed's defects are intentional. It runs only in-process, for local and authorized
CI/CD testing.
