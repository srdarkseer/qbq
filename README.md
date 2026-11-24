# QbQ ("Cubic")

An agentic QA & security testing engine. The LLM is treated as a search policy over a
typed **Behavioral State Graph**; only deterministic assertions on HTTP responses and
database state count as findings.

The full roadmap lives in [`global_agentic_qa_security_engine_build_plan.md`](global_agentic_qa_security_engine_build_plan.md).
This repository currently implements **Milestone 1** only.

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

### CLI

```bash
uv run cubic eval [--env fintech] [--runs 3] [--json reports/eval.json]
uv run cubic graph            # dump the testbed State Graph
uv run cubic schema           # dump the State Graph JSON Schema
```

The testbed's defects are intentional. It runs only in-process, for local and authorized
CI/CD testing.
