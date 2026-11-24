# QbQ architecture

This document describes each layer of QbQ: what it guarantees and where it stops.
Numbers quoted here come from [`benchmarks/SUMMARY.md`](../benchmarks/SUMMARY.md),
which is generated from the committed result snapshots.

```
            ┌────────────────────────────────────────────────────────────────────┐
 source ───►│ compiler: static AST ─┐                                            │
 running ──►│           dynamic ────┼─► merge ─► invariants ─► persona grounding │──► StateGraph
 app        └───────────────────────┴────────────────────────────────────────────┘        │
                                                                                           ▼
                     ┌──── policy (heuristic | LLM ranker) ◄─── frames ◄── search (beam) ◄─┘
                     ▼
              5 combinators ─► ProposedTest ─► DeterministicVerifier (fresh app, 3x) ─► verdict
                                                     │                          │
                               confirmed findings ◄──┘                          └──► RLVR recorder
                                      │                                                ▲
                                      ▼                                                │
               RulePatcher | LLMPatcher ─► fix gate (exploit 0/3 + functional QA) ─────┤
                                      │                                                │
                                      └─► verified .patch + standalone pytest ─────────┘
```

## Principles

1. **Models propose, the sandbox decides.** No LLM output is ever a finding, a reward or
   a verdict. Findings are deterministic checks over HTTP responses and database
   snapshots, reproduced on every one of N replays (default 3) against freshly seeded
   app instances.
2. **The graph is the context.** Search, policies, prompts and patches all work on a
   compact typed graph and an AST slice, never on raw HTML, logs or whole repositories.
   Prompts never contain persona credentials.
3. **Determinism end to end.** Compilation, search plans, verdicts, patches, benchmark
   snapshots and the RLVR dataset are byte-reproducible. Wall-clock latency is the
   only exception, and it is labeled as such.

## 1. Behavioral State Graph (`qbq.graph`)

`G = (V, E)` holds three kinds of node: API endpoints, UI screens, and entity states
(`loan.approved`). Edges are HTTP actions annotated with auth, allowed roles, required
headers, payload JSON Schema, observed statuses and provenance metadata: source file and
line, extracted guards, mutated fields, ledger writes and lock use. Actors are seeded
personas, with credentials, owned resources, and the `observed_state` captured by
grounding. Invariants are pure data. A check is one of `status_code`,
`ledger_conservation` or `state_guard`, and each invariant carries taxonomy `tags`.
Referential integrity is validated on construction.

## 2. Compiler (`qbq.compiler`)

| Pass | Output |
| :--- | :--- |
| `static.py` | Python `ast` walk of the endpoint modules: routes, methods, path/body params, auth dependencies, role / ownership / lifecycle guards (followed into same-file helpers), data preconditions (`field != "CONST"` guards), state writes, mutated fields, ledger shape, lock use, Pydantic models, feature-flag conjuncts. |
| `dynamic.py` | Runs the ASGI lifespan, fetches OpenAPI 3.x with every `$ref` inlined, binds live routes to endpoint functions, and sends one unauthenticated probe per route. |
| `merge.py` | Joins the passes on (file, decorator line), falling back to (method, path). Runtime is authoritative for paths and schemas. Source supplies guards and provenance. |
| `grounding.py` | Each authenticated persona sends a `GET` to every read route whose path parameters bind to its own resources. Scalar response fields become `observed_state`, with credential-like field names dropped. |
| `invariants.py` | Rule-based synthesis (below). |
| `adapter.py` | Wraps any `module:factory` app plus a personas file (and an optional snapshot hook) as an `Environment` for search, remediation and RLVR. |

### Invariant rules

| Id pattern | Fires on | Expects | Tags |
| :--- | :--- | :--- | :--- |
| `inv.auth.<edge>` | authenticated edges | anonymous → 401 | OWASP-API2:2023, CWE-306 |
| `inv.tenant.<edge>` | auth'd edges with `{*_id}` path params | non-owner → 403/404 | OWASP-API1:2023, CWE-639 |
| `inv.input.<edge>.<f>` | money-like numeric fields (`amount`, `quantity`, ...) | non-positive → 400/422 | CWE-20 |
| `inv.ledger.conservation` | ledger-writing edges | balance = signed ledger sum, never negative | OWASP-API6:2023 |
| `inv.lifecycle.<edge>` | edges guarded by an entity state | out-of-order call → 4xx | OWASP-API6:2023, CWE-841 |
| `inv.tool_authz.<edge>` | mutating edges naming a tool (`tool_id`, `tool_name`, ...) | tool not granted to the caller's tenant → 403/404 | OWASP-LLM06:2025, OWASP-API5:2023, CWE-863 |
| `inv.ai_bounds.<edge>.<f>` | generation / agent-loop / rate params (`max_tokens`, `max_steps`, `timeout`, ...) | value above a sane bound → 400/413/422 | OWASP-LLM10:2025, OWASP-API4:2023, CWE-770 |
| `inv.functional.<edge>` | every edge | a valid request → documented 2xx | – |

Every rule is phrased so that it *holds* on a correct application. Rules that depend on
field names (`MONEY_FIELD`, `AI_LIMIT_FIELD`, `TOOL_FIELD`) are generic naming
heuristics, not per-app configuration.

## 3. Search (`qbq.search`)

**Oracles.** Each invariant is classified from its check structure alone (never its id
or text) as unauthenticated, forbidden, rejected-input, rejected-transition, success,
conservation or state-guard. Each combinator declares which oracles it can test.

| Combinator | Builds | Tests |
| :--- | :--- | :--- |
| `cross_persona_check` | A's recorded request replayed with B's credentials, keeping A's resource ids (with an owner control step for reads) | auth, tenant isolation |
| `parallel_contention` | N barrier-synchronized copies of a mutating edge (plus a read-after-write) | ledger conservation |
| `lifecycle_order_check` | a downstream lifecycle edge on a resource still in an earlier state | lifecycle / state guards |
| `boundary_perturb` | negative, zero, below-min, above-max, huge, missing, empty, wrong-type values, or the nominal request | input bounds, functional |
| `cross_tenant_reference` | the caller's own request with one *own* body reference (`tool_id`) pointing at another tenant's resource | tool authorization (confused deputy) |

**Policy.** `HeuristicGraphPolicy` scores frames using severity, oracle fit, static risk
signals (missing bounds, missing ownership guard, unlocked ledger writes, skippable
lifecycle steps) and an uncovered-invariant bonus. It scores concrete tests using persona
**state readiness** from grounding:
- funded accounts, following one reference hop such as `agent.tenant_id → tenants[t].token_balance`;
- the observed lifecycle status of the resource;
- observed data preconditions.

It also **prunes oracle-ambiguous tests**:
- role-permitted intruders;
- success probes whose state or preconditions can't hold at step 0;
- boundary cases that don't probe the invariant's bound direction;
- probes of fields the invariant doesn't name;
- credential swaps aimed at an invariant about a body reference.

`LLMGuidedPolicy` sends a credential-free `RankingPrompt` (graph slice, invariant, candidate
traces) to any `RankingModel`. The model can reorder tests but can never bring back a
pruned one.

**Beam search** expands StateNode → `(invariant, edge, combinator)` frame → tests, in
best-first order. It keeps `beam_width` tests per frame and `per_invariant` tests per
invariant, and lazily re-scores frames as coverage grows. Every uncovered invariant is
reported with a reason.

Why pruning matters (fintech ablation): without the policy, the full combinator pool
still yields 70 false positives *after* 3-of-3 gating (FPR 9.8%). Those tests reproduce
deterministically for the wrong reason, for example an admin allowed to read any wallet.
With the policy the FPR is 0%. The verifier and the policy are complementary.

## 4. Verifier (`qbq.verifier`)

For each candidate, `DeterministicVerifier` builds a fresh app (a freshly seeded store),
snapshots it, replays the trace over in-process ASGI, snapshots again, and evaluates the
invariant, repeating `runs` times. Concurrent steps release every request from a single
`asyncio.Barrier`. Verdicts:
- `confirmed`: violated on every run. It carries the HTTP exchanges, the DB row diff and a `curl` script.
- `flaky`: violated on some runs only. It is discarded.
- `not_reproduced`
- `invalid`

## 5. Remediation (`qbq.remediate`)

| Component | Behaviour |
| :--- | :--- |
| `RulePatcher` | `schema_lower_bound`: add `Field(gt=0)` (and the import) to the body field a lower-bound invariant names. `bypassed_guard`: remove a feature-flag or configuration conjunct (or wrapper) that switches off a guard raising a status the invariant expects. `locked_branch`: keep the lock-protected branch of a flag-selected read-modify-write. |
| `LLMPatcher` | Sends a line-numbered AST slice (handler, same-file helpers, request models) plus the invariant and the trace, without credentials, to any `PatchModel`, then applies the unified diff it returns. |
| Fix gate (`verify_fix.py`) | Imports the patched source from an isolated temp copy as a uniquely named module, rebuilds the app with the same build toggles, and requires **0/N exploit violations** and **no regression** of any functional QA test that passed before. Outcomes: `verified`, `exploit_reproduces`, `functional_regression`, `patch_invalid`. |
| Exporter | `test_regression_<invariant>.py` depends only on pytest, httpx and the app module. It replays the trace (with barriers), asserts the invariant inline, and runs 3 times. |
| Pipeline | `qbq remediate`: compile → search → verify → patch + gate → export. It also checks each exported test: it must fail on the unpatched build and pass on the patched code. |

**Limits.** The rule patcher fixes the gym defect *shapes*: an unbounded input field, a
guard switched off by a flag, and an unlocked read-modify-write next to a locked one. A
guard that was deleted outright needs `LLMPatcher` (or a person). The external-app test
suite shows a deleted ownership check fixed by a model-proposed diff and accepted by the
same gate. Patches are diffs against package-relative paths; apply them with
`git apply --directory=src`.

## 6. RLVR data engine (`qbq.rlvr`)

| Outcome | Reward | Rule |
| :--- | ---: | :--- |
| `true_violation` | +1 | confirmed 3/3 on the target build *and* not on the reference (patched) build |
| `patch_verified` | +1 | passes the fix gate |
| `holds` | 0 | well-formed test, invariant holds (valid negative) |
| `false_positive` | −1 | "violation" also reproduces on the reference build (oracle-inherent, not a defect) |
| `flaky` | −1 | reproduces on some runs only |
| `malformed` | −1 | unparseable JSON or a trace that does not fit the graph |
| `patch_rejected` | −1 | diff doesn't apply/build, exploit remains, or a functional edge regresses |

Rewards are *differential* against the reference build, so they never read the
planted-bug manifest. Ground truth appears only as an analysis label. The recorder covers
the **unpruned** combinator pool on every build, so the data includes good, useless and
wrong proposals. It adds one deterministic corruption per frame (format negatives), and
for each confirmed finding every rule patch plus two deliberately bad patches (a cosmetic
no-op and one that disables the endpoint).

Rows are TRL conversational `prompt` / `completion` records with `reward` and the columns
the reward function reads (`env`, `variant`, `invariant_id`). `VerifierReward` is a
TRL-compatible `reward_funcs` callable that scores model completions online in the
sandbox. TRL is not a dependency. Splits are by *application*: train on `fintech`, held-out
eval on `saas` + `agent_gateway`. Export fails if any prompt hash is shared across splits.

**Caveat.** Differential rewards need a known-good reference build. On a third-party app
there is none. There `VerifierReward` still separates malformed, flaky and holding
proposals, but `true_violation` versus `false_positive` needs a reference build or a
human-labelled baseline.

## 7. Extending QbQ

- **New app:** write a factory returning a fresh, seeded app, plus a personas JSON. Add a
  `snapshot(app) -> DbSnapshot` hook if it has a ledger or entity state. Use
  `qbq compile/remediate --app`.
- **New gym:** follow `qbq/gym/saas`. Write `create_app(active_bugs)` and
  `snapshot(app)`, then `environment.py` with personas and ground truth (bug →
  synthesized invariant id), and register it in `qbq/gym/__init__.py`.
- **New invariant rule:** add it to `compiler/invariants.py`, phrased to hold on correct
  apps and tagged. Check that an existing combinator can test its oracle; otherwise add one.
- **New model:** implement `RankingModel.complete(RankingPrompt) -> str` or
  `PatchModel.complete(PatchPrompt) -> str`. Malformed replies fall back to the
  heuristics, and the gate still decides.
