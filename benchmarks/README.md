# Benchmarks

| Script | Measures |
| :--- | :--- |
| `compile_bench.py` | Gray-box compiler latency per phase (dynamic / static / merge+synthesis) and graph compression (`source_bytes` vs `compiled_graph_bytes`) on the fintech testbed |
| `remediation_bench.py` | Both gyms (fintech, SaaS): Recall@1 / FPR with and without persona grounding; closed-loop patch pass rate, first-attempt pass rate, rules used, and regression-test validity for every build with planted bugs; per-stage wall clock of `cubic remediate` |
| `ablation_search.py` | Autonomous search ablations on the compiled fintech graph: (1) unfiltered proposals vs verifier-gated findings, for the guided plan and for the unpruned combinator pool; (2) uniform random edge sampling vs graph-guided search (Recall@1 at equal budget, tests needed to reach full recall) |

```bash
uv run python benchmarks/compile_bench.py --iterations 25
uv run python benchmarks/ablation_search.py --runs 3 --seeds 200   # ~25 s
uv run python benchmarks/remediation_bench.py --runs 3              # ~25 s
```

Results land in `benchmarks/results/`. Byte counts, fidelity and the whole search ablation
snapshot are deterministic: re-running `ablation_search.py` reproduces it byte for byte. In
`remediation_bench.json`, everything except the `latency_s` block is deterministic.
Latencies depend on the machine, which is recorded alongside them.

On a tiny testbed (~10 KB of source), the full graph with 23 synthesized invariants
comes out *larger* than the source. The structural graph (nodes, edges, schemas,
provenance) is smaller. Compression is expected to matter for large codebases,
where the graph grows with routes and the source grows with everything else.
