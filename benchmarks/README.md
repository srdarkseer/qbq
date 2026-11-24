# Benchmarks

**Start with [`SUMMARY.md`](SUMMARY.md).** It holds all results as paper-ready tables,
generated from the snapshots in `results/` by `summarize.py`.

| Script | Snapshot | Measures |
| :--- | :--- | :--- |
| `compile_bench.py` | `results/compile_fintech.json` | Compiler latency per phase (dynamic / static / merge+synthesis / grounding), graph compression, and structural fidelity against the hand-written fintech graph |
| `ablation_search.py` | `results/search_ablation.json` | Fintech: (1) unfiltered proposals vs verifier-gated findings, for the guided plan and for the unpruned combinator pool; (2) uniform random edge sampling vs graph-guided search (Recall@1 at equal budget, tests needed to reach full recall) |
| `remediation_bench.py` | `results/remediation_bench.json` | All three gyms: Recall@1 / FPR with and without persona grounding, patch pass rate, first-attempt pass rate, rules used, regression-test validity, per-stage wall clock of `cubic remediate` |
| `cubic export-rlvr --stats` | `results/rlvr_stats.json` | RLVR dataset: rows, prompts, reward histograms per split, leakage check, dataset SHA-256 |
| `summarize.py` | `SUMMARY.md` | Renders every table above; `--check` fails if SUMMARY.md is stale |

```bash
uv run python benchmarks/compile_bench.py --iterations 25
uv run python benchmarks/ablation_search.py --runs 3 --seeds 200   # ~25 s
uv run python benchmarks/remediation_bench.py --runs 3              # ~35 s
uv run cubic export-rlvr --out datasets/cubic_rlvr.jsonl \
    --sample-out datasets/cubic_rlvr_sample.jsonl --stats benchmarks/results/rlvr_stats.json   # ~80 s
uv run python benchmarks/summarize.py
```

Everything is deterministic except wall-clock latency (`latency` in
`compile_fintech.json`, `latency_s` in `remediation_bench.json`). Re-running a script
reproduces its other fields byte for byte, and the RLVR export reproduces the same
dataset SHA-256. Latencies depend on the machine, which is recorded with them.

On a tiny testbed (~10 KB of source), the full graph with its synthesized invariants comes
out *larger* than the source. The structural graph (nodes, edges, schemas, provenance)
is smaller. Compression is expected to matter for large codebases, where the graph grows
with routes and the source grows with everything else.
