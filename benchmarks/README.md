# Benchmarks

| Script | Measures |
| :--- | :--- |
| `compile_bench.py` | Gray-box compiler latency per phase (dynamic / static / merge+synthesis) and graph compression (`source_bytes` vs `compiled_graph_bytes`) on the fintech testbed |

```bash
uv run python benchmarks/compile_bench.py --iterations 25
```

Results land in `benchmarks/results/`. Byte counts and fidelity are deterministic.
Latencies depend on the machine, which is recorded alongside them.

On a tiny testbed (~10 KB of source), the full graph with 23 synthesized invariants
comes out *larger* than the source. The structural graph (nodes, edges, schemas,
provenance) is smaller. Compression is expected to matter for large codebases,
where the graph grows with routes and the source grows with everything else.
