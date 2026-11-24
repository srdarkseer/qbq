# Contributing

```bash
uv sync
uv run pytest                                   # must stay green
uv run python benchmarks/summarize.py --check   # SUMMARY.md must match results/*.json
```

- **Determinism is a feature.** Graphs, plans, verdicts, patches, benchmark snapshots and
  the RLVR export must be byte-reproducible. If a change moves a snapshot, regenerate it
  with the commands at the end of `benchmarks/SUMMARY.md` and commit the regenerated file.
- **No model is ever the judge.** New features may *propose* with a model. Only
  deterministic checks may confirm findings, verify patches or assign rewards.
- **No app-specific rules.** Compiler and search rules must be generic, keyed on graph
  structure, schemas, guards or naming conventions. They must never be keyed on a
  particular testbed's routes. Ground truth lives only in a gym's `environment.py`.
- **Commits:** one cohesive change per commit, conventional `type: subject` messages.
- **Licensing:** contributions are dual-licensed under MIT and Apache-2.0 (see `README.md`).

See `docs/ARCHITECTURE.md` for the extension points: new apps, gyms, invariant rules and models.
