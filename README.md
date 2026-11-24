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
```
