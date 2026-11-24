"""Autonomous evaluation: Bug Gym environments probed only by graph search.

``autonomous_environment`` takes an environment and replaces its hand-written
baseline probes with the plan that ``cubic.search`` derives from the
environment's ``StateGraph``. The search sees only the graph: nodes, edges,
personas and invariants. It never sees the planted-bug manifest, the probe list
or the application source. The harness uses ground truth only afterwards, to
score the plan.
"""

from __future__ import annotations

import dataclasses

from cubic.gym.base import Environment
from cubic.search import SearchConfig, SearchPolicy, SearchResult, search
from cubic.verifier.trace import CandidateFinding


def plan_for(
    env: Environment, policy: SearchPolicy | None = None, config: SearchConfig | None = None
) -> SearchResult:
    return search(env.graph, policy, config)


def autonomous_environment(
    base: Environment,
    *,
    policy: SearchPolicy | None = None,
    config: SearchConfig | None = None,
    name: str | None = None,
) -> Environment:
    def candidates() -> tuple[CandidateFinding, ...]:
        return plan_for(base, policy, config).candidates()

    return dataclasses.replace(
        base, name=name or f"{base.name}-autonomous", baseline_candidates=candidates
    )
