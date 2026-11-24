"""Prioritized beam/tree search over ``(StateNode, InvariantAssertion, Combinator)``.

The search tree has three levels:

1. **StateNode**: every source node of an edge that some invariant constrains.
   Its priority is the best score among its frames (an optimistic bound).
2. **Frame** ``(node, invariant, edge, combinator)``: scored by the policy using
   the current coverage. Scores are re-checked lazily on pop, so a frame whose
   invariant has since been covered goes back on the queue with a lower score.
3. **Test**: the concrete ``ProposedTest`` objects the combinator expands to.
   The policy ranks them, drops pruned tests, and keeps the top
   ``beam_width``. A test's priority is its frame score plus its own score.

Tests are emitted in global priority order until every invariant has
``per_invariant`` tests or the ``max_tests`` budget runs out. Ties break on
stable ids, so the same graph and policy always give the same plan.
"""

from __future__ import annotations

import heapq
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from cubic.graph import StateGraph
from cubic.search.combinators import (
    Oracle,
    ProposedTest,
    applicable_combinators,
    classify_invariant,
    expand,
)
from cubic.search.policy import HeuristicGraphPolicy, ScoredTest, SearchFrame, SearchPolicy
from cubic.verifier.trace import CandidateFinding

_EPS = 1e-9


@dataclass(frozen=True)
class SearchConfig:
    beam_width: int = 3
    """Concrete tests kept per expanded frame."""
    per_invariant: int = 1
    """Tests emitted per invariant (the first one is what Recall@1 scores)."""
    max_tests: int | None = None
    """Global budget on emitted tests; ``None`` means until every invariant is filled."""

    def __post_init__(self) -> None:
        if self.beam_width < 1 or self.per_invariant < 1:
            raise ValueError("beam_width and per_invariant must be >= 1")
        if self.max_tests is not None and self.max_tests < 1:
            raise ValueError("max_tests must be >= 1")


class CoverageStatus(StrEnum):
    COVERED = "covered"
    UNCOVERED = "uncovered"


class InvariantCoverage(BaseModel):
    model_config = ConfigDict(frozen=True)

    invariant_id: str
    oracle: Oracle
    status: CoverageStatus
    tests: tuple[str, ...] = ()
    reason: str | None = None
    pruned: dict[str, int] = Field(default_factory=dict)


class EmittedTest(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int
    priority: float
    frame: SearchFrame
    scored: ScoredTest

    @property
    def test(self) -> ProposedTest:
        return self.scored.test


class SearchStats(BaseModel):
    nodes_expanded: int = 0
    frames_total: int = 0
    frames_expanded: int = 0
    frames_rescored: int = 0
    tests_generated: int = 0
    tests_pruned: int = 0
    tests_beam_dropped: int = 0
    tests_emitted: int = 0


class SearchResult(BaseModel):
    policy: str
    config: dict[str, int | None]
    emitted: tuple[EmittedTest, ...]
    coverage: dict[str, InvariantCoverage]
    stats: SearchStats

    @property
    def tests(self) -> tuple[ProposedTest, ...]:
        return tuple(e.test for e in self.emitted)

    def candidates(self) -> tuple[CandidateFinding, ...]:
        return tuple(t.to_candidate() for t in self.tests)

    @property
    def coverage_ratio(self) -> float:
        if not self.coverage:
            return 0.0
        covered = sum(c.status is CoverageStatus.COVERED for c in self.coverage.values())
        return covered / len(self.coverage)

    def uncovered(self) -> dict[str, str]:
        return {
            k: c.reason or "" for k, c in self.coverage.items() if c.status is CoverageStatus.UNCOVERED
        }


def build_frames(graph: StateGraph) -> tuple[SearchFrame, ...]:
    frames = []
    for invariant in sorted(graph.invariants, key=lambda i: i.id):
        for edge_id in invariant.edge_ids:
            node_id = graph.edge(edge_id).source
            for kind in applicable_combinators(invariant):
                frames.append(SearchFrame(node_id=node_id, invariant_id=invariant.id,
                                          edge_id=edge_id, combinator=kind))
    return tuple(frames)


class GraphSearch:
    """Best-first beam search that turns a ``StateGraph`` into an ordered test plan."""

    def __init__(
        self, graph: StateGraph, policy: SearchPolicy | None = None, config: SearchConfig | None = None
    ) -> None:
        self.graph = graph
        self.policy = policy or HeuristicGraphPolicy()
        self.config = config or SearchConfig()

    def run(self) -> SearchResult:
        graph, policy, cfg = self.graph, self.policy, self.config
        frames = build_frames(graph)
        stats = SearchStats(frames_total=len(frames))
        coverage: Counter[str] = Counter()
        emitted: list[EmittedTest] = []
        emitted_ids: set[str] = set()
        prune_reasons: dict[str, Counter[str]] = defaultdict(Counter)
        generated: Counter[str] = Counter()
        expanded: set[str] = set()

        by_node: dict[str, list[SearchFrame]] = defaultdict(list)
        for frame in frames:
            by_node[frame.node_id].append(frame)

        # heap entries: (-priority, level, key, seq, payload); seq keeps payloads uncompared
        heap: list[tuple[float, int, str, int, object]] = []
        seq = iter(range(1 << 62))

        def push(priority: float, level: int, key: str, payload: object) -> None:
            heapq.heappush(heap, (-priority, level, key, next(seq), payload))

        for node_id, node_frames in sorted(by_node.items()):
            bound = max(policy.score_frame(graph, f, coverage) for f in node_frames)
            push(bound, 0, node_id, node_id)

        def full(invariant_id: str) -> bool:
            return coverage[invariant_id] >= cfg.per_invariant

        while heap:
            if cfg.max_tests is not None and len(emitted) >= cfg.max_tests:
                break
            neg, level, key, _, payload = heapq.heappop(heap)
            priority = -neg

            if level == 0:  # StateNode -> frames
                stats.nodes_expanded += 1
                for frame in by_node[payload]:  # type: ignore[index]
                    push(policy.score_frame(graph, frame, coverage), 1, frame.key, frame)

            elif level == 1:  # frame -> concrete tests
                frame: SearchFrame = payload  # type: ignore[no-redef]
                if full(frame.invariant_id):
                    continue
                current = policy.score_frame(graph, frame, coverage)
                if current < priority - _EPS:
                    stats.frames_rescored += 1
                    push(current, 1, key, frame)
                    continue
                stats.frames_expanded += 1
                expanded.add(frame.invariant_id)
                tests = expand(graph, frame.combinator, graph.edge(frame.edge_id),
                               graph.invariant(frame.invariant_id))
                stats.tests_generated += len(tests)
                generated[frame.invariant_id] += len(tests)
                ranked = policy.rank(graph, frame, tests)
                live = [s for s in ranked if s.pruned is None]
                for s in ranked:
                    if s.pruned is not None:
                        stats.tests_pruned += 1
                        prune_reasons[frame.invariant_id][s.pruned] += 1
                stats.tests_beam_dropped += max(0, len(live) - cfg.beam_width)
                for s in live[: cfg.beam_width]:
                    push(current + s.score, 2, f"{frame.key}|{s.test.id}", (frame, s))

            else:  # emit a test
                frame, scored = payload  # type: ignore[misc]
                test = scored.test
                if full(test.invariant_id) or test.id in emitted_ids:
                    continue
                emitted.append(EmittedTest(rank=len(emitted), priority=priority, frame=frame, scored=scored))
                emitted_ids.add(test.id)
                coverage[test.invariant_id] += 1

        stats.tests_emitted = len(emitted)
        return SearchResult(
            policy=policy.name,
            config={"beam_width": cfg.beam_width, "per_invariant": cfg.per_invariant,
                    "max_tests": cfg.max_tests},
            emitted=tuple(emitted),
            coverage=_coverage(graph, emitted, prune_reasons, generated, expanded),
            stats=stats,
        )


def _coverage(
    graph: StateGraph, emitted: Sequence[EmittedTest],
    prune_reasons: dict[str, Counter[str]], generated: Counter[str], expanded: set[str],
) -> dict[str, InvariantCoverage]:
    tests_by_inv: dict[str, list[str]] = defaultdict(list)
    for e in emitted:
        tests_by_inv[e.test.invariant_id].append(e.test.id)
    out: dict[str, InvariantCoverage] = {}
    for invariant in sorted(graph.invariants, key=lambda i: i.id):
        oracle = classify_invariant(invariant)
        tests = tuple(tests_by_inv.get(invariant.id, ()))
        reasons = dict(sorted(prune_reasons.get(invariant.id, Counter()).items()))
        if tests:
            status, reason = CoverageStatus.COVERED, None
        elif not applicable_combinators(invariant):
            status, reason = CoverageStatus.UNCOVERED, f"no combinator tests oracle {oracle.value!r}"
        elif invariant.id not in expanded:
            status, reason = CoverageStatus.UNCOVERED, "search budget exhausted"
        elif not generated.get(invariant.id):
            status, reason = CoverageStatus.UNCOVERED, "combinators produced no replayable trace"
        elif reasons:
            status = CoverageStatus.UNCOVERED
            reason = "all tests pruned: " + "; ".join(reasons)
        else:
            status, reason = CoverageStatus.UNCOVERED, "search budget exhausted"
        out[invariant.id] = InvariantCoverage(
            invariant_id=invariant.id, oracle=oracle, status=status, tests=tests,
            reason=reason, pruned=reasons,
        )
    return out


def search(
    graph: StateGraph, policy: SearchPolicy | None = None, config: SearchConfig | None = None
) -> SearchResult:
    return GraphSearch(graph, policy, config).run()


def uniform_edge_sample(tests: Iterable[ProposedTest], seed: int) -> list[ProposedTest]:
    """Baseline ordering: repeatedly pick a uniformly random edge, then a random test on it.

    Uses no guard metadata and does no pruning. This is the ablation counterpart of
    graph-guided search.
    """
    rng = random.Random(seed)
    by_edge: dict[str, list[ProposedTest]] = defaultdict(list)
    for test in sorted(tests, key=lambda t: t.id):
        by_edge[test.edge_id].append(test)
    order: list[ProposedTest] = []
    while by_edge:
        edge = rng.choice(sorted(by_edge))
        bucket = by_edge[edge]
        order.append(bucket.pop(rng.randrange(len(bucket))))
        if not bucket:
            del by_edge[edge]
    return order
