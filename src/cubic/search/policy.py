"""Search policies: how the prioritized graph search scores and ranks synthesized tests.

A ``SearchPolicy`` makes two decisions:

* ``score_frame``: how promising a ``(StateNode, InvariantAssertion, Combinator)``
  frame is, given which invariants the search has already covered.
* ``rank``: score and order the concrete ``ProposedTest`` objects a combinator
  produced for one frame. A test can also be *pruned*, with a reason.

``HeuristicGraphPolicy`` is the default. It is deterministic and makes no
external calls. It scores tests from static guard metadata, parameter types,
compile-time observations and invariant coverage. It also prunes tests whose
oracle is ambiguous: tests that would fail on a *correct* application, such as a
role-permitted intruder or a success probe whose lifecycle precondition cannot
hold at step 0. The 3-of-3 replay verifier only removes non-reproducible
results. A test that fails deterministically for the wrong reason would pass the
verifier, so these pruning rules are what keep the false-positive rate at zero.

``LLMGuidedPolicy`` is an optional adapter. It renders a compact graph slice,
the target invariant and the surviving candidates into a ``RankingPrompt``.
Any ``RankingModel`` (hosted or local) can return a JSON ranking from that
prompt. Guard pruning still applies: the model reorders tests but cannot bring
back a pruned one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from cubic.graph import (
    InvariantAssertion,
    LedgerConservationCheck,
    Severity,
    StateGraph,
    TransitionEdge,
)
from cubic.search.combinators import (
    BoundaryCase,
    CombinatorKind,
    Oracle,
    ProposedTest,
    _numeric_bounds,
    _properties,
    classify_invariant,
    initial_state_node,
    prerequisite_path,
    required_state,
    resource_stem,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SearchFrame(_Frozen):
    """One interior node of the search tree."""

    node_id: str
    invariant_id: str
    edge_id: str
    combinator: CombinatorKind

    @property
    def key(self) -> str:
        return f"{self.invariant_id}|{self.edge_id}|{self.combinator.value}"


class ScoredTest(_Frozen):
    test: ProposedTest
    score: float = Field(ge=0.0)
    pruned: str | None = None
    rationale: str = ""


@runtime_checkable
class SearchPolicy(Protocol):
    name: str

    def score_frame(
        self, graph: StateGraph, frame: SearchFrame, coverage: Mapping[str, int]
    ) -> float: ...

    def rank(
        self, graph: StateGraph, frame: SearchFrame, tests: Sequence[ProposedTest]
    ) -> list[ScoredTest]:
        """Return every test with a score, best first, pruned ones last."""
        ...


# --------------------------------------------------------------------------- heuristics

SEVERITY_WEIGHT = {
    Severity.CRITICAL: 2.0, Severity.HIGH: 1.6, Severity.MEDIUM: 1.0,
    Severity.LOW: 0.6, Severity.INFO: 0.2,
}

ORACLE_FIT: dict[tuple[CombinatorKind, Oracle], float] = {
    (CombinatorKind.CROSS_PERSONA, Oracle.FORBIDDEN): 1.0,
    (CombinatorKind.CROSS_PERSONA, Oracle.UNAUTHENTICATED): 0.9,
    (CombinatorKind.PARALLEL_CONTENTION, Oracle.CONSERVATION): 1.0,
    (CombinatorKind.LIFECYCLE_ORDER, Oracle.REJECTED_TRANSITION): 1.0,
    (CombinatorKind.LIFECYCLE_ORDER, Oracle.STATE_GUARD): 1.0,
    (CombinatorKind.BOUNDARY_PERTURB, Oracle.REJECTED_INPUT): 1.0,
    (CombinatorKind.BOUNDARY_PERTURB, Oracle.SUCCESS): 0.6,
}

UNCOVERED_BONUS = 2.0
COVERED_PENALTY = 1.0

# Boundary preference when the targeted field declares no lower bound vs. when it does.
_UNBOUNDED_CASES = {
    BoundaryCase.NEGATIVE: 1.0, BoundaryCase.ZERO: 0.9, BoundaryCase.BELOW_MIN: 0.8,
    BoundaryCase.ABOVE_MAX: 0.5, BoundaryCase.WRONG_TYPE: 0.3, BoundaryCase.MISSING: 0.3,
    BoundaryCase.EMPTY: 0.2,
}
_BOUNDED_CASES = {
    BoundaryCase.BELOW_MIN: 0.8, BoundaryCase.NEGATIVE: 0.7, BoundaryCase.ZERO: 0.7,
    BoundaryCase.ABOVE_MAX: 0.6, BoundaryCase.WRONG_TYPE: 0.3, BoundaryCase.MISSING: 0.3,
    BoundaryCase.EMPTY: 0.2,
}
_VALID_CASES = {BoundaryCase.NOMINAL: 1.0, BoundaryCase.MIN_VALID: 1.0, BoundaryCase.MAX_VALID: 0.6}


def _guards(edge: TransitionEdge) -> dict[str, Any]:
    return edge.metadata.get("guards") or {}


def mentions_field(invariant: InvariantAssertion, field: str) -> bool:
    """Whether the invariant names ``field`` (id segment or a whole word in its text)."""
    if field in invariant.id.split("."):
        return True
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])", invariant.description) is not None


def _lacks_lower_bound(edge: TransitionEdge, field: str) -> bool:
    prop = _properties(edge).get(field) or {}
    return prop.get("type") in {"integer", "number"} and _numeric_bounds(prop)[0] is None


def _conservation_entities(graph: StateGraph) -> set[str]:
    return {
        check.entity
        for inv in graph.invariants
        for check in inv.checks
        if isinstance(check, LedgerConservationCheck)
    }


def _ledger_coupled(graph: StateGraph, edge: TransitionEdge) -> bool:
    if edge.metadata.get("writes_ledger"):
        return True
    return any(
        edge.id in inv.edge_ids and classify_invariant(inv) is Oracle.CONSERVATION
        for inv in graph.invariants
    )


def _binds_ledger_account(graph: StateGraph, edge: TransitionEdge) -> bool:
    entities = _conservation_entities(graph)
    for param in edge.path_params:
        stem = resource_stem(param)
        if stem and any(e in {stem, stem + "s", stem + "es"} for e in entities):
            return True
    return False


def _unreachable_from_seed(graph: StateGraph, edge: TransitionEdge) -> bool:
    """The edge needs an entity state that seeded resources are not in."""
    requirement = required_state(graph, edge)
    if requirement is None:
        return False
    entity, source = requirement
    initial = initial_state_node(graph, entity)
    return initial is None or bool(prerequisite_path(graph, initial, source))


class HeuristicGraphPolicy:
    """Deterministic, guard-aware scoring. Makes no external calls."""

    name = "heuristic"

    # -- frames ------------------------------------------------------------------

    def score_frame(
        self, graph: StateGraph, frame: SearchFrame, coverage: Mapping[str, int]
    ) -> float:
        invariant = graph.invariant(frame.invariant_id)
        edge = graph.edge(frame.edge_id)
        oracle = classify_invariant(invariant)
        covered = coverage.get(invariant.id, 0)
        score = SEVERITY_WEIGHT[invariant.severity]
        score += 2.0 * ORACLE_FIT.get((frame.combinator, oracle), 0.0)
        score += self.risk(graph, edge, invariant, oracle)
        score += UNCOVERED_BONUS if covered == 0 else -COVERED_PENALTY * covered
        return score

    def risk(
        self, graph: StateGraph, edge: TransitionEdge, invariant: InvariantAssertion, oracle: Oracle
    ) -> float:
        """Prior that the invariant is breakable on this edge, from static facts only."""
        guards = _guards(edge)
        if oracle is Oracle.FORBIDDEN:
            return 0.3 if guards.get("ownership") else 1.0
        if oracle is Oracle.UNAUTHENTICATED:
            observed = edge.metadata.get("observed_unauthenticated_status")
            return 0.0 if observed == 401 else 1.0  # compile-time probe already saw it hold
        if oracle is Oracle.REJECTED_INPUT:
            fields = [f for f in _properties(edge) if mentions_field(invariant, f)]
            return 1.5 if any(_lacks_lower_bound(edge, f) for f in fields) else 0.2
        if oracle is Oracle.CONSERVATION:
            score = 0.5 if edge.metadata.get("writes_ledger") else 0.0
            score += 0.5 if _binds_ledger_account(graph, edge) else 0.0
            score += 0.5 if edge.metadata.get("uses_lock") is False else 0.0
            return score
        if oracle in {Oracle.REJECTED_TRANSITION, Oracle.STATE_GUARD}:
            return 1.0 if _unreachable_from_seed(graph, edge) else 0.0
        return 0.0

    # -- concrete tests ----------------------------------------------------------

    def rank(
        self, graph: StateGraph, frame: SearchFrame, tests: Sequence[ProposedTest]
    ) -> list[ScoredTest]:
        scored = [self.assess(graph, t) for t in tests]
        return sorted(scored, key=lambda s: (s.pruned is not None, -s.score, s.test.id))

    def assess(self, graph: StateGraph, test: ProposedTest) -> ScoredTest:
        edge, invariant = graph.edge(test.edge_id), graph.invariant(test.invariant_id)
        f = test.features
        score, why, pruned = 0.0, [], None

        if test.combinator is CombinatorKind.CROSS_PERSONA:
            if not f["intruder_authenticated"]:
                score, why = 0.8, ["unauthenticated replay"]
            elif f["intruder_role"] == f["owner_role"]:
                score, why = 1.0, ["horizontal peer replay"]
            elif edge.allowed_roles and f["intruder_role"] not in edge.allowed_roles:
                score, why = 0.9, ["intruder role not permitted on edge"]
            else:
                pruned = (
                    f"intruder role {f['intruder_role']} may be permitted by role on "
                    f"{edge.id}; tenant oracle is ambiguous"
                )
            if f.get("control_step"):
                score += 0.1

        elif test.combinator is CombinatorKind.PARALLEL_CONTENTION:
            score = 0.5
            if _binds_ledger_account(graph, edge):
                score += 1.0
                why.append("contends on a single ledger account")
            score += 0.2 * f["concurrency"] / 16
            if f.get("observation_step"):
                score += 0.1
            score -= 0.005 * test.request_count

        elif test.combinator is CombinatorKind.LIFECYCLE_ORDER:
            if f["actor_owns_resource"]:
                score, why = 1.0, ["owner isolates the lifecycle guard"]
            elif _guards(edge).get("ownership"):
                pruned = "ownership guard would reject a non-owner before the lifecycle guard"
            else:
                score, why = 0.3, ["non-owner (no ownership guard extracted)"]
            score += 0.2 * len(f["skipped_edges"])

        elif test.combinator is CombinatorKind.BOUNDARY_PERTURB:
            case = BoundaryCase(f["case"])
            if test.oracle is Oracle.SUCCESS:
                pruned = self._success_infeasible(graph, edge, test)
                score = _VALID_CASES.get(case, 0.0)
            else:
                field = f["field"]
                if field is None or not mentions_field(invariant, field):
                    pruned = f"{invariant.id} does not reference field {field!r}"
                else:
                    table = _UNBOUNDED_CASES if _lacks_lower_bound(edge, field) else _BOUNDED_CASES
                    score = table.get(case, 0.1)
                    why.append(f"{case.value} on referenced field {field}")

        return ScoredTest(test=test, score=max(score, 0.0), pruned=pruned, rationale="; ".join(why))

    @staticmethod
    def _success_infeasible(graph: StateGraph, edge: TransitionEdge, test: ProposedTest) -> str | None:
        actor = graph.actor(test.features["actor"])
        if _unreachable_from_seed(graph, edge):
            return "lifecycle precondition unreachable from seeded state at step 0"
        if _guards(edge).get("preconditions"):
            return "edge has data preconditions the graph cannot verify"
        if edge.auth_required and not actor.owned_resources:
            return f"{actor.id} owns no seeded resources; not a representative principal"
        if _ledger_coupled(graph, edge):
            return "success depends on account balances the graph does not model"
        return None


# --------------------------------------------------------------------------- LLM adapter


class RankingPrompt(_Frozen):
    """Structured ranking request: a compact graph slice, one invariant, candidate tests."""

    system: str
    graph_slice: dict[str, Any]
    invariant: dict[str, Any]
    candidates: list[dict[str, Any]]
    response_schema: dict[str, Any]

    def render(self) -> str:
        payload = {
            "graph_slice": self.graph_slice, "invariant": self.invariant,
            "candidates": self.candidates, "respond_with": self.response_schema,
        }
        return json.dumps(payload, indent=None, sort_keys=True, separators=(",", ":"))

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.render()}]


@runtime_checkable
class RankingModel(Protocol):
    """Anything that turns a ``RankingPrompt`` into a JSON ranking string."""

    def complete(self, prompt: RankingPrompt) -> str: ...


RANKING_SYSTEM_PROMPT = (
    "You rank candidate test sequences for an application state graph. Each candidate "
    "is a trace of HTTP edges executed by personas. Rank candidates by how likely they "
    "are to violate the target invariant on a buggy build while holding on a correct "
    "build. Respond with JSON only, matching `respond_with`."
)

RANKING_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["ranking"],
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["key"],
                "properties": {"key": {"type": "string"}, "score": {"type": "number"},
                               "reason": {"type": "string"}},
            },
        }
    },
}


def _edge_summary(edge: TransitionEdge, *, detail: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"id": edge.id, "method": edge.method.value, "path": edge.path,
                           "source": edge.source, "target": edge.target}
    if detail:
        out |= {
            "auth_required": edge.auth_required, "allowed_roles": sorted(edge.allowed_roles),
            "payload_schema": edge.payload_schema, "guards": _guards(edge),
            "response_statuses": sorted(edge.response_statuses),
        }
    return out


def graph_slice(graph: StateGraph, edge_id: str) -> dict[str, Any]:
    """The edge, its endpoint nodes, adjacent edges, and personas (without credentials)."""
    edge = graph.edge(edge_id)
    adjacent = sorted(
        {e.id for e in graph.incoming(edge.source)} | {e.id for e in graph.outgoing(edge.target)}
        - {edge.id}
    )
    return {
        "app": graph.app_name,
        "edge": _edge_summary(edge, detail=True),
        "nodes": [graph.node(n).model_dump(mode="json", exclude={"metadata"})
                  for n in dict.fromkeys((edge.source, edge.target))],
        "adjacent_edges": [_edge_summary(graph.edge(e), detail=False) for e in adjacent],
        "actors": [
            {"id": a.id, "role": a.role, "authenticated": a.is_authenticated,
             "owns": {k: sorted(v) for k, v in sorted(a.owned_resources.items())}}
            for a in sorted(graph.actors, key=lambda a: a.id)
        ],
    }


def build_ranking_prompt(
    graph: StateGraph, frame: SearchFrame, tests: Sequence[ProposedTest]
) -> tuple[RankingPrompt, dict[str, ProposedTest]]:
    invariant = graph.invariant(frame.invariant_id)
    keys = {f"c{i}": t for i, t in enumerate(tests)}
    candidates = [
        {
            "key": key, "combinator": t.combinator.value, "hypothesis": t.hypothesis,
            "steps": [s.model_dump(mode="json", exclude_defaults=True) for s in t.trace.steps],
        }
        for key, t in keys.items()
    ]
    prompt = RankingPrompt(
        system=RANKING_SYSTEM_PROMPT,
        graph_slice=graph_slice(graph, frame.edge_id),
        invariant={
            "id": invariant.id, "category": invariant.category.value,
            "severity": invariant.severity.value, "description": invariant.description,
            "oracle": classify_invariant(invariant).value,
            "checks": [c.model_dump(mode="json") for c in invariant.checks],
        },
        candidates=candidates,
        response_schema=RANKING_RESPONSE_SCHEMA,
    )
    return prompt, keys


def parse_ranking(text: str, valid_keys: Sequence[str]) -> list[tuple[str, float]]:
    """Parse a model reply into ``[(key, score)]``; tolerant of code fences and prose.

    Unknown or duplicate keys are dropped. A missing score falls back to rank order.
    Raises ``ValueError`` if no JSON ranking can be found.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("no JSON object in model response")
    data = json.loads(match.group(0))
    items = data.get("ranking") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("response has no 'ranking' list")
    allowed, seen, out = set(valid_keys), set(), []
    n = len(items)
    for index, item in enumerate(items):
        key = item.get("key") if isinstance(item, dict) else item
        if not isinstance(key, str) or key not in allowed or key in seen:
            continue
        seen.add(key)
        raw = item.get("score") if isinstance(item, dict) else None
        score = float(raw) if isinstance(raw, (int, float)) else (n - index) / n
        out.append((key, min(max(score, 0.0), 1.0)))
    return out


class LLMGuidedPolicy:
    """Rank concrete tests with an external or local model, with heuristics as a fallback.

    Frame scoring and guard pruning come from ``fallback``. The model ranks only
    tests that survive pruning. Tests the model ranks come first, ordered by the
    model's score. Tests it leaves out keep their fallback order after them. If
    the model call fails or its reply can't be parsed, the frame falls back to
    the heuristic ranking and the error is recorded in ``errors``.
    """

    name = "llm"

    def __init__(
        self, model: RankingModel, fallback: HeuristicGraphPolicy | None = None,
        *, max_candidates: int = 24,
    ) -> None:
        self.model = model
        self.fallback = fallback or HeuristicGraphPolicy()
        self.max_candidates = max_candidates
        self.calls = 0
        self.errors: list[str] = []
        self.prompts: list[RankingPrompt] = []

    def score_frame(
        self, graph: StateGraph, frame: SearchFrame, coverage: Mapping[str, int]
    ) -> float:
        return self.fallback.score_frame(graph, frame, coverage)

    def rank(
        self, graph: StateGraph, frame: SearchFrame, tests: Sequence[ProposedTest]
    ) -> list[ScoredTest]:
        base = self.fallback.rank(graph, frame, tests)
        live = [s for s in base if s.pruned is None][: self.max_candidates]
        pruned = [s for s in base if s.pruned is not None]
        if len(live) < 2:
            return base
        prompt, keys = build_ranking_prompt(graph, frame, [s.test for s in live])
        self.prompts.append(prompt)
        self.calls += 1
        try:
            ranking = parse_ranking(self.model.complete(prompt), list(keys))
        except Exception as exc:  # noqa: BLE001 - any model failure degrades to heuristics
            self.errors.append(f"{frame.key}: {exc}")
            return base
        by_id = {s.test.id: s for s in live}
        ranked_ids = [keys[k].id for k, _ in ranking]
        scores = {keys[k].id: score for k, score in ranking}
        top = sorted(ranked_ids, key=lambda tid: (-scores[tid], ranked_ids.index(tid)))
        ordered = [
            by_id[tid].model_copy(update={"score": 1.0 + scores[tid], "rationale": "model-ranked"})
            for tid in top
        ]
        rest = [s.model_copy(update={"score": min(s.score, 1.0) * 0.5}) for s in live if s.test.id not in scores]
        return ordered + rest + pruned
