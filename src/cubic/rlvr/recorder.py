"""Trajectory recorder: every candidate test and every patch attempt becomes a scored record.

For one Bug Gym app the recorder emits three kinds of record:

* ``test`` records cover every proposal in the unpruned combinator pool, on every
  build variant. The policy's choices are only a subset of the pool, so the data
  spans good, useless and wrong proposals. Each row is labeled with whether the
  heuristic policy pruned it and whether it made the search plan.
* ``malformed`` records are one deterministic corruption per (invariant, edge)
  frame: unknown edge, unknown persona, missing path parameter, or truncated JSON.
  They teach the format, and all score -1.
* ``patch`` records cover, for each confirmed finding on each buggy build, every
  rule patch plus two deterministic bad patches: a cosmetic no-op and one that
  disables the endpoint. All are scored by the closed-loop fix gate.

Each record is ``(prompt, completion, reward)``. The prompt is a chat-format
State Graph slice with the target invariant, or the patch prompt's AST slice. The
completion is the action JSON or unified diff. The reward is the
deterministic score from ``cubic.rlvr.rewards``.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cubic.gym import load_environment
from cubic.gym.base import Environment
from cubic.remediate.models import ConfirmedFinding, FixVerdict, PatchCandidate
from cubic.remediate.patcher import (
    Edit,
    PatchError,
    RulePatcher,
    SourceContext,
    apply_edits,
    apply_unified_diff,
    build_patch_prompt,
    load_context,
    unified_diff,
)
from cubic.remediate.verify_fix import _verdicts, averify_fix
from cubic.rlvr.rewards import (
    REFERENCE_VARIANT,
    REWARDS,
    Outcome,
    RewardEngine,
    action_json,
    build_test_prompt,
    prompt_id,
)
from cubic.search import HeuristicGraphPolicy, Oracle, generate_pool, search
from cubic.verifier.replay import Verdict

APP_ENVIRONMENTS: dict[str, str] = {"fintech": "fintech-compiled", "saas": "saas-compiled"}


def load_app_environment(app: str) -> Environment:
    return load_environment(APP_ENVIRONMENTS.get(app, app))


class TrajectoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    split: str
    env: str
    variant: str
    task: str  # "test" | "patch"
    source: str  # "pool" | "malformed" | "rule_patch" | "synthetic_patch"
    prompt: list[dict[str, str]]
    prompt_id: str
    completion: str
    reward: float
    outcome: Outcome
    invariant_id: str
    finding: str | None = Field(default=None, description="CandidateFinding JSON (patch tasks)")
    labels: dict[str, Any] = Field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """TRL conversational prompt/completion row; extra columns feed the reward functions."""
        row = {
            "id": self.id, "split": self.split, "env": self.env, "variant": self.variant,
            "task": self.task, "source": self.source, "prompt": self.prompt, "prompt_id": self.prompt_id,
            "completion": [{"role": "assistant", "content": self.completion}],
            "reward": self.reward, "outcome": self.outcome.value, "invariant_id": self.invariant_id,
            "labels": self.labels,
        }
        if self.finding is not None:
            row["finding"] = self.finding
        return row


def _record(**kw: Any) -> TrajectoryRecord:
    pid = prompt_id(kw["prompt"])
    digest = hashlib.sha256(
        json.dumps([kw["split"], kw["env"], kw["variant"], kw["task"], pid, kw["completion"]]).encode()
    ).hexdigest()[:16]
    return TrajectoryRecord(id=f"{kw['env']}-{kw['task']}-{digest}", prompt_id=pid, **kw)


# --------------------------------------------------------------------------- malformed corruptions

_CORRUPTIONS = ("unknown_edge", "unknown_actor", "missing_path_param", "truncated_json")


def corrupt(text: str, kind: str) -> str:
    data = json.loads(text)
    step = data["steps"][0]
    if kind == "missing_path_param" and not step.get("path_params"):
        kind = "unknown_edge"
    if kind == "unknown_edge":
        step["edge_id"] = "edge.not_in_graph"
    elif kind == "unknown_actor":
        step["actor_id"] = "ghost_persona"
    elif kind == "missing_path_param":
        step.pop("path_params")
    else:
        return text[: len(text) // 2]
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- bad patches


def _line_offset(text: str, lineno: int) -> int:
    return sum(len(line) for line in text.splitlines(keepends=True)[: lineno - 1])


def synthetic_bad_patches(finding: ConfirmedFinding, ctx: SourceContext) -> list[PatchCandidate]:
    """Two deterministic, plausible-looking patches the gate must reject."""
    out = []
    indent = " " * ctx.handler.col_offset
    first_line = min([ctx.handler.lineno, *(d.lineno for d in ctx.handler.decorator_list)])
    def_start = _line_offset(ctx.text, first_line)
    cosmetic = [Edit(def_start, def_start, f"{indent}# reviewed: {finding.invariant.id}\n")]
    body = ctx.handler.body
    docstring = isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
    first_stmt = body[1] if docstring and len(body) > 1 else body[0]
    stmt_start = _line_offset(ctx.text, first_stmt.lineno)
    pad = " " * first_stmt.col_offset
    disable = [Edit(stmt_start, stmt_start, f'{pad}raise HTTPException(503, "endpoint disabled")\n')]
    for rule, edits, why in (("cosmetic", cosmetic, "Comment-only change."),
                             ("disable_endpoint", disable, "Disable the endpoint entirely.")):
        patched = apply_edits(ctx.text, edits)
        out.append(PatchCandidate(
            finding_id=finding.id, patcher="synthetic", rule=rule, file=ctx.display, description=why,
            diff=unified_diff(ctx.display, ctx.text, patched), patched_source=patched,
        ))
    return out


# --------------------------------------------------------------------------- recorder


async def arecord_environment(
    app: str,
    split: str,
    *,
    env: Environment | None = None,
    runs: int = 3,
    variants: Sequence[str] | None = None,
    include_patches: bool = True,
) -> list[TrajectoryRecord]:
    env = env or load_app_environment(app)
    graph = env.graph
    names = [v.name for v in env.variants() if variants is None or v.name in variants]
    if REFERENCE_VARIANT not in names:
        names.insert(0, REFERENCE_VARIANT)  # the reference build is always scored
    engine = RewardEngine(env, runs=runs)
    pool = generate_pool(graph)
    plan = search(graph)
    planned = {t.id for t in plan.tests}
    policy = HeuristicGraphPolicy()
    prune = {
        t.id: policy.assess(graph, t).pruned for t in pool
    }
    truth = {v.name: {env.bug(b).invariant_id for b in v.active_bugs} for v in env.variants()}

    records: list[TrajectoryRecord] = []
    prompts: dict[tuple[str, str], list[dict[str, str]]] = {}
    for test in pool:
        key = (test.invariant_id, test.edge_id)
        if key not in prompts:
            prompts[key] = build_test_prompt(graph, graph.invariant(test.invariant_id), test.edge_id)

    for variant in names:
        for test in pool:
            text = action_json(test.to_candidate())
            reward, outcome, target, reference = await engine.ascore_action(text, variant, test.invariant_id)
            records.append(_record(
                split=split, env=app, variant=variant, task="test", source="pool",
                prompt=prompts[(test.invariant_id, test.edge_id)], completion=text, reward=reward,
                outcome=outcome, invariant_id=test.invariant_id,
                labels={
                    "combinator": test.combinator.value, "oracle": test.oracle.value, "edge_id": test.edge_id,
                    "target_verdict": target.value if target else None,
                    "reference_verdict": reference.value if reference else None,
                    "ground_truth_violation": test.invariant_id in truth[variant],
                    "policy_pruned": prune[test.id], "in_search_plan": test.id in planned,
                },
            ))

    first_per_frame: dict[tuple[str, str], Any] = {}
    for test in pool:
        first_per_frame.setdefault((test.invariant_id, test.edge_id), test)
    for index, (key, test) in enumerate(sorted(first_per_frame.items())):
        kind = _CORRUPTIONS[index % len(_CORRUPTIONS)]
        text = corrupt(action_json(test.to_candidate()), kind)
        reward, outcome, *_ = await engine.ascore_action(text, "vulnerable", test.invariant_id)
        records.append(_record(
            split=split, env=app, variant="vulnerable", task="test", source="malformed",
            prompt=prompts[key], completion=text, reward=reward, outcome=outcome,
            invariant_id=test.invariant_id, labels={"corruption": kind, "edge_id": test.edge_id},
        ))

    if include_patches and env.app_source is not None:
        records += await _record_patches(app, split, env, engine, plan, names, runs)
    return sorted(records, key=lambda r: (r.task, r.source, r.variant, r.invariant_id, r.id))


async def _record_patches(app, split, env, engine, plan, names, runs) -> list[TrajectoryRecord]:
    graph, source = env.graph, env.app_source
    functional = [t.to_candidate() for t in plan.tests if t.oracle is Oracle.SUCCESS]
    out: list[TrajectoryRecord] = []
    for variant in env.variants():
        if not variant.active_bugs or variant.name not in names:
            continue
        build = env.app_factory(variant.active_bugs)
        baseline = await _verdicts(graph, build, env.snapshot, functional, runs)
        seen: set[str] = set()
        for candidate in plan.candidates():
            if candidate.invariant_id in seen:
                continue
            verifier = engine._verifier(variant.name)
            result = await verifier.averify(candidate)
            if not result.confirmed:
                continue
            seen.add(candidate.invariant_id)
            finding = ConfirmedFinding.from_verification(graph, candidate, result, variant=variant.name)
            try:
                ctx = load_context(finding, source)
            except PatchError:
                continue
            prompt = build_patch_prompt(finding, ctx).messages()
            patches = [(p, "rule_patch") for p in RulePatcher().propose(finding, ctx)]
            patches += [(p, "synthetic_patch") for p in synthetic_bad_patches(finding, ctx)]
            for patch, origin in patches:
                reward, outcome, fix = await ascore_patch(
                    patch.diff, finding, ctx, env=env, active=variant.active_bugs,
                    functional=functional, baseline=baseline, runs=runs)
                out.append(_record(
                    split=split, env=app, variant=variant.name, task="patch", source=origin, prompt=prompt,
                    completion=patch.diff, reward=reward, outcome=outcome, invariant_id=finding.invariant.id,
                    finding=candidate.model_dump_json(),
                    labels={"rule": patch.rule, "oracle": finding.oracle.value,
                            "fix_verdict": fix.value if fix else None},
                ))
    return out


async def ascore_patch(
    diff: str, finding: ConfirmedFinding, ctx: SourceContext, *, env: Environment, active: frozenset[str],
    functional: Sequence[Any], baseline: dict[str, Verdict] | None, runs: int,
) -> tuple[float, Outcome, FixVerdict | None]:
    """Apply a unified diff to the finding's source file and run it through the fix gate."""
    try:
        patched = apply_unified_diff(ctx.text, diff)
    except PatchError:
        return REWARDS[Outcome.PATCH_REJECTED], Outcome.PATCH_REJECTED, None
    candidate = PatchCandidate(finding_id=finding.id, patcher="scored", rule="diff", file=ctx.display,
                               description="", diff=diff, patched_source=patched)
    fix = await averify_fix(
        finding, candidate, graph=env.graph, app_source=env.app_source, active_bugs=active,
        original_factory=env.app_factory(active), original_snapshot=env.snapshot,
        functional_tests=functional, runs=runs, baseline=baseline,
    )
    outcome = Outcome.PATCH_VERIFIED if fix.verified else Outcome.PATCH_REJECTED
    return REWARDS[outcome], outcome, fix.verdict


__all__ = ["APP_ENVIRONMENTS", "TrajectoryRecord", "arecord_environment", "ascore_patch",
           "corrupt", "load_app_environment", "synthetic_bad_patches"]
