"""Dataset export: deterministic JSONL, environment-level splits, leakage checks and stats.

Splits are by *application*. By default the ``fintech`` app is train, and ``saas`` and
``agent_gateway`` are held-out eval. No prompt, graph or environment appears in more than one split, so
eval reward measures cross-domain generalization, not memorization. The export
fails if any prompt hash is shared between splits.

Every row is a TRL conversational prompt/completion record (``prompt``,
``completion``) with a ``reward`` and the columns the reward functions read
(``env``, ``variant``, ``invariant_id`` and, for patches, ``finding``). Rows are
sorted and serialized with sorted keys, so the same inputs give byte-identical
output.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from cubic.rlvr.recorder import TrajectoryRecord, arecord_environment, load_app_environment
from cubic.rlvr.rewards import REWARDS

DEFAULT_SPLITS: dict[str, tuple[str, ...]] = {"train": ("fintech",), "eval": ("saas", "agent_gateway")}


class LeakageError(ValueError):
    """The same prompt or environment appears in more than one split."""


def build_dataset(
    splits: Mapping[str, Sequence[str]] = DEFAULT_SPLITS,
    *,
    runs: int = 3,
    variants: Sequence[str] | None = None,
    include_patches: bool = True,
) -> list[TrajectoryRecord]:
    owners: dict[str, str] = {}
    for split, apps in splits.items():
        for app in apps:
            if app in owners:
                raise LeakageError(f"app {app!r} assigned to both {owners[app]!r} and {split!r}")
            owners[app] = split
    records: list[TrajectoryRecord] = []
    for split, apps in splits.items():
        for app in apps:
            env = load_app_environment(app)  # compiles synchronously, outside the event loop
            records += asyncio.run(arecord_environment(
                app, split, env=env, runs=runs, variants=variants, include_patches=include_patches))
    check_leakage(records)
    return sorted(records, key=lambda r: (r.split, r.env, r.task, r.source, r.variant, r.invariant_id, r.id))


def check_leakage(records: Iterable[TrajectoryRecord]) -> dict[str, Any]:
    by_split: dict[str, set[str]] = defaultdict(set)
    envs: dict[str, set[str]] = defaultdict(set)
    for r in records:
        by_split[r.split].add(r.prompt_id)
        envs[r.split].add(r.env)
    splits = sorted(by_split)
    shared_prompts = sorted(set.intersection(*(by_split[s] for s in splits))) if len(splits) > 1 else []
    shared_envs = sorted(set.intersection(*(envs[s] for s in splits))) if len(splits) > 1 else []
    if shared_prompts or shared_envs:
        raise LeakageError(f"shared across splits: prompts={shared_prompts[:5]} envs={shared_envs}")
    return {"shared_prompts": 0, "shared_envs": [], "envs_by_split": {s: sorted(envs[s]) for s in splits}}


def jsonl_lines(records: Iterable[TrajectoryRecord]) -> list[str]:
    return [json.dumps(r.to_row(), sort_keys=True, ensure_ascii=False, separators=(",", ":")) for r in records]


def write_jsonl(records: Sequence[TrajectoryRecord], path: Path | str) -> str:
    """Write rows; returns the SHA-256 of the file contents."""
    text = "".join(line + "\n" for line in jsonl_lines(records))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def stratified_sample(records: Sequence[TrajectoryRecord], per_stratum: int = 2) -> list[TrajectoryRecord]:
    """Up to ``per_stratum`` rows per (split, task, source, outcome), chosen by stable id order."""
    groups: dict[tuple[str, str, str, str], list[TrajectoryRecord]] = defaultdict(list)
    for r in records:
        groups[(r.split, r.task, r.source, r.outcome.value)].append(r)
    picked = [r for key in sorted(groups) for r in sorted(groups[key], key=lambda r: r.id)[:per_stratum]]
    return sorted(picked, key=lambda r: (r.split, r.env, r.task, r.source, r.variant, r.invariant_id, r.id))


def _summary(records: Sequence[TrajectoryRecord]) -> dict[str, Any]:
    rewards = [r.reward for r in records]
    tests = [r for r in records if r.task == "test" and r.source == "pool"]
    plan = [r for r in tests if r.labels.get("in_search_plan")]
    pruned = [r for r in tests if r.labels.get("policy_pruned")]
    return {
        "envs": sorted({r.env for r in records}),
        "rows": len(records),
        "prompts": len({r.prompt_id for r in records}),
        "by_task": dict(sorted(Counter(f"{r.task}/{r.source}" for r in records).items())),
        "by_outcome": dict(sorted(Counter(r.outcome.value for r in records).items())),
        "reward_histogram": {f"{k:+.1f}": v for k, v in sorted(Counter(rewards).items())},
        "mean_reward": round(sum(rewards) / len(rewards), 6) if rewards else None,
        "pool_mean_reward": round(sum(r.reward for r in tests) / len(tests), 6) if tests else None,
        "search_plan_mean_reward": round(sum(r.reward for r in plan) / len(plan), 6) if plan else None,
        "policy_pruned_mean_reward": round(sum(r.reward for r in pruned) / len(pruned), 6) if pruned else None,
        "by_oracle_mean_reward": {
            oracle: round(sum(r.reward for r in rows) / len(rows), 6)
            for oracle, rows in sorted(_group(tests, lambda r: r.labels.get("oracle")).items())
        },
    }


def _group(records: Iterable[TrajectoryRecord], key) -> dict[Any, list[TrajectoryRecord]]:
    out: dict[Any, list[TrajectoryRecord]] = defaultdict(list)
    for r in records:
        out[key(r)].append(r)
    return out


def dataset_stats(records: Sequence[TrajectoryRecord], *, sha256: str | None = None,
                  runs: int | None = None) -> dict[str, Any]:
    by_split = _group(records, lambda r: r.split)
    return {
        "dataset": "cubic-rlvr",
        "format": "jsonl; TRL conversational prompt/completion + reward",
        "reward_scheme": {o.value: v for o, v in REWARDS.items()},
        "runs_per_verification": runs,
        "rows": len(records),
        "sha256": sha256,
        "leakage": check_leakage(records),
        "splits": {split: _summary(rows) for split, rows in sorted(by_split.items())},
    }
