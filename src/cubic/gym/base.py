"""Bug Gym environment abstraction: an app, its State Graph, and ground truth."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from cubic.graph import StateGraph
from cubic.verifier.snapshot import DbSnapshot
from cubic.verifier.trace import CandidateFinding


@dataclass(frozen=True)
class GroundTruthBug:
    """A deliberately planted defect and the invariant it violates."""

    id: str
    invariant_id: str
    title: str
    cwe: str | None = None


@dataclass(frozen=True)
class Variant:
    """A build of the app with a specific set of planted bugs enabled."""

    name: str
    active_bugs: frozenset[str]


@dataclass(frozen=True)
class AppSource:
    """Where an environment's application code lives, for patching and test export.

    ``module`` must expose ``factory`` returning a fresh ASGI app: ``factory(active_bugs)``
    for Bug Gym apps, or plain ``factory()`` for third-party apps
    (``accepts_flags=False``). ``snapshot(app)`` returns a ``DbSnapshot``. When
    ``snapshot`` is None, the app has no state hook and state-based checks see an
    empty database.
    """

    module: str
    factory: str = "create_app"
    snapshot: str | None = "snapshot"
    accepts_flags: bool = True

    def build(self, factory: Callable[..., Any], active_bugs: frozenset[str] = frozenset()) -> Any:
        return factory(active_bugs) if self.accepts_flags else factory()

    def factory_args(self, active_bugs: frozenset[str]) -> tuple:
        return (active_bugs,) if self.accepts_flags else ()

    def snapshotter(self, module: Any) -> Callable[[Any], DbSnapshot]:
        return getattr(module, self.snapshot) if self.snapshot else (lambda app: DbSnapshot())


@dataclass(frozen=True)
class Environment:
    name: str
    graph: StateGraph
    bugs: tuple[GroundTruthBug, ...]
    build_app: Callable[[frozenset[str]], Any]
    snapshot: Callable[[Any], DbSnapshot]
    baseline_candidates: Callable[[], tuple[CandidateFinding, ...]] = field(default=lambda: ())
    app_source: AppSource | None = None

    def __post_init__(self) -> None:
        ids = [b.id for b in self.bugs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.name}: duplicate bug ids")
        for bug in self.bugs:
            self.graph.invariant(bug.invariant_id)  # raises KeyError if unknown

    def bug(self, bug_id: str) -> GroundTruthBug:
        for bug in self.bugs:
            if bug.id == bug_id:
                return bug
        raise KeyError(f"unknown bug {bug_id!r}")

    def variants(self) -> tuple[Variant, ...]:
        """``patched`` (no bugs), one variant per bug, and ``vulnerable`` (all bugs)."""
        singles = tuple(Variant(f"only:{b.id}", frozenset({b.id})) for b in self.bugs)
        return (
            Variant("patched", frozenset()),
            *singles,
            Variant("vulnerable", frozenset(b.id for b in self.bugs)),
        )

    def app_factory(self, active_bugs: Iterable[str]) -> Callable[[], Any]:
        bugs = frozenset(active_bugs)
        unknown = bugs - {b.id for b in self.bugs}
        if unknown:
            raise KeyError(f"unknown bugs {sorted(unknown)}")
        return lambda: self.build_app(bugs)
