"""Closed-loop verification gate: a patch is VERIFIED only if it provably fixes the bug
without breaking anything that worked.

For one candidate patch the gate:

1. writes the patched source into an isolated temporary directory and imports it as
   a fresh, uniquely named module. The original module and ``sys.modules`` are
   never touched, and the copy is deleted when the gate finishes;
2. rebuilds the app from the patched module with the *same* build toggles as the
   build the finding was confirmed on;
3. replays the confirmed exploit trace with the Milestone 1 deterministic verifier.
   The invariant must hold on every run (0 violations out of ``runs``);
4. re-runs every functional QA test (success-oracle edges from the search plan)
   that passes on the unpatched build. Each one must still pass on the patched
   build.

A patch that does not import, does not build, still reproduces the exploit, or
regresses a functional edge is rejected, along with the reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import importlib.util
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from qbq.graph import StateGraph
from qbq.gym.base import AppSource
from qbq.remediate.models import (
    ConfirmedFinding,
    FixVerdict,
    FixVerification,
    FunctionalCheck,
    PatchCandidate,
)
from qbq.remediate.patcher import resolve_module_path, resolve_source_file
from qbq.verifier.replay import DeterministicVerifier, Verdict
from qbq.verifier.snapshot import DbSnapshot
from qbq.verifier.trace import CandidateFinding

AppFactory = Callable[[], Any]
Snapshotter = Callable[[Any], DbSnapshot]


@contextlib.contextmanager
def patched_module(patch: PatchCandidate, app_source: AppSource) -> Iterator[ModuleType]:
    """Yield the app module with ``patch`` applied, imported from a throwaway copy.

    If the patch targets the app module itself, only that file is copied and loaded
    under a unique name. If it targets another module of the application (e.g.
    ``shop/routers/orders.py`` while the factory lives in ``shop/main.py``), the whole
    top-level package is copied with the patch applied. Then, for the duration of the
    context only, that copy replaces the package in ``sys.modules``. The originals are
    restored afterwards, and no file outside the temp directory is touched.
    """
    original = resolve_module_path(app_source)
    if not original.as_posix().endswith(patch.file):
        with _package_overlay(patch, app_source) as module:
            yield module
        return
    digest = hashlib.sha256(patch.patched_source.encode()).hexdigest()[:12]
    name = f"qbq_patched_{digest}_{abs(hash(patch.finding_id)) % 10**8}"
    with tempfile.TemporaryDirectory(prefix="qbq-fix-") as tmp:
        path = Path(tmp) / original.name
        path.write_text(patch.patched_source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load patched copy of {app_source.module}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # needed while executing (dataclasses, pydantic)
        try:
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.modules.pop(name, None)


_OVERLAY_LOCK = threading.RLock()


@contextlib.contextmanager
def _package_overlay(patch: PatchCandidate, app_source: AppSource) -> Iterator[ModuleType]:
    resolve_source_file(patch.file, app_source)  # refuses files outside the application package
    top = app_source.module.split(".")[0]
    if top == "qbq":
        raise ImportError("refusing to overlay the qbq package itself")
    spec = importlib.util.find_spec(top)
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(f"{top!r} is not a package")
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    with _OVERLAY_LOCK, tempfile.TemporaryDirectory(prefix="qbq-fix-") as tmp:
        shutil.copytree(package_dir, Path(tmp) / top, ignore=shutil.ignore_patterns("__pycache__"))
        (Path(tmp) / patch.file).write_text(patch.patched_source, encoding="utf-8")
        owned = lambda name: name == top or name.startswith(top + ".")  # noqa: E731
        saved = {k: v for k, v in sys.modules.items() if owned(k)}
        for name in saved:
            del sys.modules[name]
        sys.path.insert(0, tmp)
        importlib.invalidate_caches()
        try:
            yield importlib.import_module(app_source.module)
        finally:
            for name in [k for k in sys.modules if owned(k)]:
                del sys.modules[name]
            sys.modules.update(saved)
            sys.path.remove(tmp)
            importlib.invalidate_caches()


async def _verdicts(
    graph: StateGraph, factory: AppFactory, snapshot: Snapshotter,
    tests: Sequence[CandidateFinding], runs: int,
) -> dict[str, Verdict]:
    verifier = DeterministicVerifier(graph, factory, snapshot, runs=runs)
    return {t.id: (await verifier.averify(t)).verdict for t in tests}


async def averify_fix(
    finding: ConfirmedFinding,
    patch: PatchCandidate,
    *,
    graph: StateGraph,
    app_source: AppSource,
    active_bugs: frozenset[str] = frozenset(),
    original_factory: AppFactory,
    original_snapshot: Snapshotter,
    functional_tests: Sequence[CandidateFinding] = (),
    runs: int = 3,
    baseline: dict[str, Verdict] | None = None,
) -> FixVerification:
    started = time.perf_counter()
    result = FixVerification(finding_id=finding.id, patcher=patch.patcher, rule=patch.rule,
                             verdict=FixVerdict.PATCH_INVALID, runs=runs)
    if baseline is None:
        baseline = await _verdicts(graph, original_factory, original_snapshot, functional_tests, runs)
    try:
        with patched_module(patch, app_source) as module:
            build = getattr(module, app_source.factory)
            snapshot = app_source.snapshotter(module)

            def factory() -> Any:
                return app_source.build(build, active_bugs)

            factory()  # fail fast if the patched app does not build
            exploit = await DeterministicVerifier(graph, factory, snapshot, runs=runs).averify(finding.candidate)
            result.exploit_verdict = exploit.verdict.value
            result.exploit_violations = sum(r.violated for r in exploit.runs)
            passing = [t for t in functional_tests if baseline.get(t.id) is Verdict.NOT_REPRODUCED]
            after = await _verdicts(graph, factory, snapshot, passing, runs)
    except Exception as exc:  # noqa: BLE001 - any import/build failure invalidates the patch
        result.error = f"{type(exc).__name__}: {exc}"
        result.duration_s = time.perf_counter() - started
        return result

    result.functional = [
        FunctionalCheck(test_id=t.id, invariant_id=t.invariant_id,
                        before=baseline[t.id].value, after=after[t.id].value)
        for t in passing
    ]
    if exploit.verdict is Verdict.INVALID or result.exploit_violations > 0:
        result.verdict = (FixVerdict.PATCH_INVALID if exploit.verdict is Verdict.INVALID
                          else FixVerdict.EXPLOIT_REPRODUCES)
        result.error = exploit.error
    elif result.regressions:
        result.verdict = FixVerdict.FUNCTIONAL_REGRESSION
    else:
        result.verdict = FixVerdict.VERIFIED
    result.duration_s = time.perf_counter() - started
    return result


def verify_fix(finding: ConfirmedFinding, patch: PatchCandidate, **kwargs: Any) -> FixVerification:
    return asyncio.run(averify_fix(finding, patch, **kwargs))
