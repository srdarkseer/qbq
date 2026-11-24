"""Third-party target adapter: turn any FastAPI/ASGI app into a Cubic environment.

``external_environment("pkg.module:create_app", personas=[...])`` produces an
``Environment`` that ``cubic eval``-style search, ``cubic remediate`` and the RLVR
recorder can drive, exactly like a Bug Gym app:

* the app factory is called with no arguments (``accepts_flags=False``) and must
  return a *fresh* app with freshly seeded state on every call. The verifier
  rebuilds the app for every replay;
* personas come from a JSON list of ``ActorPersona`` objects with the seeded test
  credentials and the resources each one owns, or from a StateGraph JSON's
  ``actors``;
* an optional ``module:function`` snapshot hook returns a ``DbSnapshot``. It enables
  ledger and state-guard invariants. Without it, only HTTP-level invariants are checked;
* there is no ground truth, so the environment has no planted bugs. Its
  ``vulnerable`` variant is simply the app as it is.

Patching needs the target to be an importable module (``pkg.module:factory``),
because the fix gate imports a patched copy of that module's source.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cubic.compiler.merge import compile_app
from cubic.compiler.target import TargetError, ensure_cwd_importable
from cubic.graph import ActorPersona
from cubic.gym.base import AppSource, Environment
from cubic.verifier.snapshot import DbSnapshot


def load_personas(path: Path | str) -> list[ActorPersona]:
    data = json.loads(Path(path).read_text())
    items = data.get("actors", []) if isinstance(data, dict) else data
    return [ActorPersona.model_validate(item) for item in items]


def _split(spec: str, what: str) -> tuple[str, str]:
    module, sep, attr = spec.partition(":")
    if not sep or not module or not attr or module.endswith(".py") or "/" in module:
        raise TargetError(f"{what} must be an importable 'pkg.module:attribute' (got {spec!r})")
    return module, attr


def external_environment(
    app: str,
    personas: Iterable[ActorPersona] = (),
    *,
    snapshot: str | None = None,
    name: str | None = None,
    app_name: str | None = None,
) -> Environment:
    ensure_cwd_importable()
    module_name, factory_attr = _split(app, "--app")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise TargetError(f"cannot import {module_name!r}: {exc}") from exc
    factory = getattr(module, factory_attr, None)
    if not callable(factory) or hasattr(factory, "routes"):
        raise TargetError(f"{app!r} must name an app *factory* (a function returning a fresh app)")

    snap_fn: Any
    if snapshot is not None:
        snap_module, snap_attr = _split(snapshot, "--snapshot")
        snap_fn = getattr(importlib.import_module(snap_module), snap_attr)
        snap_name = snap_attr if snap_module == module_name else None
        if snap_name is None:
            raise TargetError("--snapshot must live in the same module as the app factory")
    else:
        snap_fn, snap_name = (lambda app_: DbSnapshot()), None

    graph = compile_app(lambda: factory(), personas=list(personas), app_name=app_name).graph
    return Environment(
        name=name or f"external:{module_name}",
        graph=graph,
        bugs=(),
        build_app=lambda active: factory(),
        snapshot=snap_fn,
        app_source=AppSource(module=module_name, factory=factory_attr, snapshot=snap_name, accepts_flags=False),
    )
