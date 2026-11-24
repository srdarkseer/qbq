"""Resolve a compile target (``pkg.mod:attr`` or ``path/to/app.py[:attr]``) to an app factory."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_DEFAULT_ATTRS = ("app", "application", "create_app", "app_factory")


class TargetError(ValueError):
    pass


def _is_asgi_app(obj: Any) -> bool:
    return hasattr(obj, "routes") or hasattr(obj, "router")


def _module_for_path(path: Path) -> Any:
    path = path.resolve()
    for entry in map(Path, sys.path):
        try:
            rel = path.relative_to(entry.resolve())
        except (ValueError, OSError):
            continue
        parts = rel.with_suffix("").parts
        if parts and all(p.isidentifier() for p in parts):
            name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            try:
                return importlib.import_module(name)
            except ImportError:
                continue
    spec = importlib.util.spec_from_file_location(f"cubic_target_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise TargetError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_cwd_importable() -> None:
    """Make ``pkg.module`` targets in the working directory importable from the console script."""
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def load_target(spec: str) -> Callable[[], Any]:
    """Return a zero-argument factory producing a fresh ASGI app.

    A module-level app instance is returned as-is on every call (shared state);
    a factory function such as ``create_app`` is called once per instance.
    """
    location, _, attr = spec.partition(":")
    looks_like_path = location.endswith(".py") or "/" in location or Path(location).exists()
    if looks_like_path:
        path = Path(location)
        if not path.is_file():
            raise TargetError(f"no such file: {location}")
        module = _module_for_path(path)
    else:
        ensure_cwd_importable()
        try:
            module = importlib.import_module(location)
        except ImportError as exc:
            raise TargetError(f"cannot import module {location!r}: {exc}") from exc

    names = (attr,) if attr else _DEFAULT_ATTRS
    for name in names:
        obj = getattr(module, name, None)
        if obj is None:
            continue
        if _is_asgi_app(obj):
            return lambda: obj
        if callable(obj):
            return obj
    raise TargetError(
        f"{spec!r}: no ASGI app or factory found (looked for {', '.join(names)})"
    )
