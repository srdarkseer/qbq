"""Collect the source files the static pass should read for an app.

The dynamic pass knows which modules define endpoints. Guards, schemas and
dependency chains often live elsewhere (``deps.py``, ``auth.py``, ``schemas.py``),
and router mounts live in the app factory module. This module walks the local
*import closure* of the endpoint modules and the app module, restricted to the
application's own package:

* the scope root of a module is its package, walking up past conventional
  layout folders (``routers``, ``api``, ``endpoints``, ``v1`` ...). So
  ``shop.routers.orders`` is scoped to ``shop.*``, and ``qbq.gym.fintech.app``
  to ``qbq.gym.fintech.*``;
* top-level single-file modules have no package and are read on their own;
* modules outside every scope root (the standard library, third-party
  packages, other parts of a monorepo) are never read.
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterable
from pathlib import Path

from qbq.compiler.dynamic import module_display_name

LAYOUT_FOLDERS = frozenset({
    "routers", "routes", "router", "api", "apis", "endpoints", "views", "controllers", "handlers",
    "resources", "v1", "v2", "v3", "http", "web",
})
MAX_FILES = 400


def scope_root(module: str) -> str:
    parts = module.split(".")[:-1]
    while parts and parts[-1] in LAYOUT_FOLDERS:
        parts.pop()
    return ".".join(parts)


def _in_scope(module: str, roots: set[str]) -> bool:
    return any(root and (module == root or module.startswith(root + ".")) for root in roots)


def _module_path(module: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError, AttributeError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin).resolve()


def _imported_modules(module: str, path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join([*parts, *([node.module] if node.module else [])])
            if base:
                out.add(base)
                out |= {f"{base}.{a.name}" for a in node.names}  # `from pkg import submodule`
    return out


def collect_sources(
    endpoint_files: Iterable[tuple[str, str]], app_module: str | None = None
) -> list[tuple[str, Path]]:
    """``(display_name, path)`` for every endpoint module plus its in-scope import closure."""
    seeds: dict[str, Path] = {}
    for display, path in endpoint_files:
        module = display.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        seeds[module] = Path(path)
    if app_module and (app_path := _module_path(app_module)) is not None:
        seeds.setdefault(app_module, app_path)
    roots = {scope_root(m) for m in seeds}
    found: dict[str, Path] = dict(seeds)
    queue = list(seeds)
    while queue and len(found) < MAX_FILES:
        module = queue.pop()
        for imported in sorted(_imported_modules(module, found[module])):
            if imported in found or not _in_scope(imported, roots):
                continue
            if (path := _module_path(imported)) is not None:
                found[imported] = path
                queue.append(imported)
    out = []
    for module, path in sorted(found.items()):
        display = module_display_name(module, str(path)) or path.name
        out.append((display, path))
    return out
