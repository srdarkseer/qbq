"""Static pass: extract routes, guards, models and state effects from Python source.

Uses only the standard-library ``ast`` module. The analyzer understands the
FastAPI idioms used by typical services:

* route decorators ``@<app|router>.<get|post|put|patch|delete>(path, status_code=...)``,
  including handlers nested inside an app factory;
* multi-file projects: ``APIRouter(prefix=..., dependencies=[...])`` objects mounted with
  ``include_router(router, prefix=..., dependencies=[...])`` (nested mounts too), with
  imports resolved across modules (``from .deps import CurrentUser``,
  ``from app.routers import orders``);
* dependencies via ``Annotated[..., Depends(fn)]`` aliases (also imported from another
  module), inline ``Annotated`` annotations, ``= Depends(fn)`` defaults, decorator and
  router ``dependencies=[...]``, transitive chains (``require_admin -> get_current_user``),
  and dependency factories such as ``Depends(require_role("ADMIN"))``. A route is
  authenticated if any dependency in its chain is; guards inside dependencies (ownership,
  roles) count as the route's own guards;
* guard clauses ``if <cond>: raise HTTPException(<status>)`` (role checks,
  ownership checks, lifecycle-state preconditions), followed one level into
  same-file helper functions such as ``owned_wallet(...)``;
* state effects: constant writes to ``status``/``state`` fields, dict literals that
  create rows in a given state, mutated fields, ledger writes, and lock usage;
* Pydantic request models (``class X(BaseModel)``, subclasses of other models, v2
  ``Annotated[int, Field(gt=0)]`` fields) defined in any analyzed module, and store
  collections (``dict[str, dict]`` entities, ``list[dict]`` ledgers).

``or``-guards (``if row is None or row["owner_id"] != user["id"]: raise ...``) are read as
one guard per disjunct.

Conjuncts of the form ``"flag" in/not in <name>`` are treated as feature flags and
ignored when classifying a guard; the analyzer describes the hardened code path.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qbq.graph import (
    HttpMethod,
    NodeKind,
    StateNode,
    TransitionEdge,
)

HTTP_METHODS = {m.value.lower(): m for m in HttpMethod}
LIFECYCLE_FIELDS = frozenset({"status", "state"})
ROLE_FIELDS = frozenset({"role", "roles"})
_PATH_PARAM = re.compile(r"\{([^{}:]+)(?::[^{}]*)?\}")
_JSON_TYPES = {"int": "integer", "float": "number", "str": "string", "bool": "boolean"}
_CONSTRAINTS = {
    "gt": "exclusiveMinimum", "ge": "minimum", "lt": "exclusiveMaximum", "le": "maximum",
    "min_length": "minLength", "max_length": "maxLength", "pattern": "pattern",
}


# --------------------------------------------------------------------------- facts


@dataclass(frozen=True)
class SourceLocation:
    file: str
    line: int
    function: str

    def as_dict(self) -> dict[str, Any]:
        return {"file": self.file, "function": self.function, "line": self.line}


@dataclass(frozen=True)
class ModelField:
    name: str
    json_type: str | None
    required: bool
    constraints: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class RequestModel:
    name: str
    fields: tuple[ModelField, ...]
    location: SourceLocation

    def json_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for f in self.fields:
            prop: dict[str, Any] = {} if f.json_type is None else {"type": f.json_type}
            prop.update(dict(f.constraints))
            props[f.name] = prop
        schema: dict[str, Any] = {"type": "object", "properties": props}
        required = [f.name for f in self.fields if f.required]
        if required:
            schema["required"] = required
        return schema


@dataclass(frozen=True)
class StateRequirement:
    field: str
    value: str
    status: int | None


@dataclass(frozen=True)
class LedgerSpec:
    """How the application's append-only ledger is shaped."""

    collection: str
    account_field: str
    amount_field: str
    entity: str | None = None
    balance_field: str | None = None


@dataclass(frozen=True)
class StaticRoute:
    method: HttpMethod
    path: str
    location: SourceLocation
    decorator_line: int
    status_code: int | None = None
    auth_dependency: str | None = None
    body_model: str | None = None
    role_guards: frozenset[str] = frozenset()
    ownership_guard: bool = False
    raised_statuses: frozenset[int] = frozenset()
    state_requirements: tuple[StateRequirement, ...] = ()
    preconditions: tuple[StateRequirement, ...] = ()
    state_writes: tuple[str, ...] = ()
    created_states: tuple[str, ...] = ()
    mutated_fields: frozenset[str] = frozenset()
    collections: frozenset[str] = frozenset()
    writes_ledger: bool = False
    uses_lock: bool = False
    feature_flags: frozenset[str] = frozenset()

    @property
    def path_params(self) -> tuple[str, ...]:
        return tuple(_PATH_PARAM.findall(self.path))

    @property
    def required_state(self) -> str | None:
        return self.state_requirements[0].value if len(self.state_requirements) == 1 else None

    @property
    def produced_state(self) -> str | None:
        if self.method is HttpMethod.GET:  # safe method: never a state transition
            return None
        if self.state_writes:
            return self.state_writes[-1]
        return self.created_states[-1] if self.created_states else None


@dataclass(frozen=True)
class StaticAnalysis:
    files: tuple[str, ...]
    source_bytes: int
    routes: tuple[StaticRoute, ...]
    models: tuple[RequestModel, ...]
    roles: frozenset[str]
    entity_collections: frozenset[str]
    ledger: LedgerSpec | None
    dependency_statuses: dict[str, frozenset[int]] = field(default_factory=dict)

    def model(self, name: str) -> RequestModel | None:
        return next((m for m in self.models if m.name == name), None)


# --------------------------------------------------------------------------- helpers


def _const_str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _const_int(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _key_of(node: ast.AST) -> str | None:
    """Field name accessed by ``x["k"]`` or ``x.k``."""
    if isinstance(node, ast.Subscript):
        return _const_str(node.slice)
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return None


def _http_exception_status(node: ast.AST) -> int | None:
    if not isinstance(node, ast.Raise) or _call_name(node.exc) != "HTTPException":
        return None
    call = node.exc
    assert isinstance(call, ast.Call)
    if call.args and (code := _const_int(call.args[0])) is not None:
        return code
    for kw in call.keywords:
        if kw.arg == "status_code":
            return _const_int(kw.value)
    return None


def _conjuncts(test: ast.AST) -> list[ast.AST]:
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return [c for v in test.values for c in _conjuncts(v)]
    return [test]


def _disjuncts(test: ast.AST) -> list[ast.AST]:
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        return [d for v in test.values for d in _disjuncts(v)]
    return [test]


def _depends_expr(node: ast.AST | None) -> ast.AST | None:
    """The argument of the first ``Depends(<x>)`` call inside ``node``."""
    if node is None:
        return None
    for sub in ast.walk(node):
        if _call_name(sub) == "Depends" and isinstance(sub, ast.Call) and sub.args:
            return sub.args[0]
    return None


def _module_name(display: str) -> str:
    name = display.removesuffix(".py").replace("/", ".")
    return name.removesuffix(".__init__")


def _feature_flag(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Compare) and len(node.ops) == 1
        and isinstance(node.ops[0], ast.In | ast.NotIn)
        and isinstance(node.comparators[0], ast.Name)
    ):
        return _const_str(node.left)
    return None


def _not_eq_const(node: ast.AST) -> tuple[str, str] | None:
    """Match ``x[field] != "CONST"`` -> (field, CONST)."""
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq):
        key, value = _key_of(node.left), _const_str(node.comparators[0])
        if key and value is not None:
            return key, value
    return None


def _is_ownership_compare(node: ast.AST) -> bool:
    """Match ``row["<x>_id"] != user["id"]`` (either side)."""
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq)):
        return False
    keys = {_key_of(node.left), _key_of(node.comparators[0])}
    return "id" in keys and any(k and k != "id" and k.endswith("_id") for k in keys)


def _guard_raise(stmt: ast.If) -> int | None:
    for inner in stmt.body:
        if (code := _http_exception_status(inner)) is not None:
            return code
    return None


def _dotted(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive
        return ""


def _walk_shallow(nodes: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """Walk statements without descending into nested function/class definitions."""
    stack = list(nodes)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
                stack.append(child)


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 1:
        return word[:-1]
    return word


def _plural(word: str) -> str:
    if word.endswith("y"):
        return word[:-1] + "ies"
    return word if word.endswith("s") else word + "s"


# --------------------------------------------------------------------------- analyzer


FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _field_constraints(call: ast.Call) -> list[tuple[str, Any]]:
    return [(_CONSTRAINTS[kw.arg], kw.value.value) for kw in call.keywords
            if kw.arg in _CONSTRAINTS and isinstance(kw.value, ast.Constant)]


def _model_field(stmt: ast.AnnAssign) -> ModelField:
    """One model field: ``x: int``, ``x: int = Field(gt=0)`` or ``x: Annotated[int, Field(gt=0)]``."""
    assert isinstance(stmt.target, ast.Name)
    annotation = stmt.annotation
    constraints: list[tuple[str, Any]] = []
    if (isinstance(annotation, ast.Subscript) and _dotted(annotation.value).split(".")[-1] == "Annotated"
            and isinstance(annotation.slice, ast.Tuple) and annotation.slice.elts):
        annotation, *metadata = annotation.slice.elts
        for meta in metadata:
            if isinstance(meta, ast.Call) and _call_name(meta) == "Field":
                constraints += _field_constraints(meta)
    json_type = _JSON_TYPES.get(_dotted(annotation))
    required = stmt.value is None
    if isinstance(stmt.value, ast.Call) and _call_name(stmt.value) == "Field":
        has_default = bool(stmt.value.args) and not (
            isinstance(stmt.value.args[0], ast.Constant) and stmt.value.args[0].value is Ellipsis
        )
        has_default |= any(k.arg in {"default", "default_factory"} for k in stmt.value.keywords)
        required = not has_default
        constraints += _field_constraints(stmt.value)
    return ModelField(stmt.target.id, json_type, required, tuple(sorted(constraints)))


@dataclass(frozen=True)
class DepRef:
    """A resolved dependency: function ``function`` in ``file``, called with ``args`` if a factory."""

    file: str
    function: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class DepInfo:
    auth: bool = False
    statuses: frozenset[int] = frozenset()
    roles: frozenset[str] = frozenset()
    bodies: tuple[ast.AST, ...] = ()


class _Project:
    """Cross-module symbol table over every analyzed file."""

    _TABLES = {"function": "functions", "class": "classes", "alias": "aliases", "router": "routers"}

    def __init__(self, analyzers: Iterable[_FileAnalyzer]) -> None:
        self.analyzers = list(analyzers)
        self.by_file = {a.file: a for a in self.analyzers}
        self.by_module = {a.module: a for a in self.analyzers}
        self._deps: dict[DepRef, DepInfo] = {}
        self._in_progress: set[DepRef] = set()
        for a in self.analyzers:
            a.project = self
        self.mounts: dict[tuple[str, str], tuple[str, tuple[DepRef, ...], tuple[str, str] | None]] = {}
        for a in self.analyzers:
            self._collect_mounts(a)

    # -- symbol resolution ------------------------------------------------------

    def resolve(self, an: _FileAnalyzer, name: str, kind: str, depth: int = 0) -> tuple[_FileAnalyzer, str, Any] | None:
        table = getattr(an, self._TABLES[kind])
        if name in table:
            return an, name, table[name]
        imp = an.imports.get(name)
        if imp is None or imp[1] is None or depth > 8:
            return None
        target = self.by_module.get(imp[0])
        return self.resolve(target, imp[1], kind, depth + 1) if target else None

    def module_of(self, an: _FileAnalyzer, expr: ast.AST) -> _FileAnalyzer | None:
        if isinstance(expr, ast.Name) and (imp := an.imports.get(expr.id)):
            mod, sym = imp
            return self.by_module.get(mod if sym is None else f"{mod}.{sym}")
        return None

    def dep_ref(self, an: _FileAnalyzer, expr: ast.AST | None) -> DepRef | None:
        if expr is None:
            return None
        args: tuple[str, ...] = ()
        if isinstance(expr, ast.Call):
            args = tuple(v for a in expr.args if (v := _const_str(a)) is not None)
            expr = expr.func
        found = None
        if isinstance(expr, ast.Name):
            found = self.resolve(an, expr.id, "function")
        elif isinstance(expr, ast.Attribute) and (mod := self.module_of(an, expr.value)) is not None:
            found = self.resolve(mod, expr.attr, "function")
        return DepRef(found[0].file, found[1], args) if found else None

    def alias_ref(self, an: _FileAnalyzer, name: str) -> DepRef | None:
        found = self.resolve(an, name, "alias")
        return self.dep_ref(found[0], found[2]) if found else None

    # -- dependency semantics ---------------------------------------------------

    def dep_info(self, ref: DepRef) -> DepInfo:
        if ref in self._deps:
            return self._deps[ref]
        if ref in self._in_progress:
            return DepInfo()
        self._in_progress.add(ref)
        an = self.by_file[ref.file]
        fn = an.functions[ref.function]
        inner = [n for n in fn.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]  # factory closures
        funcs = [fn, *inner]
        statuses = {c for f in funcs for n in _walk_shallow(f.body) if (c := _http_exception_status(n)) is not None}
        arg_names = {a.arg.lower() for f in funcs for a in f.args.args + f.args.kwonlyargs}
        subs = [d for f in funcs for d in an.param_dependencies(f) if d != ref]
        sub_infos = [self.dep_info(d) for d in subs]
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs} | ({fn.args.vararg.arg} if fn.args.vararg else set())
        roles: set[str] = set()
        if ref.args:
            for f in inner:
                for node in _walk_shallow(f.body):
                    if isinstance(node, ast.If) and _guard_raise(node) is not None and _compares_role_to(node.test, params):
                        roles |= set(ref.args)
        bodies = [*an.reachable_bodies(fn), *(b for f in inner for b in an.reachable_bodies(f))]
        info = DepInfo(
            auth=401 in statuses or "authorization" in arg_names or any(i.auth for i in sub_infos),
            statuses=frozenset(statuses | {c for i in sub_infos for c in i.statuses}),
            roles=frozenset(roles | {r for i in sub_infos for r in i.roles}),
            bodies=tuple(bodies) + tuple(b for i in sub_infos for b in i.bodies),
        )
        self._in_progress.discard(ref)
        self._deps[ref] = info
        return info

    # -- router mounts ----------------------------------------------------------

    def _collect_mounts(self, an: _FileAnalyzer) -> None:
        for node in ast.walk(an.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "include_router" and node.args):
                continue
            target = node.args[0]
            key: tuple[str, str] | None = None
            if isinstance(target, ast.Attribute) and (mod := self.module_of(an, target.value)) is not None:
                if target.attr in mod.routers:
                    key = (mod.file, target.attr)
            elif isinstance(target, ast.Name) and (found := self.resolve(an, target.id, "router")):
                key = (found[0].file, found[1])
            if key is None or key in self.mounts:
                continue
            prefix = next((_string_value(an, k.value) or "" for k in node.keywords if k.arg == "prefix"), "")
            deps = tuple(d for k in node.keywords if k.arg == "dependencies"
                         for d in an.dependency_list(k.value))
            receiver = node.func.value
            parent = (an.file, receiver.id) if isinstance(receiver, ast.Name) and receiver.id in an.routers else None
            self.mounts[key] = (prefix, deps, parent)

    def router_context(self, an: _FileAnalyzer, var: str, depth: int = 0) -> tuple[str, tuple[DepRef, ...]]:
        """Full path prefix and inherited dependencies of router ``var`` defined in ``an``."""
        own_prefix, own_deps = an.routers[var]
        deps = tuple(d for e in own_deps if (d := self.dep_ref(an, e)) is not None)
        mount = self.mounts.get((an.file, var))
        if mount is None or depth > 8:
            return own_prefix, deps
        prefix, mount_deps, parent = mount
        outer_prefix, outer_deps = ("", ())
        if parent is not None:
            outer_prefix, outer_deps = self.router_context(self.by_file[parent[0]], parent[1], depth + 1)
        return outer_prefix + prefix + own_prefix, outer_deps + mount_deps + deps


def _string_value(an: _FileAnalyzer, node: ast.AST | None, depth: int = 0) -> str | None:
    """A literal string, a name bound once in the file to such a value, or a ``+`` of those."""
    if node is None or depth > 8:
        return None
    if (value := _const_str(node)) is not None:
        return value
    if isinstance(node, ast.Name):
        bound = an.assigned.get(node.id, [])
        return _string_value(an, bound[0], depth + 1) if len(bound) == 1 else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _string_value(an, node.left, depth + 1), _string_value(an, node.right, depth + 1)
        return left + right if left is not None and right is not None else None
    return None


def _compares_role_to(test: ast.AST, params: set[str]) -> bool:
    """``user["role"] not in roles`` / ``user.role != role`` where the right side is a factory parameter."""
    for node in ast.walk(test):
        if (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], ast.NotIn | ast.NotEq)
                and _key_of(node.left) in ROLE_FIELDS
                and isinstance(node.comparators[0], ast.Name) and node.comparators[0].id in params):
            return True
    return False


def _router_call(value: ast.AST) -> tuple[str, list[ast.AST]] | None:
    if _call_name(value) != "APIRouter" or not isinstance(value, ast.Call):
        return None
    prefix = next((_const_str(k.value) or "" for k in value.keywords if k.arg == "prefix"), "")
    deps = [e for k in value.keywords if k.arg == "dependencies" and isinstance(k.value, ast.List | ast.Tuple)
            for elt in k.value.elts if (e := _depends_expr(elt)) is not None]
    return prefix, deps


class _FileAnalyzer:
    def __init__(self, tree: ast.Module, display_name: str) -> None:
        self.tree = tree
        self.file = display_name
        self.module = _module_name(display_name)
        self.package = self.module if display_name.endswith("__init__.py") else self.module.rpartition(".")[0]
        self.functions: dict[str, FunctionNode] = {}  # name -> first definition (helper lookup)
        self.all_functions: list[FunctionNode] = []  # every definition, including duplicates
        self.aliases: dict[str, ast.AST] = {}  # alias name -> Depends(<target>) expression
        self.classes: dict[str, ast.ClassDef] = {}
        self.routers: dict[str, tuple[str, list[ast.AST]]] = {}  # var -> (prefix, dependency exprs)
        self.imports: dict[str, tuple[str, str | None]] = {}  # local name -> (module, symbol or None)
        self.assigned: dict[str, list[ast.AST]] = {}  # name -> every value bound to it (constant prefixes)
        self.project: _Project | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                self.functions.setdefault(node.name, node)
                self.all_functions.append(node)
            elif isinstance(node, ast.ClassDef):
                self.classes.setdefault(node.name, node)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.assigned.setdefault(node.targets[0].id, []).append(node.value)
                if (router := _router_call(node.value)) is not None:
                    self.routers[node.targets[0].id] = router
                elif (dep := _depends_expr(node.value)) is not None:
                    self.aliases[node.targets[0].id] = dep
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = self.package.split(".") if self.package else []
                    parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                    base = ".".join([*parts, *([node.module] if node.module else [])])
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = (base, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    self.imports[local] = (alias.name if alias.asname else alias.name.split(".")[0], None)

    @property
    def proj(self) -> _Project:
        if self.project is None:
            _Project([self])
        assert self.project is not None
        return self.project

    # -- dependencies -------------------------------------------------------

    def param_dependencies(self, fn: FunctionNode) -> list[DepRef]:
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        defaults: list[ast.AST | None] = [None] * (len(fn.args.posonlyargs + fn.args.args) - len(fn.args.defaults))
        defaults += list(fn.args.defaults) + list(fn.args.kw_defaults)
        deps: list[DepRef] = []
        for arg, default in zip(args, defaults, strict=True):
            ann = arg.annotation
            ref = None
            if isinstance(ann, ast.Name):
                ref = self.proj.alias_ref(self, ann.id)
            if ref is None:
                ref = self.proj.dep_ref(self, _depends_expr(ann) or _depends_expr(default))
            if ref is not None:
                deps.append(ref)
        return deps

    def dependency_list(self, node: ast.AST) -> list[DepRef]:
        """Resolve a ``dependencies=[Depends(a), Depends(b("x"))]`` keyword value."""
        if not isinstance(node, ast.List | ast.Tuple):
            return []
        return [r for elt in node.elts if (r := self.proj.dep_ref(self, _depends_expr(elt))) is not None]

    def dependency_statuses(self, name: str) -> frozenset[int]:
        fn = self.functions.get(name)
        if fn is None:
            return frozenset()
        return frozenset(c for n in _walk_shallow(fn.body) if (c := _http_exception_status(n)) is not None)

    # -- helpers called from handlers -----------------------------------------

    def reachable_bodies(self, fn: FunctionNode, depth: int = 3) -> list[ast.AST]:
        """``fn``'s body plus the bodies of helpers it calls (followed across imports)."""
        alias_targets = {e.id for e in self.aliases.values() if isinstance(e, ast.Name)}
        seen = {(self.file, fn.name)}
        bodies: list[ast.AST] = list(fn.body)
        frontier: list[tuple[_FileAnalyzer, FunctionNode]] = [(self, fn)]
        for _ in range(depth):
            nxt: list[tuple[_FileAnalyzer, FunctionNode]] = []
            for an, f in frontier:
                for node in _walk_shallow(f.body):
                    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                        continue
                    found = an.proj.resolve(an, node.func.id, "function")
                    if found is None:
                        continue
                    target_an, name, callee = found
                    if (target_an.file, name) in seen or (target_an is self and name in alias_targets):
                        continue
                    seen.add((target_an.file, name))
                    bodies.extend(callee.body)
                    nxt.append((target_an, callee))
            frontier = nxt
        return bodies

    # -- class facts (store collections, ledger) -------------------------------

    def collections(self) -> tuple[set[str], set[str], dict[str, set[str]]]:
        """Return (entity collections, list collections, method -> self attrs touched)."""
        entities: set[str] = set()
        lists: set[str] = set()
        method_attrs: dict[str, set[str]] = {}
        for cls in self.classes.values():
            for stmt in cls.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    ann = stmt.annotation
                    if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
                        base, inner = ann.value.id.lower(), _dotted(ann.slice)
                        if base in {"dict", "defaultdict"} and re.search(r",\s*dict\b", inner):
                            entities.add(stmt.target.id)
                        elif base == "list" and inner.startswith("dict"):
                            lists.add(stmt.target.id)
                elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                    attrs = {
                        n.attr for n in ast.walk(stmt)
                        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
                    }
                    method_attrs[stmt.name] = attrs
        return entities, lists, method_attrs

    def ledger_spec(self, entities: set[str], lists: set[str]) -> LedgerSpec | None:
        for cls in self.classes.values():
            for node in ast.walk(cls):
                if (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append" and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr in lists and node.args and isinstance(node.args[0], ast.Dict)
                ):
                    keys = [k for k in (_const_str(k) for k in node.args[0].keys) if k]
                    account = next((k for k in keys if k.endswith("_id")), None)
                    amount = next((k for k in keys if k in {"amount", "value", "delta"}), None)
                    if account and amount:
                        entity = _plural(account.removesuffix("_id"))
                        return LedgerSpec(
                            collection=node.func.value.attr, account_field=account, amount_field=amount,
                            entity=entity if entity in entities else None,
                        )
        return None

    # -- models ---------------------------------------------------------------

    def models(self, model_names: set[str] | None = None) -> list[RequestModel]:
        names = model_names if model_names is not None else {
            c.name for c in self.classes.values()
            if any(_dotted(b).split(".")[-1] == "BaseModel" for b in c.bases)
        }
        out: list[RequestModel] = []
        for cls in self.classes.values():
            if cls.name not in names:
                continue
            fields: dict[str, ModelField] = {}
            for base in cls.bases:  # inherited fields first, then own fields override
                found = self.proj.resolve(self, _dotted(base).split(".")[-1], "class")
                if found and found[1] in names and found[2] is not cls:
                    for parent in found[0].models(names):
                        if parent.name == found[1]:
                            fields.update({f.name: f for f in parent.fields})
            for stmt in cls.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields[stmt.target.id] = _model_field(stmt)
            out.append(RequestModel(cls.name, tuple(fields.values()), SourceLocation(self.file, cls.lineno, cls.name)))
        return out

    def roles(self) -> set[str]:
        roles: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values, strict=True):
                    if _const_str(k) in ROLE_FIELDS and (r := _const_str(v)):
                        roles.add(r)
            elif isinstance(node, ast.Compare) and _key_of(node.left) in ROLE_FIELDS:
                roles.update(r for c in node.comparators if (r := _const_str(c)))
        return roles

    # -- routes ---------------------------------------------------------------

    def routes(self, ledger_methods: set[str], lock_hint: str = "lock") -> list[StaticRoute]:
        routes: list[StaticRoute] = []
        for fn in self.all_functions:
            for dec in fn.decorator_list:
                if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
                    continue
                method = HTTP_METHODS.get(dec.func.attr)
                path = _const_str(dec.args[0]) if dec.args else None
                if method is None or path is None:
                    continue
                status = next((_const_int(k.value) for k in dec.keywords if k.arg == "status_code"), None)
                prefix, extra = "", ()
                receiver = dec.func.value
                if isinstance(receiver, ast.Name) and receiver.id in self.routers:
                    prefix, extra = self.proj.router_context(self, receiver.id)
                extra = tuple(extra) + tuple(
                    d for k in dec.keywords if k.arg == "dependencies" for d in self.dependency_list(k.value))
                routes.append(self._route(fn, method, prefix + path, status, dec.lineno, ledger_methods,
                                          lock_hint, extra))
        return routes

    def _route(
        self, fn: FunctionNode, method: HttpMethod, path: str, status: int | None,
        decorator_line: int, ledger_methods: set[str], lock_hint: str,
        extra_deps: tuple[DepRef, ...] = (),
    ) -> StaticRoute:
        deps = [*extra_deps, *self.param_dependencies(fn)]
        infos = [self.proj.dep_info(d) for d in deps]
        auth = next((d.function for d, i in zip(deps, infos, strict=True) if i.auth), None)
        body_model = next(
            (a.annotation.id for a in fn.args.args
             if isinstance(a.annotation, ast.Name) and self.proj.resolve(self, a.annotation.id, "class")),
            None,
        )
        bodies = self.reachable_bodies(fn) + [b for i in infos for b in i.bodies]
        roles: set[str] = {r for i in infos for r in i.roles}
        flags: set[str] = set()
        ownership = False
        raised: set[int] = set()
        requirements: list[StateRequirement] = []
        preconditions: list[StateRequirement] = []
        writes: list[str] = []
        created: list[str] = []
        mutated: set[str] = set()
        collections: set[str] = set()
        ledger = uses_lock = False

        for node in _walk_shallow(bodies):
            if (code := _http_exception_status(node)) is not None:
                raised.add(code)
            if isinstance(node, ast.If) and (code := _guard_raise(node)) is not None:
                for test in _disjuncts(node.test):
                    conj = _conjuncts(test)
                    flags.update(f for c in conj if (f := _feature_flag(c)))
                    rest = [c for c in conj if _feature_flag(c) is None]
                    if any(_is_ownership_compare(c) for c in rest):
                        ownership = True
                    elif len(rest) == 1 and (match := _not_eq_const(rest[0])):
                        key, value = match
                        if key in ROLE_FIELDS:
                            roles.add(value)
                        elif key in LIFECYCLE_FIELDS:
                            requirements.append(StateRequirement(key, value, code))
                        else:
                            preconditions.append(StateRequirement(key, value, code))
            elif isinstance(node, ast.If):
                flags.update(f for c in _conjuncts(node.test) if (f := _feature_flag(c)))
            if isinstance(node, ast.Assign | ast.AugAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Subscript | ast.Attribute) and (key := _key_of(t)):
                        mutated.add(key)
                        if key in LIFECYCLE_FIELDS and (v := _const_str(node.value)):
                            writes.append(v)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "update":
                    for kw in node.keywords:
                        if kw.arg:
                            mutated.add(kw.arg)
                            if kw.arg in LIFECYCLE_FIELDS and (v := _const_str(kw.value)):
                                writes.append(v)
                if node.func.attr in ledger_methods:
                    ledger = True
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values, strict=True):
                    if _const_str(k) in LIFECYCLE_FIELDS and (s := _const_str(v)):
                        created.append(s)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                collections.add(node.attr)
            if isinstance(node, ast.AsyncWith | ast.With):
                uses_lock |= any(lock_hint in _dotted(item.context_expr).lower() for item in node.items)

        raised |= {c for i in infos for c in i.statuses}
        return StaticRoute(
            method=method, path=path,
            location=SourceLocation(self.file, fn.lineno, fn.name), decorator_line=decorator_line,
            status_code=status, auth_dependency=auth, body_model=body_model,
            role_guards=frozenset(roles), ownership_guard=ownership, raised_statuses=frozenset(raised),
            state_requirements=tuple(requirements), preconditions=tuple(preconditions),
            state_writes=tuple(writes), created_states=tuple(created),
            mutated_fields=frozenset(mutated - {"id"}),
            collections=frozenset(collections), writes_ledger=ledger, uses_lock=uses_lock,
            feature_flags=frozenset(flags),
        )


def analyze_sources(sources: Iterable[tuple[str, str]]) -> StaticAnalysis:
    """Analyze ``(display_name, source_text)`` pairs as one project.

    Display names are package-relative paths (``app/routers/orders.py``). They define
    module names for resolving imports between the analyzed files.
    """
    files: list[str] = []
    total = 0
    analyzers: list[_FileAnalyzer] = []
    for name, text in sorted(sources):
        files.append(name)
        total += len(text.encode())
        analyzers.append(_FileAnalyzer(ast.parse(text, filename=name), name))
    _Project(analyzers)

    entities: set[str] = set()
    ledger: LedgerSpec | None = None
    method_attrs: dict[str, set[str]] = {}
    for analyzer in analyzers:
        file_entities, lists, attrs = analyzer.collections()
        entities |= file_entities
        for method, touched in attrs.items():
            method_attrs.setdefault(method, set()).update(touched)
        ledger = ledger or analyzer.ledger_spec(file_entities, lists)
    ledger_methods = {m for m, attrs in method_attrs.items() if ledger and ledger.collection in attrs}

    routes: list[StaticRoute] = [r for a in analyzers for r in a.routes(ledger_methods)]
    if ledger and ledger.entity:
        balance = sorted({
            f for r in routes if r.writes_ledger for f in r.mutated_fields
            if f not in {"id", ledger.account_field, ledger.amount_field} | LIFECYCLE_FIELDS
        })
        if len(balance) == 1:
            ledger = LedgerSpec(ledger.collection, ledger.account_field, ledger.amount_field, ledger.entity, balance[0])

    model_names = _model_class_names(analyzers)
    models: list[RequestModel] = [m for a in analyzers for m in a.models(model_names)]
    roles: set[str] = set().union(*(a.roles() for a in analyzers)) if analyzers else set()
    dep_statuses: dict[str, frozenset[int]] = {}
    for analyzer in analyzers:
        for alias in sorted(analyzer.aliases):
            ref = analyzer.proj.alias_ref(analyzer, alias)
            if ref is not None:
                dep_statuses.setdefault(ref.function, analyzer.proj.dep_info(ref).statuses)

    routes.sort(key=lambda r: (r.path, r.method.value, r.location.file, r.location.line))
    models.sort(key=lambda m: (m.name, m.location.file))
    return StaticAnalysis(
        files=tuple(files), source_bytes=total, routes=tuple(routes), models=tuple(models),
        roles=frozenset(roles), entity_collections=frozenset(entities), ledger=ledger,
        dependency_statuses=dep_statuses,
    )


def _model_class_names(analyzers: Sequence[_FileAnalyzer]) -> set[str]:
    """Pydantic models: ``BaseModel`` subclasses, and (transitively) subclasses of those."""
    names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for analyzer in analyzers:
            for cls in analyzer.classes.values():
                if cls.name in names:
                    continue
                bases = {_dotted(b).split(".")[-1] for b in cls.bases}
                if "BaseModel" in bases or bases & names:
                    names.add(cls.name)
                    changed = True
    return names


def analyze_files(paths: Iterable[Path | str], *, root: Path | str | None = None) -> StaticAnalysis:
    """Analyze files on disk; display names are relative to ``root`` when possible."""
    base = Path(root).resolve() if root is not None else None
    pairs = []
    for raw in paths:
        path = Path(raw).resolve()
        name = path.as_posix()
        if base is not None and path.is_relative_to(base):
            name = path.relative_to(base).as_posix()
        pairs.append((name, path.read_text(encoding="utf-8")))
    return analyze_sources(pairs)


# --------------------------------------------------------------------------- candidates


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()
    return slug if slug and slug[0].isalpha() else f"x_{slug}"


def resource_of(path: str) -> str:
    for segment in path.strip("/").split("/"):
        if segment and not segment.startswith("{"):
            return _slug(segment)
    return "root"


def entity_of(path: str) -> str:
    return _singular(resource_of(path))


def api_node_id(path: str) -> str:
    return f"api.{resource_of(path)}"


def state_node_id(entity: str, state: str) -> str:
    return f"{entity}.{_slug(state)}"


def edge_endpoints(path: str, required: str | None, produced: str | None) -> tuple[str, str]:
    entity = entity_of(path)
    if required and produced:
        return state_node_id(entity, required), state_node_id(entity, produced)
    if required:
        node = state_node_id(entity, required)
        return node, node
    if produced:
        return api_node_id(path), state_node_id(entity, produced)
    return api_node_id(path), api_node_id(path)


def node_for(node_id: str, path: str) -> StateNode:
    if node_id.startswith("api."):
        resource = node_id.removeprefix("api.")
        return StateNode(id=node_id, kind=NodeKind.API_ENDPOINT, label=f"{resource.replace('_', ' ').title()} API")
    entity, _, state = node_id.partition(".")
    return StateNode(id=node_id, kind=NodeKind.ENTITY_STATE, label=f"{entity.title()}: {state.upper()}",
                     entity=entity, state=state.upper())


def route_metadata(route: StaticRoute) -> dict[str, Any]:
    return {
        "entity": entity_of(route.path),
        "guards": {
            "ownership": route.ownership_guard,
            "preconditions": [{"field": p.field, "status": p.status, "value": p.value} for p in route.preconditions],
            "roles": sorted(route.role_guards),
            "state_requires": route.required_state,
        },
        "mutates": sorted(route.mutated_fields),
        "source": route.location.as_dict(),
        "uses_lock": route.uses_lock,
        "writes_ledger": route.writes_ledger,
    }


def candidate_edge(route: StaticRoute, analysis: StaticAnalysis, edge_id: str) -> TransitionEdge:
    source, target = edge_endpoints(route.path, route.required_state, route.produced_state)
    model = analysis.model(route.body_model) if route.body_model else None
    statuses = set(route.raised_statuses) | {route.status_code or 200}
    auth = route.auth_dependency is not None
    return TransitionEdge(
        id=edge_id, source=source, target=target, method=route.method, path=_PATH_PARAM.sub(r"{\1}", route.path),
        auth_required=auth,
        allowed_roles=frozenset(route.role_guards) if auth else frozenset(),
        required_headers=frozenset({"Authorization"}) if auth else frozenset(),
        payload_schema=model.json_schema() if model else None,
        response_statuses=frozenset(statuses),
        metadata=route_metadata(route) | {"evidence": ["static"]},
    )


def unique_edge_ids(routes: Iterable[StaticRoute]) -> list[str]:
    ids: list[str] = []
    seen: dict[str, int] = {}
    for route in routes:
        base = f"edge.{_slug(route.location.function)}"
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}_{route.method.value.lower()}_{seen[base]}")
    return ids


def extract_candidates(analysis: StaticAnalysis) -> tuple[tuple[StateNode, ...], tuple[TransitionEdge, ...]]:
    """Static-only candidate nodes and edges (refined later by the dynamic pass)."""
    edges = [
        candidate_edge(route, analysis, edge_id)
        for route, edge_id in zip(analysis.routes, unique_edge_ids(analysis.routes), strict=True)
    ]
    nodes: dict[str, StateNode] = {}
    for edge in edges:
        for node_id in (edge.source, edge.target):
            nodes.setdefault(node_id, node_for(node_id, edge.path))
    return (
        tuple(sorted(nodes.values(), key=lambda n: n.id)),
        tuple(sorted(edges, key=lambda e: e.id)),
    )
