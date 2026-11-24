"""Static pass: extract routes, guards, models and state effects from Python source.

Uses only the standard-library ``ast`` module. The analyzer understands the
FastAPI idioms used by typical services:

* route decorators ``@<app|router>.<get|post|put|patch|delete>(path, status_code=...)``,
  including handlers nested inside an app factory;
* auth dependencies via ``Annotated[..., Depends(fn)]`` aliases, inline
  ``Annotated`` annotations, or ``= Depends(fn)`` defaults;
* guard clauses ``if <cond>: raise HTTPException(<status>)`` (role checks,
  ownership checks, lifecycle-state preconditions), followed one level into
  same-file helper functions such as ``owned_wallet(...)``;
* state effects: constant writes to ``status``/``state`` fields, dict literals that
  create rows in a given state, mutated fields, ledger writes, and lock usage;
* Pydantic request models (``class X(BaseModel)``) with ``Field`` constraints and
  store collections (``dict[str, dict]`` entities, ``list[dict]`` ledgers).

Conjuncts of the form ``"flag" in/not in <name>`` are treated as feature flags and
ignored when classifying a guard; the analyzer describes the hardened code path.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cubic.graph import (
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


class _FileAnalyzer:
    def __init__(self, tree: ast.Module, display_name: str) -> None:
        self.tree = tree
        self.file = display_name
        self.functions: dict[str, FunctionNode] = {}  # name -> first definition (helper lookup)
        self.all_functions: list[FunctionNode] = []  # every definition, including duplicates
        self.aliases: dict[str, str] = {}  # alias name -> dependency function
        self.classes: dict[str, ast.ClassDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                self.functions.setdefault(node.name, node)
                self.all_functions.append(node)
            elif isinstance(node, ast.ClassDef):
                self.classes.setdefault(node.name, node)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if (dep := self._depends_in(node.value)) is not None:
                    self.aliases[node.targets[0].id] = dep

    # -- dependencies -------------------------------------------------------

    def _depends_in(self, node: ast.AST | None) -> str | None:
        if node is None:
            return None
        for sub in ast.walk(node):
            if _call_name(sub) == "Depends" and isinstance(sub, ast.Call) and sub.args:
                target = sub.args[0]
                if isinstance(target, ast.Name):
                    return target.id
        return None

    def _param_dependencies(self, fn: FunctionNode) -> list[str]:
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        defaults: list[ast.AST | None] = [None] * (len(fn.args.posonlyargs + fn.args.args) - len(fn.args.defaults))
        defaults += list(fn.args.defaults) + list(fn.args.kw_defaults)
        deps: list[str] = []
        for arg, default in zip(args, defaults, strict=True):
            ann = arg.annotation
            if isinstance(ann, ast.Name) and ann.id in self.aliases:
                deps.append(self.aliases[ann.id])
            elif (dep := self._depends_in(ann)) is not None or (dep := self._depends_in(default)) is not None:
                deps.append(dep)
        return deps

    def dependency_statuses(self, name: str) -> frozenset[int]:
        fn = self.functions.get(name)
        if fn is None:
            return frozenset()
        return frozenset(c for n in _walk_shallow(fn.body) if (c := _http_exception_status(n)) is not None)

    def _is_auth_dependency(self, name: str) -> bool:
        fn = self.functions.get(name)
        if fn is None:
            return False
        arg_names = {a.arg.lower() for a in fn.args.args + fn.args.kwonlyargs}
        return 401 in self.dependency_statuses(name) or "authorization" in arg_names

    # -- helpers called from handlers -----------------------------------------

    def _reachable_bodies(self, fn: FunctionNode, depth: int = 3) -> list[ast.AST]:
        seen = {fn.name}
        bodies: list[ast.AST] = list(fn.body)
        frontier = [fn]
        for _ in range(depth):
            nxt: list[FunctionNode] = []
            for f in frontier:
                for node in _walk_shallow(f.body):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        callee = self.functions.get(node.func.id)
                        if callee is not None and callee.name not in seen and callee.name not in self.aliases.values():
                            seen.add(callee.name)
                            bodies.extend(callee.body)
                            nxt.append(callee)
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

    def models(self) -> list[RequestModel]:
        out: list[RequestModel] = []
        for cls in self.classes.values():
            if not any(_dotted(b).split(".")[-1] == "BaseModel" for b in cls.bases):
                continue
            fields: list[ModelField] = []
            for stmt in cls.body:
                if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                    continue
                json_type = _JSON_TYPES.get(_dotted(stmt.annotation))
                required, constraints = stmt.value is None, []
                if isinstance(stmt.value, ast.Call) and _call_name(stmt.value) == "Field":
                    has_default = bool(stmt.value.args) and not (
                        isinstance(stmt.value.args[0], ast.Constant) and stmt.value.args[0].value is Ellipsis
                    )
                    has_default |= any(k.arg in {"default", "default_factory"} for k in stmt.value.keywords)
                    required = not has_default
                    for kw in stmt.value.keywords:
                        if kw.arg in _CONSTRAINTS and isinstance(kw.value, ast.Constant):
                            constraints.append((_CONSTRAINTS[kw.arg], kw.value.value))
                fields.append(ModelField(stmt.target.id, json_type, required, tuple(sorted(constraints))))
            out.append(RequestModel(cls.name, tuple(fields), SourceLocation(self.file, cls.lineno, cls.name)))
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
                routes.append(self._route(fn, method, path, status, dec.lineno, ledger_methods, lock_hint))
        return routes

    def _route(
        self, fn: FunctionNode, method: HttpMethod, path: str, status: int | None,
        decorator_line: int, ledger_methods: set[str], lock_hint: str,
    ) -> StaticRoute:
        deps = self._param_dependencies(fn)
        auth = next((d for d in deps if self._is_auth_dependency(d)), None)
        body_model = next(
            (a.annotation.id for a in fn.args.args
             if isinstance(a.annotation, ast.Name) and a.annotation.id in self.classes),
            None,
        )
        bodies = self._reachable_bodies(fn)
        roles: set[str] = set()
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
                conj = _conjuncts(node.test)
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

        if auth:
            raised |= self.dependency_statuses(auth)
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
    """Analyze ``(display_name, source_text)`` pairs."""
    files: list[str] = []
    total = 0
    routes: list[StaticRoute] = []
    models: list[RequestModel] = []
    roles: set[str] = set()
    entities: set[str] = set()
    ledger: LedgerSpec | None = None
    dep_statuses: dict[str, frozenset[int]] = {}

    for name, text in sorted(sources):
        files.append(name)
        total += len(text.encode())
        analyzer = _FileAnalyzer(ast.parse(text, filename=name), name)
        file_entities, lists, method_attrs = analyzer.collections()
        entities |= file_entities
        spec = analyzer.ledger_spec(file_entities, lists)
        ledger_methods = {m for m, attrs in method_attrs.items() if spec and spec.collection in attrs}
        file_routes = analyzer.routes(ledger_methods)
        if spec and spec.entity:
            balance = sorted({
                f for r in file_routes if r.writes_ledger for f in r.mutated_fields
                if f not in {"id", spec.account_field, spec.amount_field} | LIFECYCLE_FIELDS
            })
            if len(balance) == 1:
                spec = LedgerSpec(spec.collection, spec.account_field, spec.amount_field, spec.entity, balance[0])
        ledger = ledger or spec
        routes += file_routes
        models += analyzer.models()
        roles |= analyzer.roles()
        for dep in sorted(set(analyzer.aliases.values())):
            dep_statuses[dep] = analyzer.dependency_statuses(dep)

    routes.sort(key=lambda r: (r.path, r.method.value, r.location.file, r.location.line))
    models.sort(key=lambda m: (m.name, m.location.file))
    return StaticAnalysis(
        files=tuple(files), source_bytes=total, routes=tuple(routes), models=tuple(models),
        roles=frozenset(roles), entity_collections=frozenset(entities), ledger=ledger,
        dependency_statuses=dep_statuses,
    )


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
