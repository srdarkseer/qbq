"""Patch synthesis: turn a confirmed finding plus its AST context into a unified diff.

Two patchers share one interface, ``Patcher.propose(finding, context)``:

``RulePatcher`` is deterministic, offline, and needs no API keys. It applies
narrowly scoped AST rewrites keyed on the violated invariant's oracle:

* ``schema_lower_bound`` (input rejection): add ``Field(gt=0)`` to the body-model
  field that the invariant names, so validation rejects non-positive values.
* ``bypassed_guard`` (tenant / auth / lifecycle / input): a guard in the handler
  or a same-file helper is switched off by a feature-flag or configuration
  conjunct, such as ``if "flag" not in flags and owner != user: raise 404``.
  The rule removes the bypass so the guard always runs. It only applies when the
  guard raises a status the invariant expects.
* ``locked_branch`` (ledger conservation): a flag selects between an unlocked
  read-modify-write and a lock-protected one. The rule keeps the locked branch.

``LLMPatcher`` hands the same finding and AST slice to any ``PatchModel`` as a
structured ``PatchPrompt`` and applies the unified diff it returns.

Neither patcher is trusted. ``cubic.remediate.verify_fix`` decides whether a
patch fixes anything.
"""

from __future__ import annotations

import ast
import difflib
import importlib.util
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from cubic.gym.base import AppSource
from cubic.remediate.models import ConfirmedFinding, PatchCandidate
from cubic.search.combinators import Oracle
from cubic.search.policy import bound_direction, mentions_field

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


class PatchError(ValueError):
    """A patch could not be produced or applied."""


# --------------------------------------------------------------------------- source context


@dataclass(frozen=True)
class SourceContext:
    """The finding's source file, parsed, with the handler and its same-file helpers."""

    display: str
    path: Path
    text: str
    tree: ast.Module
    handler: FunctionNode
    helpers: tuple[FunctionNode, ...]
    body_models: tuple[ast.ClassDef, ...]

    @property
    def functions(self) -> tuple[FunctionNode, ...]:
        return (self.handler, *self.helpers)

    def segment(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.text, node) or ""

    def numbered(self, node: ast.AST) -> str:
        lines = self.text.splitlines()
        start, end = node.lineno, node.end_lineno or node.lineno
        return "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))


def resolve_module_path(app_source: AppSource) -> Path:
    spec = importlib.util.find_spec(app_source.module)
    if spec is None or not spec.origin:
        raise PatchError(f"cannot locate module {app_source.module!r}")
    return Path(spec.origin).resolve()


def load_context(finding: ConfirmedFinding, app_source: AppSource) -> SourceContext:
    if finding.source is None:
        raise PatchError(f"{finding.id}: no source provenance on {finding.edge_id}")
    path = resolve_module_path(app_source)
    if not path.as_posix().endswith(finding.source.file):
        raise PatchError(f"{finding.source.file} is not provided by module {app_source.module}")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    handler = next(
        (f for f in functions if f.name == finding.source.function
         and (f.lineno == finding.source.line or any(d.lineno == finding.source.line for d in f.decorator_list))),
        next((f for f in functions if f.name == finding.source.function), None),
    )
    if handler is None:
        raise PatchError(f"handler {finding.source.function!r} not found in {finding.source.file}")
    by_name = {f.name: f for f in functions}
    helpers: list[FunctionNode] = []
    frontier, seen = [handler], {handler.name}
    for _ in range(3):
        nxt = []
        for fn in frontier:
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    callee = by_name.get(node.func.id)
                    if callee is not None and callee.name not in seen:
                        seen.add(callee.name)
                        helpers.append(callee)
                        nxt.append(callee)
        frontier = nxt
    classes = {c.name: c for c in ast.walk(tree) if isinstance(c, ast.ClassDef)}
    models = tuple(
        classes[a.annotation.id] for a in handler.args.args
        if isinstance(a.annotation, ast.Name) and a.annotation.id in classes
    )
    return SourceContext(finding.source.file, path, text, tree, handler, tuple(helpers), models)


# --------------------------------------------------------------------------- text edits


@dataclass(frozen=True)
class Edit:
    start: int  # absolute character offsets into the source
    end: int
    replacement: str


def _line_starts(text: str) -> list[int]:
    starts, offset = [], 0
    for line in text.splitlines(keepends=True):
        starts.append(offset)
        offset += len(line)
    starts.append(offset)
    return starts


def _offset(text: str, starts: list[int], lineno: int, col_bytes: int) -> int:
    line = text[starts[lineno - 1]: starts[lineno]]
    return starts[lineno - 1] + len(line.encode()[:col_bytes].decode(errors="ignore"))


def node_span(text: str, node: ast.AST) -> tuple[int, int]:
    starts = _line_starts(text)
    return (_offset(text, starts, node.lineno, node.col_offset),
            _offset(text, starts, node.end_lineno or node.lineno, node.end_col_offset or 0))


def _line_span(text: str, first: int, last: int) -> tuple[int, int]:
    starts = _line_starts(text)
    return starts[first - 1], starts[last]


def apply_edits(text: str, edits: Sequence[Edit]) -> str:
    ordered = sorted(edits, key=lambda e: e.start, reverse=True)
    for a, b in zip(ordered, ordered[1:], strict=False):
        if b.end > a.start:
            raise PatchError("overlapping edits")
    for e in ordered:
        text = text[: e.start] + e.replacement + text[e.end:]
    return text


def unified_diff(display: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{display}", tofile=f"b/{display}", n=3,
    ))


def _replace_with_branch(ctx: SourceContext, stmt: ast.If, branch: Sequence[ast.stmt]) -> Edit:
    """Replace a whole ``if`` statement by one of its branches, dedented to the ``if``'s level."""
    lines = ctx.text.splitlines(keepends=True)
    first, last = branch[0].lineno, branch[-1].end_lineno or branch[-1].lineno
    shift = branch[0].col_offset - stmt.col_offset
    block = "".join(lines[first - 1: last])
    dedented = "".join(
        line[shift:] if line[:shift].strip() == "" else line.lstrip()
        for line in block.splitlines(keepends=True)
    )
    start, end = _line_span(ctx.text, stmt.lineno, stmt.end_lineno or stmt.lineno)
    return Edit(start, end, dedented)


# --------------------------------------------------------------------------- AST predicates


def _flag_check(node: ast.AST) -> tuple[str, bool] | None:
    """``"flag" not in FLAGS`` -> ("flag", True); ``"flag" in FLAGS`` -> ("flag", False)."""
    if (
        isinstance(node, ast.Compare) and len(node.ops) == 1
        and isinstance(node.ops[0], ast.In | ast.NotIn)
        and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
        and isinstance(node.comparators[0], ast.Name | ast.Attribute)
    ):
        return node.left.value, isinstance(node.ops[0], ast.NotIn)
    return None


def _raised_status(stmt: ast.AST) -> int | None:
    if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
        func = stmt.exc.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == "HTTPException":
            if stmt.exc.args and isinstance(stmt.exc.args[0], ast.Constant):
                return stmt.exc.args[0].value
            for kw in stmt.exc.keywords:
                if kw.arg == "status_code" and isinstance(kw.value, ast.Constant):
                    return kw.value.value
    return None


def _guard_statuses(nodes: Sequence[ast.AST]) -> set[int]:
    return {
        code for n in nodes for sub in ast.walk(n)
        if isinstance(sub, ast.If) for inner in sub.body if (code := _raised_status(inner)) is not None
    }


def _has_lock(nodes: Sequence[ast.AST]) -> bool:
    return any(
        isinstance(sub, ast.With | ast.AsyncWith)
        and any("lock" in ast.unparse(item.context_expr).lower() for item in sub.items)
        for n in nodes for sub in ast.walk(n)
    )


def _expected_statuses(finding: ConfirmedFinding) -> frozenset[int]:
    allowed: set[int] = set()
    for check in finding.invariant.checks:
        allowed |= set(getattr(check, "allowed", ()))
    return frozenset(allowed) or frozenset(range(400, 500))


def _ifs(ctx: SourceContext) -> Iterator[ast.If]:
    seen: set[int] = set()
    for fn in ctx.functions:
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and id(node) not in seen:
                seen.add(id(node))
                yield node


# --------------------------------------------------------------------------- patchers


@runtime_checkable
class Patcher(Protocol):
    name: str

    def propose(self, finding: ConfirmedFinding, ctx: SourceContext) -> list[PatchCandidate]: ...


def _candidate(patcher: str, rule: str, finding: ConfirmedFinding, ctx: SourceContext,
               edits: Sequence[Edit], description: str) -> PatchCandidate:
    patched = apply_edits(ctx.text, edits)
    ast.parse(patched)  # never emit a patch that does not parse
    return PatchCandidate(
        finding_id=finding.id, patcher=patcher, rule=rule, file=ctx.display, description=description,
        diff=unified_diff(ctx.display, ctx.text, patched), patched_source=patched,
    )


class RulePatcher:
    """Deterministic AST-rule patcher (no network, no model)."""

    name = "rule"

    def propose(self, finding: ConfirmedFinding, ctx: SourceContext) -> list[PatchCandidate]:
        out: list[PatchCandidate] = []
        rules = {
            Oracle.REJECTED_INPUT: (self._schema_lower_bound, self._bypassed_guard),
            Oracle.FORBIDDEN: (self._bypassed_guard,),
            Oracle.UNAUTHENTICATED: (self._bypassed_guard,),
            Oracle.REJECTED_TRANSITION: (self._bypassed_guard,),
            Oracle.STATE_GUARD: (self._bypassed_guard,),
            Oracle.CONSERVATION: (self._locked_branch,),
        }.get(finding.oracle, ())
        for rule in rules:
            candidate = rule(finding, ctx)
            if candidate is not None and all(c.diff != candidate.diff for c in out):
                out.append(candidate)
        return out

    # -- rules ------------------------------------------------------------------

    def _schema_lower_bound(self, finding: ConfirmedFinding, ctx: SourceContext) -> PatchCandidate | None:
        if bound_direction(finding.invariant) == "upper":
            return None  # the invariant is about values that are too large; gt=0 would not help
        for model in ctx.body_models:
            for stmt in model.body:
                if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                    continue
                field = stmt.target.id
                if ast.unparse(stmt.annotation) not in {"int", "float"} or not mentions_field(finding.invariant, field):
                    continue
                edits: list[Edit] = []
                if stmt.value is None:
                    start, end = node_span(ctx.text, stmt)
                    line_end = ctx.text.find("\n", end)
                    if ctx.text[end:line_end].lstrip().startswith("#"):
                        end = line_end  # the trailing comment described the old, unconstrained field
                    edits.append(Edit(start, end, f"{field}: {ast.unparse(stmt.annotation)} = Field(gt=0)"))
                elif isinstance(stmt.value, ast.Call) and ast.unparse(stmt.value.func).endswith("Field"):
                    if any(k.arg in {"gt", "ge"} for k in stmt.value.keywords):
                        continue
                    _, end = node_span(ctx.text, stmt.value)
                    sep = ", " if stmt.value.args or stmt.value.keywords else ""
                    edits.append(Edit(end - 1, end - 1, f"{sep}gt=0"))
                else:
                    continue
                edits += self._ensure_field_import(ctx)
                return _candidate(
                    self.name, "schema_lower_bound", finding, ctx, edits,
                    f"Reject non-positive {model.name}.{field} at the schema boundary (Field(gt=0)).",
                )
        return None

    @staticmethod
    def _ensure_field_import(ctx: SourceContext) -> list[Edit]:
        for node in ctx.tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "pydantic":
                if any(a.name == "Field" for a in node.names):
                    return []
                start, end = node_span(ctx.text, node)
                names = sorted([*(ast.unparse(a) for a in node.names), "Field"])
                return [Edit(start, end, f"from pydantic import {', '.join(names)}")]
        first = next(n for n in ctx.tree.body if isinstance(n, ast.Import | ast.ImportFrom))
        start, _ = node_span(ctx.text, first)
        return [Edit(start, start, "from pydantic import Field\n")]

    def _bypassed_guard(self, finding: ConfirmedFinding, ctx: SourceContext) -> PatchCandidate | None:
        expected = _expected_statuses(finding)
        for stmt in _ifs(ctx):
            # conjunct form: if "flag" not in FLAGS and <guard>: raise HTTPException(code)
            if isinstance(stmt.test, ast.BoolOp) and isinstance(stmt.test.op, ast.And):
                flags = [v for v in stmt.test.values if (f := _flag_check(v)) and f[1]]
                rest = [v for v in stmt.test.values if v not in flags]
                codes = {c for s in stmt.body if (c := _raised_status(s)) is not None}
                if flags and rest and codes & expected:
                    if finding.oracle is Oracle.REJECTED_INPUT and not any(
                        mentions_field(finding.invariant, n.attr) for r in rest for n in ast.walk(r)
                        if isinstance(n, ast.Attribute)
                    ):
                        continue
                    text = " and ".join(
                        f"({ctx.segment(v)})" if isinstance(v, ast.BoolOp) else ctx.segment(v) for v in rest
                    )
                    start, end = node_span(ctx.text, stmt.test)
                    flag = _flag_check(flags[0])[0]  # type: ignore[index]
                    return _candidate(
                        self.name, "bypassed_guard", finding, ctx, [Edit(start, end, text)],
                        f"Enforce the {sorted(codes)} guard in {ctx.handler.name} unconditionally "
                        f"(remove the {flag!r} bypass).",
                    )
            # wrapper form: if "flag" not in FLAGS: <guards raising code>
            flag = _flag_check(stmt.test)
            if flag and flag[1] and not stmt.orelse and _guard_statuses(stmt.body) & expected:
                return _candidate(
                    self.name, "bypassed_guard", finding, ctx, [_replace_with_branch(ctx, stmt, stmt.body)],
                    f"Run the guard block in {ctx.handler.name} unconditionally (remove the {flag[0]!r} bypass).",
                )
        return None

    def _locked_branch(self, finding: ConfirmedFinding, ctx: SourceContext) -> PatchCandidate | None:
        for stmt in _ifs(ctx):
            flag = _flag_check(stmt.test)
            if not flag or not stmt.orelse:
                continue
            body_locked, else_locked = _has_lock(stmt.body), _has_lock(stmt.orelse)
            if body_locked == else_locked:
                continue
            keep = stmt.body if body_locked else stmt.orelse
            return _candidate(
                self.name, "locked_branch", finding, ctx, [_replace_with_branch(ctx, stmt, keep)],
                f"Always take the lock-protected read-modify-write in {ctx.handler.name} "
                f"(remove the {flag[0]!r} unlocked path).",
            )
        return None


# --------------------------------------------------------------------------- unified diff application


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_unified_diff(source: str, diff: str) -> str:
    """Apply a single-file unified diff. Context must match exactly; hunks may drift."""
    lines = source.splitlines(keepends=True)
    hunks: list[tuple[int, list[str], list[str]]] = []
    current: tuple[int, list[str], list[str]] | None = None
    for raw in diff.splitlines(keepends=True):
        if raw.startswith(("---", "+++")) and current is None:
            continue
        if m := _HUNK.match(raw):
            current = (int(m.group(1)), [], [])
            hunks.append(current)
            continue
        if current is None or raw.startswith("\\"):
            continue
        body = raw[1:] if raw[:1] in {" ", "-", "+"} else raw
        if not body.endswith("\n"):
            body += "\n"
        if raw.startswith("-"):
            current[1].append(body)
        elif raw.startswith("+"):
            current[2].append(body)
        else:
            current[1].append(body)
            current[2].append(body)
    if not hunks:
        raise PatchError("no hunks in diff")
    offset = 0
    for start, old, new in hunks:
        guess = max(start - 1 + offset, 0)
        positions = [i for i in range(len(lines) - len(old) + 1) if lines[i: i + len(old)] == old]
        if not positions:
            raise PatchError(f"hunk at line {start} does not match the source")
        at = min(positions, key=lambda i: abs(i - guess))
        lines[at: at + len(old)] = new
        offset += len(new) - len(old)
    return "".join(lines)


# --------------------------------------------------------------------------- LLM patcher


class PatchPrompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: str
    finding: dict
    file: str
    ast_slice: str

    def render(self) -> str:
        return json.dumps({"finding": self.finding, "file": self.file, "ast_slice": self.ast_slice},
                          sort_keys=True, indent=None, separators=(",", ":"))

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.render()}]


@runtime_checkable
class PatchModel(Protocol):
    def complete(self, prompt: PatchPrompt) -> str: ...


PATCH_SYSTEM_PROMPT = (
    "You fix a verified security or business-logic defect in a Python web service. You get the "
    "violated invariant, the replayed trace, and line-numbered source for the handler, its "
    "helpers and its request models. Reply with a minimal unified diff against the given file "
    "(--- a/<file> / +++ b/<file>), in a ```diff block. Do not change unrelated behaviour."
)


def build_patch_prompt(finding: ConfirmedFinding, ctx: SourceContext) -> PatchPrompt:
    parts = [ctx.numbered(n) for n in (*ctx.body_models, *ctx.helpers, ctx.handler)]
    return PatchPrompt(
        system=PATCH_SYSTEM_PROMPT,
        finding={
            "invariant": finding.invariant.model_dump(mode="json"),
            "oracle": finding.oracle.value,
            "hypothesis": finding.candidate.hypothesis,
            # personas are referenced by id; their credentials are never sent to a model
            "trace": [s.model_dump(mode="json", exclude_defaults=True) for s in finding.candidate.trace.steps],
            "edge": finding.edge_id,
        },
        file=ctx.display,
        ast_slice="\n\n".join(parts),
    )


def extract_diff(text: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return fenced.group(1) if fenced else text


class LLMPatcher:
    """Ask a ``PatchModel`` for a unified diff. Unusable replies are recorded in ``errors``."""

    name = "llm"

    def __init__(self, model: PatchModel) -> None:
        self.model = model
        self.errors: list[str] = []
        self.prompts: list[PatchPrompt] = []

    def propose(self, finding: ConfirmedFinding, ctx: SourceContext) -> list[PatchCandidate]:
        prompt = build_patch_prompt(finding, ctx)
        self.prompts.append(prompt)
        try:
            patched = apply_unified_diff(ctx.text, extract_diff(self.model.complete(prompt)))
            ast.parse(patched)
        except Exception as exc:  # noqa: BLE001 - any model failure means "no candidate"
            self.errors.append(f"{finding.id}: {exc}")
            return []
        if patched == ctx.text:
            self.errors.append(f"{finding.id}: empty patch")
            return []
        return [PatchCandidate(
            finding_id=finding.id, patcher=self.name, rule="model", file=ctx.display,
            description="Model-proposed patch.", diff=unified_diff(ctx.display, ctx.text, patched),
            patched_source=patched,
        )]


def propose_patches(
    finding: ConfirmedFinding, app_source: AppSource, patchers: Sequence[Patcher] | None = None
) -> list[PatchCandidate]:
    ctx = load_context(finding, app_source)
    out: list[PatchCandidate] = []
    for patcher in patchers or (RulePatcher(),):
        out += patcher.propose(finding, ctx)
    return out
