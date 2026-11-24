"""Database snapshots and row-level diffs (gray-box view of application state)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DbSnapshot(BaseModel):
    """Point-in-time copy of application state.

    ``entities`` maps entity name -> row id -> row fields; ``ledger`` is an
    append-only list of signed entries.
    """

    model_config = ConfigDict(frozen=True)

    entities: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)
    ledger: tuple[dict[str, Any], ...] = ()


class FieldChange(BaseModel):
    entity: str
    row_id: str
    field: str
    before: Any = None
    after: Any = None


class DbDiff(BaseModel):
    changed: list[FieldChange] = Field(default_factory=list)
    added_rows: dict[str, list[str]] = Field(default_factory=dict)
    removed_rows: dict[str, list[str]] = Field(default_factory=dict)
    ledger_appended: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.changed or self.added_rows or self.removed_rows or self.ledger_appended)


_MISSING = object()


def diff_snapshots(before: DbSnapshot, after: DbSnapshot) -> DbDiff:
    diff = DbDiff()
    for entity in sorted(before.entities.keys() | after.entities.keys()):
        old_rows = before.entities.get(entity, {})
        new_rows = after.entities.get(entity, {})
        if added := sorted(new_rows.keys() - old_rows.keys()):
            diff.added_rows[entity] = added
        if removed := sorted(old_rows.keys() - new_rows.keys()):
            diff.removed_rows[entity] = removed
        for row_id in sorted(old_rows.keys() & new_rows.keys()):
            old, new = old_rows[row_id], new_rows[row_id]
            for field in sorted(old.keys() | new.keys()):
                a, b = old.get(field, _MISSING), new.get(field, _MISSING)
                if a != b:
                    diff.changed.append(
                        FieldChange(
                            entity=entity, row_id=row_id, field=field,
                            before=None if a is _MISSING else a,
                            after=None if b is _MISSING else b,
                        )
                    )
    if after.ledger[: len(before.ledger)] == before.ledger:
        diff.ledger_appended = list(after.ledger[len(before.ledger):])
    else:  # history was rewritten (append-only violated): report the full new ledger
        diff.ledger_appended = list(after.ledger)
    return diff
