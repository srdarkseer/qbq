"""Shared field types for state-graph elements."""

from typing import Annotated

from pydantic import PlainSerializer, StringConstraints

Identifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$"),
]
"""Stable, human-readable id (e.g. ``edge.withdraw``). Must start with a letter."""

# Sets serialize as sorted lists so that identical graphs yield byte-identical JSON.
StrSet = Annotated[frozenset[str], PlainSerializer(sorted, return_type=list[str])]
IntSet = Annotated[frozenset[int], PlainSerializer(sorted, return_type=list[int])]
