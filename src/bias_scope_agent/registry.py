"""Opaque handles for objects that must not cross the tool-call boundary.

Backends may hold live model state; Reports may be large. Rather than
serializing either through a tool call, callers get a UUID handle and look
the real object up server-side via a registry owned by the AgentSession.
"""

from __future__ import annotations

from typing import Dict, Generic, TypeVar
from uuid import uuid4

T = TypeVar("T")


class HandleRegistry(Generic[T]):
    """Maps opaque string handles to live objects, one registry per session."""

    def __init__(self) -> None:
        self._items: Dict[str, T] = {}

    def register(self, item: T) -> str:
        handle = uuid4().hex
        self._items[handle] = item
        return handle

    def get(self, handle: str) -> T:
        try:
            return self._items[handle]
        except KeyError:
            raise KeyError(f"unknown handle {handle!r}") from None
