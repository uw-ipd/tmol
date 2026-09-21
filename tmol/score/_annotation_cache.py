"""Bounded owner annotations keyed by weak source identity."""

from dataclasses import dataclass
import threading
import weakref

import torch

from tmol.utility import resolve_device
from tmol.utility.weak_identity_cache import (
    _record_cuda_readiness,
    _wait_for_cuda_readiness,
)

_ANNOTATION_CACHE_SIZE = 2


def _normalize_setting(value):
    if isinstance(value, torch.device):
        value = resolve_device(value)
        return torch.device("cpu") if value.type == "cpu" else value
    if isinstance(value, tuple):
        return tuple(_normalize_setting(item) for item in value)
    return value


@dataclass(frozen=True, slots=True, eq=False)
class AnnotationKey:
    """Identify an annotation by live source objects and scalar settings.

    Equal but distinct sources intentionally do not match. Sources are expected
    to be immutable snapshots; call :func:`invalidate_annotation` after an
    unsupported in-place mutation of data used by an annotation.
    """

    sources: tuple
    settings: tuple = ()

    @classmethod
    def from_sources(cls, *sources, settings=()):
        return cls(
            tuple(weakref.ref(source) for source in sources),
            tuple(_normalize_setting(setting) for setting in settings),
        )

    def matches(self, other):
        # A compiled configuration remains valid after its source expires.
        # Distinct keys still require live, identical source referents.
        if self is other:
            return True
        return (
            isinstance(other, AnnotationKey)
            and self.settings == other.settings
            and len(self.sources) == len(other.sources)
            and all(
                (source := first()) is not None and source is second()
                for first, second in zip(self.sources, other.sources)
            )
        )


@dataclass(frozen=True, slots=True)
class _AnnotationEntry:
    key: AnnotationKey
    value: object
    attributes: tuple
    bindings: tuple
    readiness: tuple


class _AnnotationLRU:
    """Small per-owner LRU that does not retain annotation key sources."""

    def __init__(self, capacity=_ANNOTATION_CACHE_SIZE):
        self.capacity = capacity
        self._entries = []
        self._lock = threading.RLock()

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def get(self, key, *, restore_owner=None):
        with self._lock:
            for index in range(len(self._entries) - 1, -1, -1):
                entry = self._entries[index]
                if key.matches(entry.key):
                    self._entries.append(self._entries.pop(index))
                    break
            else:
                return None
        _wait_for_cuda_readiness(entry.readiness)
        if restore_owner is not None:
            for name, value in entry.attributes:
                setattr(restore_owner, name, value)
            for target_ref, name, value in entry.bindings:
                target = target_ref()
                if target is not None:
                    setattr(target, name, value)
        return entry.value

    def latest(self):
        with self._lock:
            entry = self._entries[-1] if self._entries else None
        if entry is None:
            return None
        _wait_for_cuda_readiness(entry.readiness)
        return entry.value

    def store(self, key, value, attributes, bindings):
        readiness = _record_cuda_readiness(
            (
                value,
                tuple(item[1] for item in attributes),
                tuple(item[2] for item in bindings),
            )
        )
        entry = _AnnotationEntry(key, value, attributes, bindings, readiness)
        with self._lock:
            self._entries = [
                current for current in self._entries if not key.matches(current.key)
            ]
            self._entries.append(entry)
            del self._entries[: -self.capacity]
        _wait_for_cuda_readiness(readiness)
        return value

    def invalidate(self, key=None):
        with self._lock:
            if key is None:
                self._entries.clear()
            else:
                self._entries = [
                    entry for entry in self._entries if not key.matches(entry.key)
                ]


def _cache(owner, attribute):
    cache = getattr(owner, attribute, None)
    if not isinstance(cache, _AnnotationLRU):
        cache = _AnnotationLRU()
        setattr(owner, attribute, cache)
    return cache


def cached_annotation(owner, attribute, key):
    cache = getattr(owner, attribute, None)
    if not isinstance(cache, _AnnotationLRU):
        return None
    return cache.get(key, restore_owner=owner)


def store_annotation(owner, attribute, key, value, *, fields=(), bindings=()):
    """Store a snapshot and owner attributes that must be restored on a hit."""
    attributes = tuple((name, getattr(owner, name)) for name in fields)
    weak_bindings = tuple(
        (weakref.ref(target), name, getattr(target, name)) for target, name in bindings
    )
    return _cache(owner, attribute).store(key, value, attributes, weak_bindings)


def latest_annotation(owner, attribute):
    """Return the newest snapshot regardless of key, without republishing it."""
    cache = getattr(owner, attribute, None)
    return cache.latest() if isinstance(cache, _AnnotationLRU) else None


def invalidate_annotation(owner, attribute, key=None):
    """Invalidate one matching annotation or every entry for an owner field."""
    cache = getattr(owner, attribute, None)
    if isinstance(cache, _AnnotationLRU):
        cache.invalidate(key)
