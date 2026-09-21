"""Bounded derived-data caches for immutable, possibly unhashable owners."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import threading
import weakref

import torch


def _cuda_devices(value):
    """Find CUDA tensor devices nested in supported cache value containers."""
    devices = set()
    pending = [value]
    seen = set()
    while pending:
        item = pending.pop()
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)

        if isinstance(item, torch.Tensor):
            if item.device.type == "cuda":
                devices.add(item.device)
        elif isinstance(item, Mapping):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            pending.extend(item)
        elif not isinstance(item, type) and is_dataclass(item):
            pending.extend(getattr(item, field.name) for field in fields(item))
        else:
            attrs_fields = getattr(type(item), "__attrs_attrs__", ())
            pending.extend(getattr(item, field.name) for field in attrs_fields)
    return devices


def _record_cuda_readiness(value):
    """Record producer-stream readiness for every CUDA device in a value."""
    readiness = []
    for device in sorted(_cuda_devices(value), key=lambda item: item.index):
        with torch.cuda.device(device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
        readiness.append((device, event))
    return tuple(readiness)


def _wait_for_cuda_readiness(readiness):
    """Make each device's caller-current stream wait for cached production."""
    for device, event in readiness:
        torch.cuda.current_stream(device).wait_event(event)


class WeakIdentityLRU:
    """Cache by owner identity and configuration without retaining the owner.

    Values must not retain their owner. Owners must be weak-referenceable and
    remain unchanged while cached; publish a new owner after changing its data.
    Factories run outside the lock, so simultaneous misses may compute twice.
    CUDA tensor values carry producer-stream events; each caller's current
    stream waits for the published value without synchronizing the device.
    """

    def __init__(self, capacity=32):
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def get_or_create(self, owner, configuration, factory):
        return self.get_or_create_many((owner,), configuration, factory)

    def get_or_create_many(self, owners, configuration, factory):
        """Cache a value derived from several independently owned databases."""
        owners = tuple(owners)
        if not owners:
            raise ValueError("at least one cache owner is required")
        key = (tuple(id(owner) for owner in owners), configuration)

        def matches(entry):
            return entry is not None and all(
                ref() is owner for ref, owner in zip(entry[0], owners)
            )

        with self._lock:
            entry = self._entries.get(key)
            if matches(entry):
                self._entries.move_to_end(key)
                value = entry[1]
                readiness = entry[2] if len(entry) > 2 else ()
                found = True
            else:
                found = False
        if found:
            _wait_for_cuda_readiness(readiness)
            return value

        value = factory()
        readiness = _record_cuda_readiness(value)

        def discard(ref):
            with self._lock:
                current = self._entries.get(key)
                if current is not None and any(stored is ref for stored in current[0]):
                    del self._entries[key]

        with self._lock:
            # Another caller may have completed the same miss meanwhile.
            current = self._entries.get(key)
            if matches(current):
                self._entries.move_to_end(key)
                value = current[1]
                readiness = current[2] if len(current) > 2 else ()
            else:
                self._entries[key] = (
                    tuple(weakref.ref(owner, discard) for owner in owners),
                    value,
                    readiness,
                )
                self._entries.move_to_end(key)
                while len(self._entries) > self.capacity:
                    self._entries.popitem(last=False)
        _wait_for_cuda_readiness(readiness)
        return value
