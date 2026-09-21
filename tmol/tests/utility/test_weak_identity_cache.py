"""Weak identity cache publication must respect owner and CUDA lifetimes."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading

import attr
import pytest
import torch

from tmol.tests._torch import requires_cuda
from tmol.utility.weak_identity_cache import WeakIdentityLRU


@dataclass
class Owner:
    """Weak-referenceable, unhashable cache owner."""

    label: str


@dataclass(frozen=True)
class NestedTensors:
    """Dataclass layer matching nested cached table structures."""

    by_name: dict


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Resolver:
    """Attrs layer matching cached TMol parameter resolvers."""

    nested: NestedTensors


def _resolver(tensor):
    return Resolver(NestedTensors({"table": (tensor,)}))


def test_cpu_values_preserve_hit_and_none_semantics():
    cache = WeakIdentityLRU()
    owner = Owner("cpu")
    value = Resolver(NestedTensors({"table": (torch.ones(1),)}))

    assert cache.get_or_create(owner, "value", lambda: value) is value
    assert cache.get_or_create(owner, "value", pytest.fail) is value
    assert cache.get_or_create(owner, "none", lambda: None) is None
    assert cache.get_or_create(owner, "none", pytest.fail) is None


@requires_cuda
def test_cuda_hit_waits_for_nested_value_producer_stream():
    cache = WeakIdentityLRU()
    owner = Owner("cuda")
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()

    with torch.cuda.stream(producer):
        tensor = torch.zeros(1, device="cuda")
        torch.cuda._sleep(200_000_000)
        tensor.fill_(17)
        value = cache.get_or_create(owner, "table", lambda: _resolver(tensor))

    with torch.cuda.stream(consumer):
        cached = cache.get_or_create(owner, "table", pytest.fail)
        observed = cached.nested.by_name["table"][0].clone()
        consumed = torch.cuda.Event()
        consumed.record(consumer)

    assert cached is value
    assert not consumed.query()
    consumed.synchronize()
    assert observed.item() == 17


@requires_cuda
def test_simultaneous_cuda_miss_loser_waits_for_winner_value():
    cache = WeakIdentityLRU()
    owner = Owner("shared")
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]

    def build_and_consume(index):
        with torch.cuda.stream(streams[index]):

            def factory():
                tensor = torch.zeros(1, device="cuda")
                if index == 0:
                    torch.cuda._sleep(200_000_000)
                tensor.fill_(29 + 12 * index)
                entered[index].set()
                assert release[index].wait(timeout=10)
                return _resolver(tensor)

            result = cache.get_or_create(owner, "table", factory)
            observed = result.nested.by_name["table"][0].clone()
            consumed = torch.cuda.Event()
            consumed.record(streams[index])
            return result, observed, consumed

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(build_and_consume, index) for index in range(2)]
        assert all(event.wait(timeout=10) for event in entered)

        release[0].set()
        winner, _, _ = futures[0].result(timeout=10)
        release[1].set()
        loser, observed, consumed = futures[1].result(timeout=10)

    assert loser is winner
    assert not consumed.query()
    consumed.synchronize()
    assert observed.item() == 29
