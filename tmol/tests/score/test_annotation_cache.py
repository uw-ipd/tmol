"""Bounded annotations preserve identity, ownership, and stream readiness."""

from dataclasses import dataclass
import gc
import weakref
from types import SimpleNamespace

import torch

from tmol.score._annotation_cache import (
    AnnotationKey,
    cached_annotation,
    invalidate_annotation,
    store_annotation,
)
from tmol.tests._torch import requires_cuda


@dataclass
class Source:
    label: str = "same"

    def __eq__(self, other):
        return isinstance(other, Source)


def test_annotation_keys_match_identity_and_expire_without_retaining_sources():
    first, second = Source(), Source()
    key = AnnotationKey.from_sources(first, second, settings=(32, "cpu"))
    same = AnnotationKey.from_sources(first, second, settings=(32, "cpu"))
    assert key.matches(same)
    assert not key.matches(
        AnnotationKey.from_sources(Source(), second, settings=(32, "cpu"))
    )
    assert not key.matches(
        AnnotationKey.from_sources(first, second, settings=(64, "cpu"))
    )
    owner = SimpleNamespace()
    value = object()
    assert store_annotation(owner, "annotation", key, value) is value
    assert cached_annotation(owner, "annotation", same) is value
    reference = weakref.ref(first)
    del first
    gc.collect()
    assert reference() is None
    assert not key.matches(same)
    assert cached_annotation(owner, "annotation", same) is None
    # The same compiled term can still use its own snapshot after source expiry.
    assert cached_annotation(owner, "annotation", key) is value


def test_equal_content_sources_do_not_claim_content_sharing():
    owner = SimpleNamespace()
    first = Source()
    equal = Source()
    first_key = AnnotationKey.from_sources(first)
    equal_key = AnnotationKey.from_sources(equal)
    value = object()
    store_annotation(owner, "annotation", first_key, value)

    assert first == equal
    assert cached_annotation(owner, "annotation", equal_key) is None
    assert cached_annotation(owner, "annotation", first_key) is value


def test_equivalent_device_settings_match(torch_device):
    source = Source()
    alias = torch.device("cuda" if torch_device.type == "cuda" else "cpu:0")
    first = AnnotationKey.from_sources(source, settings=(torch_device,))
    second = AnnotationKey.from_sources(source, settings=(alias,))
    assert first.matches(second)


@requires_cuda
def test_alternating_cuda_and_cpu_devices_retain_both_bounded_entries():
    owner = SimpleNamespace()
    source = Source()
    keys = [
        AnnotationKey.from_sources(source, settings=(torch.device(device),))
        for device in ("cpu", "cuda")
    ]
    values = [object(), object()]
    for key, value in zip(keys, values):
        store_annotation(owner, "annotation", key, value)

    for _ in range(3):
        assert cached_annotation(owner, "annotation", keys[0]) is values[0]
        assert cached_annotation(owner, "annotation", keys[1]) is values[1]
    assert len(owner.annotation) == 2


def test_annotation_lru_is_bounded_and_invalidates_mutated_sources():
    owner = SimpleNamespace()
    sources = [Source(str(index)) for index in range(3)]
    keys = [AnnotationKey.from_sources(source) for source in sources]
    values = [object() for _ in sources]
    for key, value in zip(keys, values):
        store_annotation(owner, "annotation", key, value)

    assert len(owner.annotation) == 2
    assert cached_annotation(owner, "annotation", keys[0]) is None
    assert cached_annotation(owner, "annotation", keys[1]) is values[1]
    sources[1].label = "mutated"
    # Identity keys require explicit invalidation after unsupported mutation.
    invalidate_annotation(owner, "annotation", keys[1])
    assert cached_annotation(owner, "annotation", keys[1]) is None
    assert len(owner.annotation) == 1


def test_cache_hit_restores_owner_attributes_and_does_not_retain_owner():
    source = Source()
    key = AnnotationKey.from_sources(source)
    owner = SimpleNamespace(published=object())
    expected = owner.published
    child = Source("child")
    child.published = object()
    expected_child = child.published
    store_annotation(
        owner,
        "annotation",
        key,
        "snapshot",
        fields=("published",),
        bindings=((child, "published"),),
    )
    owner.published = object()
    child.published = object()

    assert cached_annotation(owner, "annotation", key) == "snapshot"
    assert owner.published is expected
    assert child.published is expected_child

    class WeakOwner:
        pass

    weak_owner = WeakOwner()
    store_annotation(weak_owner, "annotation", key, object())
    cache = weak_owner.annotation
    reference = weakref.ref(weak_owner)
    del weak_owner
    gc.collect()
    assert reference() is None
    assert len(cache) == 1


@dataclass(frozen=True)
class NestedTensor:
    values: tuple


@requires_cuda
def test_cuda_hit_waits_for_nested_annotation_producer_stream():
    owner = SimpleNamespace()
    source = Source()
    key = AnnotationKey.from_sources(source, settings=(torch.device("cuda"),))
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()

    with torch.cuda.stream(producer):
        tensor = torch.zeros(1, device="cuda")
        torch.cuda._sleep(200_000_000)
        tensor.fill_(17)
        value = NestedTensor((tensor,))
        store_annotation(owner, "annotation", key, value)

    with torch.cuda.stream(consumer):
        cached = cached_annotation(owner, "annotation", key)
        observed = cached.values[0].clone()
        consumed = torch.cuda.Event()
        consumed.record(consumer)

    assert cached is value
    assert not consumed.query()
    consumed.synchronize()
    assert observed.item() == 17
