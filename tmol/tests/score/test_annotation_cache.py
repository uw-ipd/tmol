"""Annotation identity must not alias expired or merely equal databases."""

import gc
import weakref
from types import SimpleNamespace

from tmol.score._annotation_cache import (
    AnnotationKey,
    cached_annotation,
    store_annotation,
)


class Source:
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
