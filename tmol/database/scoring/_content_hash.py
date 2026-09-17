"""Content-derived identity for parameter databases.

A database's id is used as a cache key, so it must change whenever the contents
change; otherwise a cache hands one database's parameters to another.
"""

import hashlib

import attr
import torch


def content_hash(*parts) -> str:
    """A sha256 over attrs records, tensors and scalars, in the order given."""
    digest = hashlib.sha256()
    for part in parts:
        _update(digest, part)
    return digest.hexdigest()


def _update(digest, value) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor:")
        _update(digest, (tuple(value.shape), str(value.dtype)))
        array = value.detach().cpu().contiguous().numpy().reshape(-1)
        _write(digest, memoryview(array).cast("B"))
    elif attr.has(type(value)):
        digest.update(b"attrs:")
        _write(digest, type(value).__qualname__.encode())
        for field in attr.fields(type(value)):
            _write(digest, field.name.encode())
            _update(digest, getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence:")
        digest.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update(digest, item)
    elif value is None or isinstance(value, (str, bytes, bool, int, float)):
        digest.update(b"scalar:")
        _write(digest, repr(value).encode())
    else:
        # repr is not a safe stand-in for contents: numpy truncates past 1000
        # elements, so two large arrays differing in the middle hash alike, and
        # a set reprs in a different order each interpreter run. A collision
        # here hands one database's parameters to another, which is the single
        # thing this identity exists to prevent.
        raise TypeError(
            f"content hashing does not describe {type(value).__name__}; add a "
            "case for it rather than letting repr stand in for its contents"
        )


def _write(digest, data):
    """Frame each byte string so distinct adjacent values cannot alias."""
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)
