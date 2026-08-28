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
        digest.update(repr((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    elif attr.has(type(value)):
        for field in attr.fields(type(value)):
            digest.update(field.name.encode())
            _update(digest, getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _update(digest, item)
    else:
        digest.update(repr(value).encode())
