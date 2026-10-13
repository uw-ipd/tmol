"""Shared type helpers for tmol_jax.

Replaces tmol's `tmol.types.torch.Tensor[...]`, `TensorGroup`, and
`attrs`-based frozen dataclasses with JAX-friendly pytrees backed by
`jax.tree_util.register_pytree_node_class`.

The key idea: in JAX you want your container objects to be pytrees so
`jit`, `vmap`, `grad`, and `lax.scan` can flow through them. We provide
a simple `@pytree_dataclass` decorator that turns an `attrs.define` class
into a registered pytree. All array fields become "children" (leaves), all
non-array fields become "aux" (static) metadata.
"""

from __future__ import annotations

from typing import Any

import attrs
import jax
import jax.numpy as jnp
import numpy as np

# Type aliases used throughout the package. They are intentionally loose:
# JAX arrays, numpy arrays, and Python scalars all satisfy them.
Array = jax.Array | np.ndarray
NDArray = np.ndarray


_PYTREE_TYPES = set()


def is_array(x: Any) -> bool:
    return isinstance(x, (jax.Array, jax.core.Tracer, np.ndarray))


def pytree_dataclass(cls):
    """Decorator: turn an attrs class into a registered JAX pytree.

    Array-valued fields are children; everything else is aux data.
    The set of array fields is determined *per instance* so classes with
    optional array fields still work.
    """

    if not hasattr(cls, "__attrs_attrs__"):
        cls = attrs.define(slots=False, frozen=True)(cls)

    field_names = tuple(f.name for f in cls.__attrs_attrs__)

    def tree_flatten(obj):
        values = [getattr(obj, n) for n in field_names]
        child_mask = tuple(is_array(v) or type(v) in _PYTREE_TYPES for v in values)
        children = tuple(v for v, m in zip(values, child_mask) if m)
        aux_values = tuple(v for v, m in zip(values, child_mask) if not m)
        return children, (child_mask, aux_values)

    def tree_unflatten(aux, children):
        child_mask, aux_values = aux
        c_iter = iter(children)
        a_iter = iter(aux_values)
        kw = {}
        for name, is_child in zip(field_names, child_mask):
            kw[name] = next(c_iter) if is_child else next(a_iter)
        return cls(**kw)

    _PYTREE_TYPES.add(cls)
    jax.tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)
    return cls


def as_jnp(x, dtype=None):
    """Coerce array-like to a jax.Array."""
    if dtype is None:
        return jnp.asarray(x)
    return jnp.asarray(x, dtype=dtype)


def as_np(x, dtype=None):
    """Coerce array-like to a numpy array (for indices / static metadata)."""
    if dtype is None:
        return np.asarray(x)
    return np.asarray(x, dtype=dtype)


__all__ = [
    "Array",
    "NDArray",
    "is_array",
    "pytree_dataclass",
    "as_jnp",
    "as_np",
]
