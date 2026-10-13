"""Shared YAML -> attrs-dataclass structurer for the scoring database.

Same pattern as :mod:`tmol_jax.database.chemical`: a small pure-Python
recursive walker that matches YAML ``dict`` keys against the field names
of ``@attr.s(frozen=True)`` classes. This keeps us independent of
``cattrs`` registered hooks (which mutate global state) and of
``torch`` at import time.

Supported target types:

* Primitives (``str``, ``int``, ``float``, ``bool``) — value is returned
  as-is (cast to ``float`` if the target is ``float`` but the YAML
  scalar parsed as ``int``).
* ``Optional[T]`` — ``None`` passes through, otherwise recurse into ``T``.
* ``Tuple[T, ...]`` — variadic tuples; each element is structured into
  ``T`` and the whole thing wrapped in a ``tuple``.
* ``Dict[str, V]`` — keys kept as-is, values structured into ``V``.
* Nested attrs-decorated classes — recursively structured using the
  field list from ``__attrs_attrs__``.

A single ``Angle`` scalar type alias is re-exported here; it is a
``NewType`` over ``float`` that accepts unit-qualified strings such as
``"134.5 deg"`` and resolves them to radians.
"""

from __future__ import annotations

import typing
from typing import Any, NewType, Tuple

import attr

from tmol_jax.utility.angle import parse_angle as _parse_angle

# --------------------------------------------------------------------- #
# Angle parsing (re-exported from :mod:`tmol_jax.utility.angle`).       #
# --------------------------------------------------------------------- #

Angle = NewType("Angle", float)


# --------------------------------------------------------------------- #
# Recursive structurer                                                  #
# --------------------------------------------------------------------- #


def _is_attrs_class(t: Any) -> bool:
    return isinstance(t, type) and hasattr(t, "__attrs_attrs__")


def _unwrap_optional(t: Any) -> Tuple[Any, bool]:
    """If ``t`` is ``Optional[X]`` / ``Union[X, None]`` return ``(X, True)``."""
    origin = typing.get_origin(t)
    if origin is typing.Union:
        args = [a for a in typing.get_args(t) if a is not type(None)]  # noqa: E721
        if len(args) == 1:
            return args[0], True
    return t, False


def structure(value: Any, target_type: Any) -> Any:
    """Recursively structure a YAML-decoded ``value`` into ``target_type``.

    This is the single entry point used by every
    ``<term>Database.from_file`` classmethod in this package.
    """

    target_type, _optional = _unwrap_optional(target_type)

    if value is None:
        return None

    # Custom scalar hooks -------------------------------------------------
    if target_type is Angle:
        return _parse_angle(value)

    # Nested attrs dataclasses --------------------------------------------
    if _is_attrs_class(target_type):
        if not isinstance(value, dict):
            raise TypeError(
                f"Expected mapping for {target_type.__name__}, "
                f"got {type(value).__name__}: {value!r}"
            )
        try:
            hints = typing.get_type_hints(target_type)
        except Exception:
            hints = {a.name: a.type for a in target_type.__attrs_attrs__}
        kwargs = {}
        for field in target_type.__attrs_attrs__:
            if field.name in value:
                kwargs[field.name] = structure(
                    value[field.name], hints.get(field.name, field.type)
                )
            elif field.default is not attr.NOTHING:
                # Let attrs apply the field default.
                pass
            else:
                raise KeyError(
                    f"Missing required field {field.name!r} for "
                    f"{target_type.__name__}"
                )
        return target_type(**kwargs)

    if target_type is tuple:
        return tuple(value)

    origin = typing.get_origin(target_type)
    args = typing.get_args(target_type)

    # Tuples --------------------------------------------------------------
    if origin in (tuple, Tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            elem_t = args[0]
            return tuple(structure(v, elem_t) for v in value)
        if len(args) == len(value):
            return tuple(structure(v, a) for v, a in zip(value, args))
        if args:
            return tuple(structure(v, args[0]) for v in value)
        return tuple(value)

    # Lists ---------------------------------------------------------------
    if origin is list:
        elem_t = args[0] if args else Any
        return [structure(v, elem_t) for v in value]

    # Dicts ---------------------------------------------------------------
    if origin is dict:
        val_t = args[1] if len(args) >= 2 else Any
        return {k: structure(v, val_t) for k, v in value.items()}

    # Plain primitives ----------------------------------------------------
    if target_type is float and isinstance(value, int):
        return float(value)
    return value


__all__ = ["Angle", "structure"]
