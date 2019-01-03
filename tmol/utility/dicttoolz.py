from collections.abc import Mapping

from toolz.dicttoolz import (  # noqa
    assoc,
    dissoc,
    assoc_in,
    get_in,
    keyfilter,
    keymap,
    itemfilter,
    itemmap,
    merge,
    merge_with,
    update_in,
    valfilter,
    valmap,
)


def items(d):
    return d.items()


def keys(d):
    return d.keys()


def vals(d):
    return d.vals()


def flat_items(d):
    """Iterate items from potentially nested mapping.

    Iterates [(keys,...): value] items from a potentially nested mapping of
    mappings, where keys is a tuple of key-path leading to value. Traverses all
    collections.abc.Mapping subtypes.
    """

    for k, v in items(d):
        if not isinstance(v, Mapping):
            yield ((k,), v)
        else:
            for sk, v in flat_items(v):
                yield ((k,) + sk, v)


def unflatten(keys_values, factory=dict):
    """Construct potentially-nested mapping from (keys, value) items.

    Construct a potentially-nested mapping from a flat iterator of (keys,
    value) pairs. This is functionally equivalent to reducing items via
    assoc_in.
    """
    if isinstance(keys_values, Mapping):
        keys_values = items(keys_values)

    d = factory()

    for keys, val in keys_values:
        update_inplace(d, keys, lambda _: val, val, factory)

    return d


def update_inplace(d, keys, func, default=None, factory=dict):
    """ Update value in a (potentially) nested dictionary inplace

    inputs:
    d - dictionary on which to operate
    keys - list or tuple giving the location of the value to be changed in d
    func - function to operate on that value

    If keys == [k0,..,kX] and d[k0]..[kX] == v, update_inplace updates the
    original dictionary with v replaced by func(v).

    If k0 is not a key in d, update_inplace creates nested dictionaries to the
    depth specified by the keys, with the innermost value set to func(default).

    Returns d.
    """

    assert len(keys) > 0
    k, ks = keys[0], keys[1:]
    if ks:
        if k not in d:
            d[k] = factory()
        update_inplace(d[k], ks, func, default, factory)
        return d
    else:
        d[k] = func(d[k]) if (k in d) else func(default)
        return d
