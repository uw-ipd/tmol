"""Mixin components for attrs-based classes."""

import attr

from collections.abc import Mapping, MutableMapping


class AttrMapping(Mapping):
    """Mixin adding Mapping interface to attr classes."""

    def __attrs_keys__(self):
        if not attr.has(self):
            raise TypeError("AttrMapping on non-attr class.")

        return tuple(f.name for f in self.__attrs_attrs__)

    def __iter__(self):
        return iter(self.__attrs_keys__())

    def __getitem__(self, key):
        if key not in self.__attrs_keys__():
            raise KeyError(key)

        return getattr(self, key)

    def __len__(self):
        return len(self.__attrs_keys__())


class AttrMutableMapping(AttrMapping, MutableMapping):
    """Mixin adding a subset of the mutable mapping interface to attr classes.

    As the keys of an attrs-based class are based on defined properties, this mixin
    does *not* support ``__delitem__``-based components of the MutableMapping interface,
    (eg. ``m.pop(key)``, ``del m[key]``, ...)
    """

    def __setitem__(self, key, value):
        if key not in self.__attrs_keys__():
            raise KeyError(key)

        return setattr(self, key, value)

    def __delitem__(self, key):
        raise TypeError("AttrMapping does not support item deletion.")
