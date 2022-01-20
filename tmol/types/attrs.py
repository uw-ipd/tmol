"""Mixins for attrs-class type conversion and validation."""

from .converters import get_converter
from .validators import get_validator


class ConvertAttrs:
    def __attrs_post_init__(self):
        for a in self.__attrs_attrs__:
            if not a.converter:
                object.__setattr__(
                    self, a.name, get_converter(a.type)(getattr(self, a.name))
                )


class ValidateAttrs:
    def __attrs_post_init__(self):
        for a in self.__attrs_attrs__:
            if not a.validator:
                try:
                    get_validator(a.type)(getattr(self, a.name))
                except TypeError as e:
                    raise TypeError(
                        "Failed to validate attribute '" + a.name + "': " + str(e)
                    )
